import torch
import torch.nn as nn
from model.fusion.CFAM import CFAMFusion
from model.head.dfl import DFLHead
from model.backbone3D.build_backbone3D import build_backbone3D
from model.backbone2D.build_backbone2D import build_backbone2D
from model.fusion.build_fusion import build_fusion_block

def pad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1
    if p is None:
        p = k // 2
    return p

class Conv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k=1, s=1, p=None, d=1, g=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, k, s, pad(k, p, d), d, g, False)
        self.norm = torch.nn.BatchNorm2d(out_ch, 0.001, 0.03)
        self.relu = torch.nn.SiLU(inplace=True)

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

    def fuse_forward(self, x):
        return self.relu(self.conv(x))

class DecoupleHead(torch.nn.Module):

    def __init__(self, interchannels, filters=()):
        super().__init__()
        self.nl = len(filters)  # number of detection layers

        self.cls = torch.nn.ModuleList(torch.nn.Sequential(Conv(x, interchannels, 3),
                                                           Conv(interchannels, interchannels, 3)) for x in filters)
        self.box = torch.nn.ModuleList(torch.nn.Sequential(Conv(x, interchannels, 3),
                                                           Conv(interchannels, interchannels, 3)) for x in filters)

    def forward(self, x):
        out = []
        for i in range(self.nl):
            #print(self.box[i](x[i]).shape) [B, 4 * n_dfl_channel, H, W]
            #print(self.cls[i](x[i]).shape) [B, nclass, H, W]
            out.append([self.box[i](x[i]), self.cls[i](x[i])])

        return out

class YOWOv3(torch.nn.Module):
    def __init__(self, num_classes, backbone2D, backbone3D, interchannels, mode, img_size, pretrain_path=None,
                 freeze_bb2D=False, freeze_bb3D=False, fusion_block='CFAM'):
        super().__init__()
        assert mode in ['coupled', 'decoupled']
        self.mode = mode

        self.freeze_bb2D = freeze_bb2D
        self.freeze_bb3D = freeze_bb3D

        self.inter_channels_decoupled = interchannels[0] 
        self.inter_channels_fusion    = interchannels[1]
        self.inter_channels_detection = interchannels[2]

        self.net2D = backbone2D
        self.net3D = backbone3D

        dummy_img3D = torch.zeros(1, 3, 16, img_size, img_size)
        dummy_img2D = torch.zeros(1, 3, img_size, img_size)

        out_2D = self.net2D(dummy_img2D)
        out_3D = self.net3D(dummy_img3D)

        assert out_3D.shape[2] == 1, "output of 3D branch must have D = 1"

        out_channels_2D = [x.shape[1] for x in out_2D]
        out_channels_3D = out_3D.shape[1]

        if self.mode == 'decoupled':
            self.decoupled_head = DecoupleHead(self.inter_channels_decoupled, out_channels_2D)
            # [[box, cls], [box, cls], [box, cls]]
            out_2D = self.decoupled_head(out_2D)
            out_channels_2D = [[x[0].shape[1], x[1].shape[1]] for x in out_2D]    
            last2dimension1 = [[x[0].shape[-1], x[0].shape[-2]] for x in out_2D]
            last2dimension2 = [[x[1].shape[-1], x[1].shape[-2]] for x in out_2D]

        self.fusion = build_fusion_block(out_channels_2D, out_channels_3D, 
                                         self.inter_channels_fusion, mode, fusion_block, 
                                         [last2dimension1, last2dimension2])

        self.detection_head = DFLHead(num_classes, img_size,
                                      self.inter_channels_detection, 
                                      [self.inter_channels_fusion for x in range(len(out_channels_2D))], 
                                      mode=self.mode)
        self.detection_head.stride = torch.tensor([img_size / x[0].shape[-2] for x in out_2D])
        self.stride = self.detection_head.stride

        if pretrain_path is not None:
            self.load_pretrain(pretrain_path)
        else : 
            self.net2D.load_pretrain()
            self.net3D.load_pretrain()
            self.init_conv2d()
            self.detection_head.initialize_biases()
        
        if freeze_bb2D == True:
            for param in self.net2D.parameters():
                param.require_grad = False
            print("backbone2D freezed!")
        
        if freeze_bb3D == True:
            for param in self.net3D.parameters():
                param.require_grad = False
            print("backbone3D freezed!")

    def forward(self, clips):
        key_frames = clips[:, :, -1, :, :]

        ft_2D = self.net2D(key_frames)
        ft_3D = self.net3D(clips).squeeze(2)
        
        if self.mode == 'decoupled':
            ft_2D = self.decoupled_head(ft_2D)

        ft = self.fusion(ft_2D, ft_3D)

        # [B, 4 + num_classes, 1029]
        return self.detection_head(list(ft))
    
    def load_pretrain(self, pretrain_yowov3):
        state_dict = self.state_dict()
        pretrain_state_dict = torch.load(pretrain_yowov3, weights_only=True)
        flag = 0
        
        for param_name, value in pretrain_state_dict.items():
            if param_name not in state_dict:
                if param_name.endswith("total_params") or param_name.endswith("total_ops"):
                    continue
                flag = 1
                continue
            state_dict[param_name] = value

        try:
            self.load_state_dict(state_dict)
        except:
            flag = 1

        if flag == 1:
            print("WARNING !")
            print("########################################################################")
            print("There are some tensors in the model that do not match the checkpoint.") 
            print("The model automatically ignores them for the purpose of fine-tuning.") 
            print("Please ensure that this is your intention.")
            print("########################################################################")
            print()
            self.detection_head.initialize_biases()
    
    def init_conv2d(self):
        """
        Initialize convolution parameters.
        """

        #if self.mode == 'decoupled':
            #for c in self.decoupled_head.modules():
                #if isinstance(c, nn.Conv2d):
                    #nn.init.kaiming_normal_(c.weight)
                    #if c.bias is not None:
                        #nn.init.constant_(c.bias, 0.)

        #for c in self.fusion.modules():
            #if isinstance(c, nn.Conv2d):
                #nn.init.kaiming_normal_(c.weight)
                #if c.bias is not None:
                    #nn.init.constant_(c.bias, 0.)

        #for c in self.detection_head.modules():
            #if isinstance(c, nn.Conv2d) and c is not self.detection_head.dfl.conv:
                #nn.init.kaiming_normal_(c.weight)
                #if c.bias is not None:
                    #nn.init.constant_(c.bias, 0.)

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03


def build_yowov3(config):
    num_classes   = config['num_classes']
    backbone2D    = build_backbone2D(config)
    backbone3D    = build_backbone3D(config)
    interchannels = config['interchannels']
    mode          = config['mode']
    pretrain_path = config['pretrain_path']
    img_size      = config['img_size']
    fusion_block  = config['fusion_module']

    try:
        freeze_bb2D   = config['freeze_bb2D']
        freeze_bb3D   = config['freeze_bb3D']
    except:
        freeze_bb2D = False
        freeze_bb3D = False

    return YOWOv3(num_classes, backbone2D, backbone3D, interchannels, mode, img_size, pretrain_path,
                  freeze_bb2D, freeze_bb3D, fusion_block)