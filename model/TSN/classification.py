import torch
import torch.nn as nn
from model.fusion.CFAM import CFAMFusion
from model.head.dfl import DFLHead
from model.backbone3D.build_backbone3D import build_backbone3D
from model.backbone2D.build_backbone2D import build_backbone2D

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
    
class ClassificationHead(nn.Module):
    def __init__(self, shape, num_classes):
        super().__init__()
        self.predict = nn.Conv2d(shape[0], num_classes, kernel_size=(shape[1], shape[2]))

    def forward(self, x):
        return self.predict(x)

class ClassificationModel(torch.nn.Module):
    def __init__(self, num_classes, backbone2D, backbone3D, interchannels, mode, img_size, pretrain_path=None,
                 freeze_bb2D=False, freeze_bb3D=False):
        super().__init__()
        assert mode in ['coupled']
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

        self.fusion = CFAMFusion(out_channels_2D, 
                                 out_channels_3D, 
                                 self.inter_channels_fusion, 
                                 mode=self.mode)
        
        #for x in out_2D:
            #print(x.shape)
        
        self.head   = ClassificationHead([self.inter_channels_fusion, out_2D[-1].shape[2], out_2D[-1].shape[3]], num_classes)

        if pretrain_path is not None:
            self.load_pretrain(pretrain_path)
        else : 
            self.net2D.load_pretrain()
            self.net3D.load_pretrain()
            self.init_conv2d()
        
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
        

        ft = self.fusion(ft_2D, ft_3D)

        # [B, 4 + num_classes, 1029]
        #import sys
        #sys.exit()
        return self.head(ft[-1]).squeeze(-1).squeeze(-1)
    
    def load_pretrain(self, pretrain_yowov3):
        state_dict = self.state_dict()
        pretrain_state_dict = torch.load(pretrain_yowov3)
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
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03


def build_classificationmodel(config):
    num_classes   = config['num_classes']
    backbone2D    = build_backbone2D(config)
    backbone3D    = build_backbone3D(config)
    interchannels = config['interchannels']
    mode          = config['mode']
    pretrain_path = config['pretrain_path']
    img_size      = config['img_size']

    try:
        freeze_bb2D   = config['freeze_bb2D']
        freeze_bb3D   = config['freeze_bb3D']
    except:
        freeze_bb2D = False
        freeze_bb3D = False

    return ClassificationModel(num_classes, backbone2D, backbone3D, interchannels, mode, img_size, pretrain_path,
                  freeze_bb2D, freeze_bb3D)

from model.backbone2D.YOLOv8 import YOLO
from model.backbone3D.i3d import InceptionI3d
if __name__ == "__main__":
    num_classes   = 2
    depth = [3, 6, 6]
    width = [3, 80, 160, 320, 640, 640]
    
    backbone2D    = YOLO(width, depth)
    backbone3D    = InceptionI3d(in_channels=3, pretrain_path=None)
    interchannels = [256, 256, 256]
    mode          = 'coupled'
    pretrain_path = None
    img_size      = 224

    try:
        freeze_bb2D   = config['freeze_bb2D']
        freeze_bb3D   = config['freeze_bb3D']
    except:
        freeze_bb2D = False
        freeze_bb3D = False

    model = ClassificationModel(num_classes, backbone2D, backbone3D, interchannels, mode, img_size, pretrain_path,
                  freeze_bb2D, freeze_bb3D)
    
    clip = torch.zeros((8, 3, 16, 224, 224))
    out  = model(clip)
    print(out.shape)