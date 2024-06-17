'''MobileNet in PyTorch.

See the paper "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv3d(inp, oup, kernel_size=3, stride=stride, padding=(1,1,1), bias=False),
        nn.BatchNorm3d(oup),
        nn.ReLU(inplace=True)
    )


class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm3d(in_planes)
        self.conv2 = nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm3d(out_planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out


class MobileNet(nn.Module):
    def __init__(self, width_mult=1., pretrain_path=None):
        super(MobileNet, self).__init__()

        input_channel = 32
        last_channel = 1024
        input_channel = int(input_channel * width_mult)
        last_channel = int(last_channel * width_mult)
        cfg = [
        # c, n, s
        [64,   1, (2,2,2)],
        [128,  2, (2,2,2)],
        [256,  2, (2,2,2)],
        [512,  6, (2,2,2)],
        [1024, 2, (1,1,1)],
        ]

        self.features = [conv_bn(3, input_channel, (1,2,2))]
        # building inverted residual blocks
        for c, n, s in cfg:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(Block(input_channel, output_channel, stride))
                input_channel = output_channel
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)
        self.avgpool = nn.AvgPool3d((2, 1, 1), stride=1)
        self.pretrain_path = pretrain_path

    def forward(self, x):
        x = self.features(x)

        if x.size(2) == 2:
            x = self.avgpool(x)
        
        return x
    
    def load_pretrain(self):
        
        state_dict = self.state_dict()

        pretrain_state_dict = torch.load(self.pretrain_path)

        for param_name, value in pretrain_state_dict['state_dict'].items():
            param_name = param_name.split('.', 1)[1] # param_name has 'module' at first!
            
            if param_name not in state_dict:
                continue
            state_dict[param_name] = value
            
        self.load_state_dict(state_dict)

        print("backbone3D : mobilenet pretrained loaded!", flush=True)


def get_fine_tuning_parameters(model, ft_portion):
    if ft_portion == "complete":
        return model.parameters()

    elif ft_portion == "last_layer":
        ft_module_names = []
        ft_module_names.append('classifier')

        parameters = []
        for k, v in model.named_parameters():
            for ft_module in ft_module_names:
                if ft_module in k:
                    parameters.append({'params': v})
                    break
            else:
                parameters.append({'params': v, 'lr': 0.0})
        return parameters

    else:
        raise ValueError("Unsupported ft_portion: 'complete' or 'last_layer' expected")
    

def get_model(**kwargs):
    """
    Returns the model.
    """
    model = MobileNet(**kwargs)
    return model

def build_mobilenet(config):
    width_mult = config['BACKBONE3D']['MOBILENET']['width_mult']
    assert width_mult in [0.5, 1.0, 1.5, 2.0], "wrong width_mult of mobilenet!"
    pretrain_dict = config['BACKBONE3D']['MOBILENET']['PRETRAIN']
            
    if width_mult == 0.5:
        pretrain_path = pretrain_dict['width_mult_0.5x']
    elif width_mult == 1.0:
        pretrain_path = pretrain_dict['width_mult_1.0x']
    elif width_mult == 1.5:
        pretrain_path = pretrain_dict['width_mult_1.5x']
    elif width_mult == 2.0:
        pretrain_path = pretrain_dict['width_mult_2.0x']

    return MobileNet(width_mult=width_mult, pretrain_path=pretrain_path)

if __name__ == '__main__':
    model = get_model(width_mult=1.)
    model = model.cuda()
    model = nn.DataParallel(model, device_ids=None)
    print(model)

    input_var = Variable(torch.randn(8, 3, 16, 112, 112))
    output = model(input_var)
    print(output.shape)
