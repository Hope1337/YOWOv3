import math

import torch

def pad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1
    if p is None:
        p = k // 2
    return p


def fuse_conv(conv, norm):
    fused_conv = torch.nn.Conv2d(conv.in_channels,
                                 conv.out_channels,
                                 kernel_size=conv.kernel_size,
                                 stride=conv.stride,
                                 padding=conv.padding,
                                 groups=conv.groups,
                                 bias=True).requires_grad_(False).to(conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_norm = torch.diag(norm.weight.div(torch.sqrt(norm.eps + norm.running_var)))
    fused_conv.weight.copy_(torch.mm(w_norm, w_conv).view(fused_conv.weight.size()))

    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_norm = norm.bias - norm.weight.mul(norm.running_mean).div(torch.sqrt(norm.running_var + norm.eps))
    fused_conv.bias.copy_(torch.mm(w_norm, b_conv.reshape(-1, 1)).reshape(-1) + b_norm)

    return fused_conv


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


class Residual(torch.nn.Module):
    def __init__(self, ch, add=True):
        super().__init__()
        self.add_m = add
        self.res_m = torch.nn.Sequential(Conv(ch, ch, 3),
                                         Conv(ch, ch, 3))

    def forward(self, x):
        return self.res_m(x) + x if self.add_m else self.res_m(x)


class CSP(torch.nn.Module):
    def __init__(self, in_ch, out_ch, n=1, add=True):
        super().__init__()
        self.conv1 = Conv(in_ch, out_ch // 2)
        self.conv2 = Conv(in_ch, out_ch // 2)
        self.conv3 = Conv((2 + n) * out_ch // 2, out_ch)
        self.res_m = torch.nn.ModuleList(Residual(out_ch // 2, add) for _ in range(n))

    def forward(self, x):
        y = [self.conv1(x), self.conv2(x)]
        y.extend(m(y[-1]) for m in self.res_m)
        return self.conv3(torch.cat(y, dim=1))


class SPP(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k=5):
        super().__init__()
        self.conv1 = Conv(in_ch, in_ch // 2)
        self.conv2 = Conv(in_ch * 2, out_ch)
        self.res_m = torch.nn.MaxPool2d(k, 1, k // 2)

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.res_m(x)
        y2 = self.res_m(y1)
        return self.conv2(torch.cat([x, y1, y2, self.res_m(y2)], 1))


class DarkNet(torch.nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        p1 = [Conv(width[0], width[1], 3, 2)]
        p2 = [Conv(width[1], width[2], 3, 2),
              CSP(width[2], width[2], depth[0])]
        p3 = [Conv(width[2], width[3], 3, 2),
              CSP(width[3], width[3], depth[1])]
        p4 = [Conv(width[3], width[4], 3, 2),
              CSP(width[4], width[4], depth[2])]
        p5 = [Conv(width[4], width[5], 3, 2),
              CSP(width[5], width[5], depth[0]),
              SPP(width[5], width[5])]

        self.p1 = torch.nn.Sequential(*p1)
        self.p2 = torch.nn.Sequential(*p2)
        self.p3 = torch.nn.Sequential(*p3)
        self.p4 = torch.nn.Sequential(*p4)
        self.p5 = torch.nn.Sequential(*p5)

    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(p1)
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        return p3, p4, p5


class DarkFPN(torch.nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.up = torch.nn.Upsample(None, 2)
        self.h1 = CSP(width[4] + width[5], width[4], depth[0], False)
        self.h2 = CSP(width[3] + width[4], width[3], depth[0], False)
        self.h3 = Conv(width[3], width[3], 3, 2)
        self.h4 = CSP(width[3] + width[4], width[4], depth[0], False)
        self.h5 = Conv(width[4], width[4], 3, 2)
        self.h6 = CSP(width[4] + width[5], width[5], depth[0], False)

    def forward(self, x):
        p3, p4, p5 = x
        h1 = self.h1(torch.cat([self.up(p5), p4], 1))
        h2 = self.h2(torch.cat([self.up(h1), p3], 1))
        h4 = self.h4(torch.cat([self.h3(h2), h1], 1))
        h6 = self.h6(torch.cat([self.h5(h4), p5], 1))
        return h2, h4, h6

class YOLO(torch.nn.Module):
    def __init__(self, width, depth, pretrain_path=None):
        super().__init__()
        self.net = DarkNet(width, depth)
        self.fpn = DarkFPN(width, depth)

        self.pretrain_path = pretrain_path

    def forward(self, x):
        x = self.net(x)
        return self.fpn(x)

    def fuse(self):
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'norm'):
                m.conv = fuse_conv(m.conv, m.norm)
                m.forward = m.fuse_forward
                delattr(m, 'norm')
        return self

    def load_pretrain(self):
        if self.pretrain_path is None:
            return
        state_dict = self.state_dict()

        pretrain_state_dict = torch.load(self.pretrain_path, weights_only=True)
        
        for param_name, value in pretrain_state_dict.items():
            if param_name not in state_dict:
                continue
            state_dict[param_name] = value
            
        self.load_state_dict(state_dict)

        print("backbone2D : YOLOv8 pretrained loaded!", flush=True)
        

#def yolo_v8_n(pretrain_path=config['pretrain_yolov8_n']):
    #depth = [1, 2, 2]
    #width = [3, 16, 32, 64, 128, 256]
    #return YOLO(width, depth, pretrain_path)


#def yolo_v8_s(pretrain_path=config['pretrain_yolov8_s']):
    #depth = [1, 2, 2]
    #width = [3, 32, 64, 128, 256, 512]
    #return YOLO(width, depth, pretrain_path)


#def yolo_v8_m(pretrain_path=config['pretrain_yolov8_m']):
    #depth = [2, 4, 4]
    #width = [3, 48, 96, 192, 384, 576]
    #return YOLO(width, depth, pretrain_path)


#def yolo_v8_l(pretrain_path=config['pretrain_yolov8_l']):
    #depth = [3, 6, 6]
    #width = [3, 64, 128, 256, 512, 512]
    #return YOLO(width, depth, pretrain_path)


#def yolo_v8_x(pretrain_path=config['pretrain_yolov8_x']):
    #depth = [3, 6, 6]
    #width = [3, 80, 160, 320, 640, 640]
    #return YOLO(width, depth, pretrain_path)


def build_yolov8(config):
    ver = config['BACKBONE2D']['YOLOv8']['ver']
    assert ver in ['n', 's', 'm', 'l', 'x'], "wrong version of YOLOv8!"
    pretrain_path = config['BACKBONE2D']['YOLOv8']['PRETRAIN'][ver]

    if ver == 'n':
        depth = [1, 2, 2]
        width = [3, 16, 32, 64, 128, 256]
    elif ver == 's':
        depth = [1, 2, 2]
        width = [3, 32, 64, 128, 256, 512]
    elif ver == 'm':
        depth = [2, 4, 4]
        width = [3, 48, 96, 192, 384, 576]
    elif ver == 'l':
        depth = [3, 6, 6]
        width = [3, 64, 128, 256, 512, 512]
    elif ver == 'x':
        depth = [3, 6, 6]
        width = [3, 80, 160, 320, 640, 640]

    return YOLO(width, depth, pretrain_path)
