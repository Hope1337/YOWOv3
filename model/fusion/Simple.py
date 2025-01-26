import torch.nn as nn
import torch

class SimpleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        interchannels = out_channels

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, interchannels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(interchannels, interchannels, 1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class SimpleFusion(nn.Module):
    def __init__(self, channels_2D, channels_3D, interchannels, mode='decoupled'):
        super().__init__()
        assert mode in ['decoupled'], "SimpleBlock currently only support for decoupled mode"
        self.mode = mode

        if mode == 'decoupled':
            box = []
            cls = []
            for channels2D in channels_2D:
                box.append(SimpleBlock(channels2D[0] + channels_3D, interchannels))
                cls.append(SimpleBlock(channels2D[1] + channels_3D, interchannels))
            self.box = nn.ModuleList(box)
            self.cls = nn.ModuleList(cls)

    def forward(self, ft_2D, ft_3D):
        _, C_3D, H_3D, W_3D = ft_3D.shape

        fts = []

        if self.mode == 'decoupled':
            for idx, ft2D in enumerate(ft_2D):
                _, C_2D, H_2D, W_2D = ft2D[0].shape
                assert H_2D/H_3D == W_2D/W_3D, "can't upscale"

                upsampling = nn.Upsample(scale_factor=H_2D/H_3D)
                ft_3D_t = upsampling(ft_3D)
                ft_box = torch.cat((ft2D[0], ft_3D_t), dim = 1)
                ft_cls = torch.cat((ft2D[1], ft_3D_t), dim = 1)
                fts.append([self.box[idx](ft_box), self.cls[idx](ft_cls)])
        
        return fts