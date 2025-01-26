import torch.nn as nn
import torch

class SEModule(nn.Module):
    def __init__(self, channels, interchannels):
        super(SEModule, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(in_channels=channels,
                      out_channels=interchannels,
                      kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=interchannels,
                      out_channels=interchannels,
                      kernel_size=1),
            nn.Sigmoid()
        )
        self.reduce_channels = nn.Sequential(
            nn.Conv2d(in_channels=channels,
                      out_channels=interchannels,
                      kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        attention = self.squeeze(x)
        attention = self.excitation(attention)
        x         = self.reduce_channels(x)

        return attention * x
    

class SEFusion(nn.Module):
    def __init__(self, channels_2D, channels_3D, interchannels, mode='decoupled'):
        super().__init__()
        assert mode in ['decoupled'], "SEFusion currently only support for decoupled mode"
        self.mode = mode

        if mode == 'decoupled':
            box = []
            cls = []
            for channels2D in channels_2D:
                box.append(SEModule(channels2D[0] + channels_3D, interchannels))
                cls.append(SEModule(channels2D[1] + channels_3D, interchannels))

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