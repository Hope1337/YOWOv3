import torch.nn as nn
import torch 

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, channels, reduction_rate=16):
        super(ChannelAttention, self).__init__()
        assert channels % reduction_rate == 0, "output channels is not divisible by reduction rate, ChannelAtt"
        self.squeeze = nn.ModuleList([
            nn.AdaptiveAvgPool2d(1),
            nn.AdaptiveMaxPool2d(1)
        ])
        self.excitation = nn.Sequential(
            nn.Conv2d(in_channels=channels,
                      out_channels=channels // reduction_rate,
                      kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels // reduction_rate,
                      out_channels=channels,
                      kernel_size=1)
        )
        self.sigmoid = nn.Sigmoid()

        self.reduceChannels = nn.Sequential(
            nn.Conv2d(
                in_channels =in_channels,
                out_channels=channels,
                kernel_size =1
            ),
            nn.ReLU()
        )

    def forward(self, x):
        x        = self.reduceChannels(x)
        # perform squeeze with independent Pooling
        avg_feat = self.squeeze[0](x)
        max_feat = self.squeeze[1](x)
        # perform excitation with the same excitation sub-net
        avg_out = self.excitation(avg_feat)
        max_out = self.excitation(max_feat)
        # attention
        attention = self.sigmoid(avg_out + max_out)
        return attention * x

class ChannelFusion(nn.Module):
    def __init__(self, channels_2D, channels_3D, interchannels, mode='decoupled'):
        super().__init__()
        assert mode in ['decoupled'], "ChannelFusion currently only support for decoupled mode"
        self.mode = mode

        if mode == 'decoupled':
            box = []
            cls = []
            for channels2D in channels_2D:
                box.append(ChannelAttention(channels2D[0] + channels_3D, interchannels))
                cls.append(ChannelAttention(channels2D[1] + channels_3D, interchannels))
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