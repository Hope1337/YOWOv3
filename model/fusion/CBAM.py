import torch.nn as nn
import torch

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction_rate=16):
        super(ChannelAttention, self).__init__()
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

    def forward(self, x):
        # perform squeeze with independent Pooling
        avg_feat = self.squeeze[0](x)
        max_feat = self.squeeze[1](x)
        # perform excitation with the same excitation sub-net
        avg_out = self.excitation(avg_feat)
        max_out = self.excitation(max_feat)
        # attention
        attention = self.sigmoid(avg_out + max_out)
        return attention * x
    

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # mean on spatial dim
        avg_feat    = torch.mean(x, dim=1, keepdim=True)
        # max on spatial dim
        max_feat, _ = torch.max(x, dim=1, keepdim=True)
        feat = torch.cat([avg_feat, max_feat], dim=1)
        out_feat = self.conv(feat)
        attention = self.sigmoid(out_feat)
        return attention * x
    

class CBAM(nn.Module):
    def __init__(self, in_channels, channels, reduction_rate=16, kernel_size=7):
        super(CBAM, self).__init__()
        assert channels % reduction_rate == 0, "can not reduce channel in CBAM"
        self.channel_attention = ChannelAttention(channels,
                                                  reduction_rate)
        self.spatial_attention = SpatialAttention(kernel_size)
        self.reduceChannels    = nn.Sequential(
            nn.Conv2d(
                in_channels= in_channels,
                out_channels=channels,
                kernel_size=1
            ),
            nn.ReLU()
        )
        
    def forward(self, x):
        x   = self.reduceChannels(x)
        out = self.channel_attention(x)
        out = self.spatial_attention(out)
        
        return out

class CBAMFusion(nn.Module):
    def __init__(self, channels_2D, channels_3D, interchannels, mode='decoupled'):
        super().__init__()
        assert mode in ['decoupled'], "SimpleBlock currently only support for decoupled mode"
        self.mode = mode

        if mode == 'decoupled':
            box = []
            cls = []
            for channels2D in channels_2D:
                box.append(CBAM(channels2D[0] + channels_3D, interchannels))
                cls.append(CBAM(channels2D[1] + channels_3D, interchannels))
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