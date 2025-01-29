from .CFAM import CFAMFusion
from .SE import SEFusion
from .Simple import SimpleFusion
from .MultiHead import MultiHeadFusion
from .Channel import ChannelFusion
from .Spatial import SpatialFusion
from .CBAM import CBAMFusion
from .LKA import LKAFusion

# major of Attention are adopted from https://viblo.asia/p/mot-chut-ve-co-che-attention-trong-computer-vision-x7Z4D622LnX
def build_fusion_block(out_channels_2D, out_channels_3D, inter_channels_fusion, mode, fusion_block, lastdimension):
    if fusion_block == 'CFAM':
        return CFAMFusion(out_channels_2D, out_channels_3D, inter_channels_fusion, mode)
    elif fusion_block == 'SE':
        return SEFusion(out_channels_2D, out_channels_3D, inter_channels_fusion, mode)
    elif fusion_block == 'Simple':
        return SimpleFusion(out_channels_2D, out_channels_3D, inter_channels_fusion, mode)
    elif fusion_block == 'MultiHead':
        return MultiHeadFusion(out_channels_2D, out_channels_3D, inter_channels_fusion, lastdimension, mode, h=1)
    elif fusion_block == 'Channel':
        return ChannelFusion(out_channels_2D, out_channels_3D, inter_channels_fusion, mode)
    elif fusion_block == 'Spatial':
        return SpatialFusion(out_channels_2D, out_channels_3D, inter_channels_fusion, mode)
    elif fusion_block == 'CBAM':
        return CBAMFusion(out_channels_2D, out_channels_3D, inter_channels_fusion, mode)
    elif fusion_block == 'LKA':
        return LKAFusion(out_channels_2D, out_channels_3D, inter_channels_fusion, mode)