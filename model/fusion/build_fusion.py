from .CFAM import CFAMFusion
from .SE import SEFusion
from .Simple import SimpleFusion
from .MultiHead import MultiHeadFusion

def build_fusion_block(out_channels_2D, out_channels_3D, inter_channels_fusion, mode, fusion_block, lastdimension):
    if fusion_block == 'CFAM':
        return CFAMFusion(out_channels_2D, out_channels_3D, inter_channels_fusion, mode)
    elif fusion_block == 'SE':
        return SEFusion(out_channels_2D, out_channels_3D, inter_channels_fusion, mode)
    elif fusion_block == 'Simple':
        return SimpleFusion(out_channels_2D, out_channels_3D, inter_channels_fusion, mode)
    elif fusion_block == 'MultiHead':
        return MultiHeadFusion(out_channels_2D, out_channels_3D, inter_channels_fusion, lastdimension, mode, h=7)