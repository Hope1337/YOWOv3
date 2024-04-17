from model.backbone2D import YOLOv8

def build_backbone2D(config):
    backbone_2D = config['backbone2D']

    if backbone_2D == 'yolov8':
        backbone2D = YOLOv8.build_yolov8(config)
    
    return backbone2D