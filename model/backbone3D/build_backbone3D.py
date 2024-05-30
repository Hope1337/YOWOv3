from model.backbone3D import resnext, resnet, mobilenet, mobilenetv2, shufflenet, shufflenetv2, i3d

def build_backbone3D(config):
    backbone_3D = config['backbone3D']

    if backbone_3D == 'resnext101':
        backbone3D = resnext.resnext101(config)
    if backbone_3D == 'resnet':
        backbone3D = resnet.build_resnet(config)
    elif backbone_3D == 'mobilenet':
        backbone3D = mobilenet.build_mobilenet(config)
    elif backbone_3D == 'mobilenetv2':
        backbone3D = mobilenetv2.build_mobilenetv2(config)
    elif backbone_3D == 'shufflenet':
        backbone3D = shufflenet.build_shufflenet(config)
    elif backbone_3D == 'shufflenetv2':
        backbone3D = shufflenetv2.build_shufflenetv2(config)
    elif backbone_3D == 'i3d':
        backbone3D = i3d.build_i3d(config)

    return backbone3D