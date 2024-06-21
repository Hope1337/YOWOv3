import torch
from thop import profile
from model.TSN.YOWOv3 import build_yowov3

def FLOPs_and_Params(model, img_size, len_clip, device):
    # generate init video clip
    model.to(device)
    video_clip = torch.randn(1, 3, len_clip, img_size, img_size).to(device)

    # set eval mode
    model.trainable = False
    model.eval()

    flops, params = profile(model, inputs=(video_clip, ))

    return flops, params
    


def get_info(config):
    model = build_yowov3(config)

    flops, params = FLOPs_and_Params(model, config['img_size'], 16, device='cuda')

    print('==============================')
    print('FLOPs : {:.2f} G'.format(flops / 1e9))
    print('Params : {:.2f} M'.format(params / 1e6))
    print('==============================')

if __name__ == "__main__":
    pass