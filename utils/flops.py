import torch
from thop import profile
from model.TSN.YOWOv3 import build_yowov3

def get_info(config, model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    video_clip = torch.randn(1, 3, config['clip_length'], config['img_size'], config['img_size']).to(device)

    # set eval mode
    model.trainable = False
    model.eval()

    flops, params = profile(model, inputs=(video_clip, ), verbose=False)

    print('==============================')
    print('FLOPs : {:.2f} G'.format(flops / 1e9))
    print('Params : {:.2f} M'.format(params / 1e6))
    print('==============================')

    model.trainable = True

if __name__ == "__main__":
    pass