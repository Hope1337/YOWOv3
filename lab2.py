import torch
import torch.utils.data as data
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as FT
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt
import time
import xml.etree.ElementTree as ET
import os
import cv2
import random
import sys
import glob

from math import sqrt
from utils.gradflow_check import plot_grad_flow
from utils.EMA import EMA
import logging
from utils.build_config import build_config
from datasets.ucf.load_data import UCF_dataset
from datasets.collate_fn import collate_fn
from datasets.build_dataset import build_dataset
from model.TSN.YOLO2Stream import build_yolo2stream
from utils.loss import build_loss
from utils.warmup_lr import LinearWarmup
import tqdm

config = build_config()
#dataset = build_dataset(config, phase='train')

#cnt = torch.zeros(80)
#print(cnt.shape)
#for idx in tqdm.tqdm(range(dataset.__len__())):
    #clip, box, label = dataset.__getitem__(idx)
    
    #for t in label:
        #cnt[t.bool()] += 1

#for x in cnt:
    #print(x)

#print(config['train_class_ratio'][31] + 1)

#print(type(config['LOSS']['SIMOTA']['dynamic_k']))

#config = build_config()
#state_dict = torch.load(config['pretrain_path'])
#check_point_dict = {'config' : config, 'state_dict' : state_dict}
#torch.save(check_point_dict, 'ckpt.pth')

#check_point_dict = torch.load('ckpt.pth')
#print(check_point_dict['config'])

config = build_config()
from datasets.ucf.load_data import UCF_dataset
from datasets.collate_fn import collate_fn
from datasets.build_dataset import build_dataset
from model.TSN.YOLO2Stream import build_yolo2stream
from utils.loss import build_loss


import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap

import torch
from thop import profile

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
    model = build_yolo2stream(config)

    flops, params = FLOPs_and_Params(model, 224, 16, device='cuda')

    print('==============================')
    print('FLOPs : {:.2f} G'.format(flops / 1e9))
    print('Params : {:.2f} M'.format(params / 1e6))
    print('==============================')

config = build_config()
get_info(config)