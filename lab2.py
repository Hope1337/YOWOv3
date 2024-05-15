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
model = build_yolo2stream(config)

g = [], [], []  # optimizer parameter groups
bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
for v in model.modules():
    for p_name, p in v.named_parameters(recurse=0):
        if p_name == "bias":  # bias (no decay)
            g[2].append(p)
        elif p_name == "weight" and isinstance(v, bn):  # weight (no decay)
            g[1].append(p)
        else:
            g[0].append(p)  # weight (with decay)

optimizer = torch.optim.AdamW(g[0], lr=config['lr'], weight_decay=config['weight_decay'])
optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})  
optimizer.add_param_group({"params": g[2], "weight_decay": 0.0}) 


for j, y in enumerate(optimizer.param_groups):
    print(y)
    time.sleep(10)