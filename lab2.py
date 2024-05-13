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

coconf = build_config()
import yaml
import shutil

model = nn.BatchNorm2d(2)
biases = []
not_biases = []

for param_name, param in model.named_parameters():
    if param.requires_grad:
        if param_name.endswith('bias'):
            biases.append(param)
        else:
            not_biases.append(param)
            
print(biases)
print(not_biases)