
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

from datasets.build_dataset import build_dataset
from utils.box import draw_bounding_box
from utils.box import non_max_suppression
from model.TSN.YOWOv3 import build_yowov3
from utils.build_config import build_config
from PIL import Image
from utils.flops import get_info

class live_transform():
    """
    Args:
        clip  : list of (num_frame) np.array [H, W, C] (BGR order, 0..1)
        boxes : list of (num_frame) list of (num_box, in ucf101-24 = 1) np.array [(x, y, w, h)] relative coordinate
    
    Return:
        clip  : torch.tensor [C, num_frame, H, W] (RGB order, 0..1)
        boxes : not change
    """

    def __init__(self, img_size):
        self.img_size = img_size
        pass

    def to_tensor(self, image):
        return FT.to_tensor(image)
    
    def normalize(self, clip, mean=[0.4345, 0.4051, 0.3775], std=[0.2768, 0.2713, 0.2737]):
        mean  = torch.FloatTensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std   = torch.FloatTensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        clip -= mean
        clip /= std
        return clip
    
    def __call__(self, img):
        W, H = img.size
        img = img.resize([self.img_size, self.img_size])
        img = self.to_tensor(img)
        img = self.normalize(img)

        return img

from model.TSN.classification import build_classificationmodel

def detect(config):

    model   = build_classificationmodel(config) 
    model.to("cuda")
    model.eval()

    #FLOPs_and_Params(model, 224, 16, 'cuda')
    cap = cv2.VideoCapture(0) 

    frame_list = []
    transform = live_transform(config['img_size'])

    while True:
    # Đọc frame ảnh từ camera
        ret, frame = cap.read()

        origin_image = Image.fromarray(frame)
        frame_list.append(transform(origin_image))
        if (len(frame_list) > 16):
            frame_list.pop(0)
        if (len(frame_list) < 16):
            continue

        clip = torch.stack(frame_list, 0).permute(1, 0, 2, 3).contiguous()

        clip = clip.unsqueeze(0).to("cuda")
        outputs = model(clip)

        origin_image = frame
        origin_image = cv2.resize(origin_image, (config['img_size'], config['img_size']))

        #origin_image = cv2.resize(origin_image, (512, 512))
        print(outputs.item())
        cv2.imshow('img', origin_image)
        k = cv2.waitKey(1)
        if k == ord('q'):
            break
         

if __name__ == "__main__":
    config = build_config('config/rwf2000_config.yaml')
    detect(config)