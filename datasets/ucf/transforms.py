from . import ucf_config
import torch
import numpy as np

class UCF_transform():
    """
    Args:
        clip  : list of (num_frame) np.array [H, W, C] (BGR order, 0..1)
        boxes : list of (num_frame) list of (num_box, in ucf101-24 = 1) np.array [(x, y, w, h)] relative coordinate
    
    Return:
        clip  : torch.tensor [C, num_frame, H, W] (RGB order, 0..1)
        boxes : not change
    """

    def __init__(self):
        pass

    def to_tensor(self, video_clip):
        return [F.to_tensor(image) for image in video_clip]

    def normalize(self, clip, mean=ucf_config.MEAN, std=ucf_config.STD):
        mean  = torch.FloatTensor([0.485, 0.456, 0.406]).view(-1, 1, 1, 1)
        std   = torch.FloatTensor([0.229, 0.224, 0.225]).view(-1, 1, 1, 1)
        clip -= mean
        clip /= std
        return clip
    
    def __call__(self, clip, targets):
        W, H = clip[-1].size
        targets[:, :4] /= np.array([W, H, W, H])
        clip = [img.resize([224, 224]) for img in clip]
        clip = self.to_tensor(clip)
        clip = torch.stack(clip, dim=1)
        clip = self.normalize(clip)
        targets = torch.as_tensor(targets).float()
        return clip, targets

import torch
import torch.utils.data as data
import argparse
import yaml
import os
import cv2
import pickle
import numpy as np
from PIL import Image
import sys

import random
import torchvision.transforms.functional as F

class Augmentation(object):
    def __init__(self, img_size=224, jitter=0.2, hue=0.1, saturation=1.5, exposure=1.5):
        self.img_size = img_size
        self.jitter = jitter
        self.hue = hue
        self.saturation = saturation
        self.exposure = exposure


    def rand_scale(self, s):
        scale = random.uniform(1, s)

        if random.randint(0, 1): 
            return scale

        return 1./scale


    def random_distort_image(self, video_clip):
        dhue = random.uniform(-self.hue, self.hue)
        dsat = self.rand_scale(self.saturation)
        dexp = self.rand_scale(self.exposure)
        
        video_clip_ = []
        for image in video_clip:
            image = image.convert('HSV')
            cs = list(image.split())
            cs[1] = cs[1].point(lambda i: i * dsat)
            cs[2] = cs[2].point(lambda i: i * dexp)
            
            def change_hue(x):
                x += dhue * 255
                if x > 255:
                    x -= 255
                if x < 0:
                    x += 255
                return x

            cs[0] = cs[0].point(change_hue)
            image = Image.merge(image.mode, tuple(cs))

            image = image.convert('RGB')

            video_clip_.append(image)

        return video_clip_


    def random_crop(self, video_clip, width, height):
        dw =int(width * self.jitter)
        dh =int(height * self.jitter)

        pleft  = random.randint(-dw, dw)
        pright = random.randint(-dw, dw)
        ptop   = random.randint(-dh, dh)
        pbot   = random.randint(-dh, dh)

        swidth =  width - pleft - pright
        sheight = height - ptop - pbot

        sx = float(swidth)  / width
        sy = float(sheight) / height
        
        dx = (float(pleft) / width)/sx
        dy = (float(ptop) / height)/sy

        # random crop
        cropped_clip = [img.crop((pleft, ptop, pleft + swidth - 1, ptop + sheight - 1)) for img in video_clip]

        return cropped_clip, dx, dy, sx, sy


    def apply_bbox(self, target, ow, oh, dx, dy, sx, sy):
        sx, sy = 1./sx, 1./sy
        # apply deltas on bbox
        target[..., 0] = np.minimum(0.999, np.maximum(0, target[..., 0] / ow * sx - dx)) 
        target[..., 1] = np.minimum(0.999, np.maximum(0, target[..., 1] / oh * sy - dy)) 
        target[..., 2] = np.minimum(0.999, np.maximum(0, target[..., 2] / ow * sx - dx)) 
        target[..., 3] = np.minimum(0.999, np.maximum(0, target[..., 3] / oh * sy - dy)) 

        # refine target
        refine_target = []
        for i in range(target.shape[0]):
            tgt = target[i]
            bw = (tgt[2] - tgt[0]) * ow
            bh = (tgt[3] - tgt[1]) * oh

            if bw < 1. or bh < 1.:
                continue
            
            refine_target.append(tgt)

        refine_target = np.array(refine_target).reshape(-1, target.shape[-1])

        return refine_target
        

    def to_tensor(self, video_clip):
        return [F.to_tensor(image) for image in video_clip]
    
    def normalization(self, video_clip):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)

        video_clip = (video_clip - mean)/std
        return video_clip


    def __call__(self, video_clip, target):
        # Initialize Random Variables
        oh = video_clip[-1].height  
        ow = video_clip[-1].width
        
        # random crop
        video_clip, dx, dy, sx, sy = self.random_crop(video_clip, ow, oh)

        # resize
        video_clip = [img.resize([self.img_size, self.img_size]) for img in video_clip]

        # random flip
        flip = random.randint(0, 1)
        if flip:
            video_clip = [img.transpose(Image.Transpose.FLIP_LEFT_RIGHT) for img in video_clip]

        # distort
        video_clip = self.random_distort_image(video_clip)

        # process target
        if target is not None:
            target = self.apply_bbox(target, ow, oh, dx, dy, sx, sy)
            if flip:
                target[..., [0, 2]] = 1.0 - target[..., [2, 0]]
        else:
            target = np.array([])
            
        # to tensor
        video_clip = self.to_tensor(video_clip)
        video_clip = torch.stack(video_clip, dim=1)
        
        video_clip = self.normalization(video_clip)
        
        target = torch.as_tensor(target).float()

        return video_clip, target 

def UCF_collate_fn(batch_data):
    clips  = []
    boxes  = []
    labels = []
    for b in batch_data:
        clips.append(b[0])
        boxes.append(b[1])
        labels.append(b[2])
    
    clips = torch.stack(clips, dim=0) # [batch_size, num_frame, C, H, W]
    return clips, boxes, labels


