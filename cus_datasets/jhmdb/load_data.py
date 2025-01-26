import torch
import torch.utils.data as data
import argparse
import yaml
import os
import cv2
import pickle
import numpy as np
from cus_datasets.ucf.transforms import Augmentation, UCF_transform
from PIL import Image
import csv

class JHMDB_dataset(data.Dataset):

    def __init__(self, root_path, phase, split_path, 
                 clip_length, sampling_rate, img_size, transform=Augmentation()):
        self.root_path     = root_path
        self.phase         = phase
        self.img_folder    = os.path.join(root_path, "Images")
        self.ann_folder    = os.path.join(root_path, "Annotations")
        self.split_path    = os.path.join(root_path, split_path)
        self.transform     = transform
        self.clip_length   = clip_length 
        self.sampling_rate = sampling_rate
        self.img_size      = img_size
        self.data_path_list = []
        self.data_len   = 0

        self.build_path_list()
        pass

    def build_path_list(self): 
        with open(self.split_path, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                line  = line.rstrip()
                self.data_path_list.append(line)
        self.data_len = len(self.data_path_list)

    def read_clip(self, path):
        clip        = []
        splited     = path.split('/')
        key_idx     = int(splited[-1].split('.')[0])
        splited     = splited[:-1]
        path_wo_idx = os.path.join(self.img_folder, os.path.join(*splited))

        for i in reversed(range(self.clip_length)):
            cur_frame_idx = key_idx - i*self.sampling_rate
            if cur_frame_idx < 1:
                cur_frame_idx = 1
            
            path_w_idx = os.path.join(path_wo_idx, '{:05d}.png'.format(cur_frame_idx))
            cur_frame  = Image.open(path_w_idx).convert('RGB')
            clip.append(cur_frame)

        return clip

    def read_ann(self, path):
        t = path.split('/')
        path = os.path.join(self.ann_folder, t[0] + '_' + t[1] + '_' + t[2])

        labels = []
        boxes  = []

        with open(path, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                line  = line.rstrip().split(' ')
                label = int(line[0]) - 1

                if self.phase == 'train':
                    onehot_vector = np.zeros(21)
                    onehot_vector[label] = 1.
                    labels.append(onehot_vector)
                elif self.phase == 'test':
                    labels.append(label)
                
                box = [float(line[x]) for x in range(1, len(line))]
                boxes.append(box)
        
        boxes = np.array(boxes)

        if self.phase == 'train':
            labels = np.array(labels)
        elif self.phase == 'test':
            labels = np.expand_dims(np.array(labels), axis=1)

        return boxes, labels
    
    def read_key_frame(self, path):
        path = path.split('.')[0] + '.png'
        path = os.path.join(self.img_folder, path)
        return cv2.imread(path)

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx, get_origin_image=False):
        path           = self.data_path_list[idx]
        clip           = self.read_clip(path)
        boxes, labels  = self.read_ann(path)

        targets       = np.concatenate((boxes, labels), axis=1)
        clip, targets = self.transform(clip, targets)

        boxes         = targets[:, :4]
        
        if self.phase == 'train':
            labels = targets[:, 4:]
        elif self.phase == 'test':
            labels = targets[:, -1]

        if get_origin_image == True:
            origin_image = self.read_key_frame(path) 
            return origin_image, clip, boxes, labels
        else:
            return clip, boxes, labels




def build_jhmdb_dataset(config, phase):
    root_path     = config['data_root']
    clip_length   = config['clip_length']
    sampling_rate = config['sampling_rate']
    img_size      = config['img_size']

    if phase == 'train':
        split_path = 'trainlist.txt'
        return JHMDB_dataset(root_path, phase, split_path, clip_length,
                             sampling_rate, img_size, 
                             transform=Augmentation(img_size=img_size))
    elif phase == 'test':
        split_path = 'testlist.txt'
        return JHMDB_dataset(root_path, phase, split_path, clip_length,
                             sampling_rate, img_size, 
                             transform=UCF_transform(img_size=img_size))
    

if __name__ == '__main__':
    root_path = '/home/manh/Datasets/jhmdb'
    clip_length   = 16
    sampling_rate = 1
    img_size      = 224

    dataset = JHMDB_dataset(root_path, 'test', 'trainlist.txt', clip_length,
                             sampling_rate, img_size, 
                             transform=UCF_transform(img_size=img_size))


    for i in range(dataset.__len__()):
        origin_image, clip, boxes, label = dataset.__getitem__(i, True)
        print(i)