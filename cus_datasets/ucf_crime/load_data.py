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

class UCFCrime_dataset(data.Dataset):

    def __init__(self, root_path, phase, split_path,
                 clip_length, sampling_rate, img_size, 
                 name2idx, transform=Augmentation()):
        
        self.root_path      = root_path
        self.phase          = phase
        self.img_folder     = os.path.join(root_path, "Anomaly/Images")
        self.ann_folder     = os.path.join(root_path, "Annotations")
        self.split_path     = os.path.join(self.ann_folder, split_path)
        self.transform      = transform
        self.clip_length    = clip_length
        self.sampling_rate  = sampling_rate
        self.img_size       = img_size
        self.name2idx       = name2idx
        self.data_list      = []
        self.data_len       = 0

        self.build_path_list()
        pass
    
    def build_path_list(self):
        with open(self.split_path, 'rb') as f:
            data = pickle.load(f)

            for name in data:
                for x in data[name]:
                    line = []
                    path = x[0].split('/')[5:]
                    path = '/'.join(path)

                    line = [path, x[1], x[2], x[3], x[4]]
                    self.data_list.append(line)
        self.data_len = len(self.data_list)

    def read_clip(self, path):
        clip      = []
        path      = os.path.join(self.img_folder, path[0])
        splited   = path.split('/')
        key_idx   = int(splited[-1].split('_')[1].split('.')[0])
        splited   = splited[:-1]
        path_wo_idx = '/' + os.path.join(*splited)
        #print(path_wo_idx)

        for i in reversed(range(self.clip_length)):
            cur_frame_idx = key_idx - i*self.sampling_rate
            if cur_frame_idx < 1:
                cur_frame_idx = 1
            
            path_w_idx = os.path.join(path_wo_idx, 'image_{:04d}.jpg'.format(cur_frame_idx))
            
            cur_frame  = Image.open(path_w_idx).convert('RGB')
            clip.append(cur_frame)

        return clip

    def read_ann(self, path):
        labels = []
        boxes  = []

        label  = path[0].split('/')[0]
        label  = int(self.name2idx[label])

        if self.phase == 'train':
            onehot_vector = np.zeros(13)
            onehot_vector[label] = 1.
            labels.append(onehot_vector)
        elif self.phase == 'test':
            labels.append(label)

        box    = [float(path[x]) for x in range(1, len(path))]
        boxes.append(box) 
        
        boxes  = np.array(boxes)

        if self.phase == 'train':
            labels = np.array(labels)
        elif self.phase == 'test':
            labels = np.expand_dims(np.array(labels), axis=1)
        
        return boxes, labels

    def read_key_frame(self, path):
        path = os.path.join(self.img_folder, path[0])
        return cv2.imread(path)

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx, get_origin_image=False):
        path          = self.data_list[idx]
        clip          = self.read_clip(path)
        boxes, labels = self.read_ann(path)

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
        
def build_ucfcrime_dataset(config, phase):
    root_path     = config['data_root']
    clip_length   = config['clip_length']
    sampling_rate = config['sampling_rate']
    img_size      = config['img_size']
    name2idx      = config['name2idx']

    if phase == 'train':
        split_path = 'Train2_annotation.pkl'
        return UCFCrime_dataset(root_path, phase, split_path,
                                clip_length, sampling_rate, img_size,
                                name2idx, Augmentation(img_size=img_size))
    elif phase == 'test':
        split_path = 'Test2_annotation.pkl'
        return UCFCrime_dataset(root_path, phase, split_path,
                                clip_length, sampling_rate, img_size,
                                name2idx, UCF_transform(img_size=img_size))