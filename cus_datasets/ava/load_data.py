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
            
class AVA_dataset(data.Dataset):

    def __init__(self, root_path, split_path, data_path, clip_length, sampling_rate, img_size, transform=Augmentation(), phase='train'):
       self.root_path     = root_path
       self.split_path    = os.path.join(root_path, 'annotations', split_path)
       self.data_path     = os.path.join(root_path, data_path)
       self.clip_length   = clip_length
       self.sampling_rate = sampling_rate
       self.transform     = transform
       self.valid_frame   = range(902, 1799) 
       self.num_classes   = 80
       self.phase         = phase
       self.img_size      = img_size

       self.read_ann_csv()
    
    def read_ann_csv(self):
        my_dict = dict()

        with open(self.split_path, 'r') as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                # Combine the first two columns to form the key
                key = '/'.join([row[0], row[1]])

                # Combine the next four columns to form the subkey
                subkey = '/'.join([row[2], row[3], row[4], row[5]])

                # Get the dictionary associated with the key, or create a new one if it doesn't exist
                sub_dict = my_dict.setdefault(key, dict())

                # Get the list associated with the subkey, or create a new one if it doesn't exist
                sub_list = sub_dict.setdefault(subkey, [])

                # Append the value to the sub-list
                sub_list.append(int(row[6]))

        self.data_dict = my_dict
        self.data_list = list(my_dict.keys())
        self.data_len  = len(self.data_list)

    def __len__(self):
        return self.data_len
    
    def __getitem__(self, index, get_origin_image=False):

        video_name, sec = self.data_list[index].split('/')
        str_sec = sec
        sec = int(sec)
        key_frame_idx = (sec - 900) * 30 + 1
        video_path = os.path.join(self.data_path, video_name)

        clip        = []
        boxes       = []
        for i in reversed(range(self.clip_length)):
            cur_frame_idx = key_frame_idx - i*self.sampling_rate

            if cur_frame_idx < 1:
                cur_frame_idx = 1

            # get frame
            cur_frame_path = os.path.join(video_path, video_name + '_{:06d}.jpg'.format(cur_frame_idx))
            cur_frame = Image.open(cur_frame_path).convert('RGB')
            clip.append(cur_frame)

        if get_origin_image == True:
            key_frame_path     = os.path.join(video_path, video_name + '_{:06d}.jpg'.format(key_frame_idx))
            original_image     = cv2.imread(key_frame_path)

        boxes  = []
        labels = [] 

        W, H =  clip[-1].size

        cur_frame_dict = self.data_dict[self.data_list[index]]
        for raw_bboxes in cur_frame_dict.keys():
            box = list(raw_bboxes.split('/'))
            box = [float(x) for x in box]
            box[0] *= W
            box[1] *= H
            box[2] *= W
            box[3] *= H

            label = np.zeros(self.num_classes)
            for x in cur_frame_dict[raw_bboxes]:
                label[x - 1] = 1

            boxes.append(box)
            labels.append(label)

        boxes = np.array(boxes)
        labels = np.array(labels)
        
        # clip   : list of (num_frame) PIL image (RGB order, 0 .. 255)
        # boxes  : np array [nbox, 4], absolute coordinate (tl-br)
        # labels : np array [nbox, nclass]


        targets = np.concatenate((boxes, labels), axis=1)
        clip, targets = self.transform(clip, targets)
        
        boxes = targets[:, :4]
        labels = targets[:, 4:]

        # clip   : tensor [C, numframe, H, W] (RBG order)
        # boxes  : tensor [nbox, 4], relative coordinate (tl-br)
        # labels : tensor [nbox, nclass]
        if get_origin_image == True : 
            return original_image, clip, boxes, labels
        elif self.phase == 'test':
            return clip, boxes, labels, video_name, str_sec
        else:
            return clip, boxes, labels
        

def build_ava_dataset(config, phase):
    root_path     = config['data_root']
    data_path     = "frames"
    clip_length   = config['clip_length']
    sampling_rate = config['sampling_rate']
    img_size      = config['img_size']

    if phase == 'train':
        split_path    = "ava_train_v2.2.csv"
        return AVA_dataset(root_path=root_path, split_path=split_path, data_path=data_path, clip_length=clip_length,
                           sampling_rate=sampling_rate, img_size=img_size, transform=Augmentation(img_size=img_size), phase=phase)
    elif phase == 'test':
        split_path    = "ava_val_v2.2.csv"
        return AVA_dataset(root_path=root_path, split_path=split_path, data_path=data_path, clip_length=clip_length,
                           sampling_rate=sampling_rate, img_size=img_size, transform=UCF_transform(img_size=img_size), phase=phase)
        
