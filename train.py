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
import shutil


def train_model(config):

    # Save config file
    #######################################################
    source_file = config['config_path']
    destination_file = os.path.join(config['save_folder'], 'config.yaml')
    shutil.copyfile(source_file, destination_file)
    #######################################################
    
    # create dataloader, model, criterion
    ####################################################
    dataset = build_dataset(config, phase='train')
    
    dataloader = data.DataLoader(dataset, config['batch_size'], True, collate_fn=collate_fn
                                 , num_workers=config['num_workers'], pin_memory=True, drop_last=True)
    
    model = build_yolo2stream(config)
    
    total_params = round(sum(p.numel() for p in model.parameters()) // 1e6)
    print(f"Tổng số lượng tham số: {total_params}")
    #sys.exit()
    model.train()
    model.to("cuda")
    
    criterion = build_loss(model, config)
    #####################################################

    biases = []
    not_biases = []

    for param_name, param in model.named_parameters():
        if param.requires_grad:
            if param_name.endswith('bias'):
                biases.append(param)
            else:
                not_biases.append(param)

    #optimizer  = optim.AdamW(params=model.parameters(), lr= config['lr'], weight_decay=config['weight_decay'])
    optimizer = optim.AdamW([
        {'params': not_biases, 'lr': config['lr']},  # Learning rate for non-bias parameters
        {'params': biases, 'lr': config['lr'] * 2}  # Double learning rate for bias parameters
        ], weight_decay=config['weight_decay'])
    
    warmup_lr  = LinearWarmup(config)

    adjustlr_schedule = config['adjustlr_schedule']
    acc_grad          = config['acc_grad'] 
    max_epoch         = config['max_epoch'] 
    lr_decay          = config['lr_decay']
    save_folder       = config['save_folder']
    
    torch.backends.cudnn.benchmark = True
    cur_epoch = 1
    loss_acc = 0.0
    ema = EMA(model)

    while(cur_epoch <= max_epoch):
        cnt_pram_update = 0
        for iteration, (batch_clip, batch_bboxes, batch_labels) in enumerate(dataloader): 
            t_batch = time.time()

            batch_size   = batch_clip.shape[0]
            batch_clip   = batch_clip.to("cuda")
            for idx in range(batch_size):
                batch_bboxes[idx]       = batch_bboxes[idx].to("cuda")
                batch_labels[idx]       = batch_labels[idx].to("cuda")

            outputs = model(batch_clip)

            targets = []
            for i, (bboxes, labels) in enumerate(zip(batch_bboxes, batch_labels)):
                nbox = bboxes.shape[0]
                nclass = labels.shape[1]
                target = torch.Tensor(nbox, 5 + nclass)
                target[:, 0] = i
                target[:, 1:5] = bboxes
                target[:, 5:] = labels
                targets.append(target)

            targets = torch.cat(targets, dim=0)

            loss = criterion(outputs, targets) / acc_grad
            loss_acc += loss.item()
            loss.backward()
            #plot_grad_flow(model.named_parameters()) #model too large, can't see anything!
            #plt.show()

            if ((iteration + 1) % acc_grad == 0) or (iteration + 1 == len(dataloader)):
                cnt_pram_update = cnt_pram_update + 1
                if cur_epoch == 1:
                    warmup_lr(optimizer, cnt_pram_update)
                nn.utils.clip_grad_value_(model.parameters(), clip_value=2.0)
                optimizer.step()
                optimizer.zero_grad()
                ema.update(model)

                print("epoch : {}, update : {}, time = {}, loss = {}".format(cur_epoch,  cnt_pram_update, round(time.time() - t_batch, 2), loss_acc))
                loss_acc = 0.0
                #if cnt_pram_update % 500 == 0:
                    #torch.save(model.state_dict(), r"/home/manh/Projects/My-YOWO/weights/model_checkpoint/epch_{}_update_".format(cur_epoch) + str(cnt_pram_update) + ".pth")

        if cur_epoch in adjustlr_schedule:
            for param_group in optimizer.param_groups: 
                param_group['lr'] *= lr_decay
        
        #          model.state_dict()
        save_path_ema = os.path.join(save_folder, "ema_epoch_" + str(cur_epoch) + ".pth")
        torch.save(ema.ema.state_dict(), save_path_ema)

        save_path     = os.path.join(save_folder, "epoch_"     + str(cur_epoch) + ".pth")
        torch.save(model.state_dict(), save_path)

        print("Saved model at epoch : {}".format(cur_epoch))

        #log_path = '/home/manh/Projects/YOLO2Stream/training.log'
        #map50, mean_ap = call_eval(save_path)
        #logging.basicConfig(filename=log_path, level=logging.INFO)
        #logging.info('mAP 0.5 : {}, mAP : {}'.format(map50, mean_ap))

        cur_epoch += 1

if __name__ == "__main__":
    config = build_config()
    train_model(config)   