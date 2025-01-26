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
from cus_datasets.ucf.load_data import UCF_dataset
from cus_datasets.collate_fn import collate_fn
from cus_datasets.build_dataset import build_dataset
from model.TSN.YOWOv3 import build_yowov3 
from utils.loss import build_loss
from utils.warmup_lr import LinearWarmup
import shutil
from utils.flops import get_info
from model.TSN.classification import build_classificationmodel
from cus_datasets.rwf200.load_data import build_rwf2000_dataset, collate_fn_rwf2000

def train_model(config):

    # Save config file
    #######################################################
    source_file = config['config_path']
    destination_file = os.path.join(config['save_folder'], 'config.yaml')
    shutil.copyfile(source_file, destination_file)
    #######################################################
    
    # create dataloader, model, criterion
    ####################################################
    dataset = build_rwf2000_dataset(config, phase='train')
    
    dataloader = data.DataLoader(dataset, config['batch_size'], True, collate_fn=collate_fn_rwf2000
                                 , num_workers=config['num_workers'], pin_memory=True)
    
    model = build_classificationmodel(config)
    model.to("cuda")
    model.train()
    
    criterion = nn.CrossEntropyLoss()
    #####################################################

    #optimizer  = optim.AdamW(params=model.parameters(), lr= config['lr'], weight_decay=config['weight_decay'])

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
        for iteration, (batch_clip, batch_label) in enumerate(dataloader):
            batch_clip  = batch_clip.to('cuda')
            batch_label = batch_label.to('cuda').to(torch.long).squeeze(-1)
            output      = model(batch_clip)

            loss        = criterion(output, batch_label)

            loss.backward()

            cnt_pram_update = cnt_pram_update + 1
            if cur_epoch == 1:
                warmup_lr(optimizer, cnt_pram_update)
            nn.utils.clip_grad_value_(model.parameters(), clip_value=2.0)
            optimizer.step()
            optimizer.zero_grad()
            ema.update(model)

            print("epoch : {}, update : {}, loss = {}".format(cur_epoch,  cnt_pram_update, loss), flush=True)
            with open(os.path.join(config['save_folder'], "logging.txt"), "w") as f:
                f.write("epoch : {}, update : {}, loss = {}".format(cur_epoch,  cnt_pram_update, loss))

            #if cnt_pram_update % 500 == 0:
                #torch.save(model.state_dict(), r"/home/manh/Projects/My-YOWO/weights/model_checkpoint/epch_{}_update_".format(cur_epoch) + str(cnt_pram_update) + ".pth")

        if cur_epoch in adjustlr_schedule:
            for param_group in optimizer.param_groups: 
                param_group['lr'] *= lr_decay
        
        save_path_ema = os.path.join(save_folder, "ema_epoch_" + str(cur_epoch) + ".pth")
        torch.save(ema.ema.state_dict(), save_path_ema)

        save_path     = os.path.join(save_folder, "epoch_"     + str(cur_epoch) + ".pth")
        torch.save(model.state_dict(), save_path)

        print("Saved model at epoch : {}".format(cur_epoch), flush=True)


        cur_epoch += 1

if __name__ == "__main__":
    config = build_config('config/rwf2000_config.yaml')
    train_model(config)   