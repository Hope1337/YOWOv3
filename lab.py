import cv2
import torch
from cus_datasets.build_dataset import build_dataset
from model.TSN.YOWOv3 import build_yowov3
from utils.box import non_max_suppression
from utils.build_config import build_config
import csv 
import tqdm
from evaluator.Evaluation import get_ava_performance
from utils.flops import get_info
import torch.utils.data as data

ava_result_file = 't.csv'
results         = [['hi', 'hi2'], ['t1', 't2']]

with open(ava_result_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(results)
