import torch
import torch.utils.data as data
import argparse
import yaml
import os
import cv2
import pickle
import numpy as np
from datasets.ucf.transforms import Augmentation, UCF_transform
from PIL import Image
import csv
            
path = 'weights/v8_n.pt'

state = torch.load(path)
keys = list(state.keys())
print(keys)
