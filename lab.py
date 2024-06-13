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
            

a = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(a.shape)
a = torch.where(a > 3)
print(a)