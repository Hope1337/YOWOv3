import cv2
import numpy as np
from evaluator.Evaluation import get_ava_performance
from utils.build_config import build_config

config = build_config('config/ava_config.yaml')
get_ava_performance.eval(config)