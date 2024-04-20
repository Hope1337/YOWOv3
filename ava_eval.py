import cv2
import torch
from datasets.build_dataset import build_dataset
from model.TSN.YOLO2Stream import build_yolo2stream
from utils.box import non_max_suppression
from utils.build_config import build_config
import csv 
import tqdm
from evaluator.Evaluation import get_ava_performance

def eval(config):

    dataset = build_dataset(config, phase='test')
    model = build_yolo2stream(config)
    model.to("cuda")
    model.eval()

    ava_result_file = config['detections']
    with open(ava_result_file, 'w', newline='') as file:

        for idx in tqdm.tqdm(range(dataset.__len__())):
            clip, boxes, labels, video_name, sec = dataset.__getitem__(idx)
            clip = clip.unsqueeze(0).to('cuda')

            outputs = model(clip)
            outputs = non_max_suppression(outputs, 0.1, 0.5)[0]

            H = 224
            W = 224

            outputs[:, 0] /= W
            outputs[:, 1] /= H
            outputs[:, 2] /= W
            outputs[:, 3] /= H
        
            writer = csv.writer(file)
            for output in outputs:
                writer.writerow([video_name, sec, output[0].item(), output[1].item(),
                                 output[2].item(), output[3].item(), int(output[5].item()) + 1, output[4].item()])
            

config = build_config()
#eval(config)

get_ava_performance.eval(config)