#import cv2
#import torch
#from cus_datasets.build_dataset import build_dataset
#from model.TSN.YOWOv3 import build_yowov3
#from utils.box import non_max_suppression
#from utils.build_config import build_config
#import csv 
#import tqdm
#from evaluator.Evaluation import get_ava_performance
#from utils.flops import get_info

#def eval(config):

    #dataset = build_dataset(config, phase='test')
    #model = build_yowov3(config)
    #get_info(config, model)
    #model.to("cuda")
    #model.eval()
    
    #black_list = [2, 16, 18, 19, 21, 23, 25, 31, 32, 33, 35, 39, 40, 42, 44, 50, 53, 55, 71, 75]
    #ava_result_file = config['detections']
    #with open(ava_result_file, 'w', newline='') as file:

        #for idx in tqdm.tqdm(range(dataset.__len__())):
            #clip, boxes, labels, video_name, sec = dataset.__getitem__(idx)
            #clip = clip.unsqueeze(0).to('cuda')

            #outputs = model(clip)
            #outputs = non_max_suppression(outputs, 0.1, 0.5)[0]

            #H = config['img_size']
            #W = config['img_size']

            #outputs[:, 0] /= W
            #outputs[:, 1] /= H
            #outputs[:, 2] /= W
            #outputs[:, 3] /= H
        
            #writer = csv.writer(file)
            #for output in outputs:
                #if (int(output[5].item()) + 1) in black_list:
                    #continue
                #tl_x = round(output[0].item(), 3)
                #tl_y = round(output[1].item(), 3)
                #br_x = round(output[2].item(), 3)
                #br_y = round(output[3].item(), 3)
                #conf = round(output[4].item(), 3)
                #writer.writerow([video_name, sec, tl_x, tl_y, br_x, 
                    #br_y, int(output[5].item()) + 1, conf])


    #get_ava_performance.eval(config)

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

def ava_eval_collate_fn(batch):
    clips  = []
    vid_n  = []
    secs   = []

    for b in batch:
        clips.append(b[0])
        vid_n.append(b[3])
        secs.append(b[4])
    
    clips = torch.stack(clips, dim=0)
    return clips, vid_n, secs

def eval(config):

    dataset    = build_dataset(config, phase='test')
    dataloader = data.DataLoader(dataset, 32, False, collate_fn=ava_eval_collate_fn, num_workers=6, pin_memory=True)
    model      = build_yowov3(config)
    get_info(config, model)
    model.to("cuda")
    model.eval()
    
    black_list = [2, 16, 18, 19, 21, 23, 25, 31, 32, 33, 35, 39, 40, 42, 44, 50, 53, 55, 71, 75]
    ava_result_file = config['detections']

    results = []

    with torch.no_grad():
        for clips, video_names, secs in tqdm.tqdm(dataloader):
            clips = clips.to('cuda')
            outputss = model(clips)
            outputss = outputss.cpu()
            
            for outputs, video_name, sec in zip(outputss, video_names, secs):   
                outputs = outputs.unsqueeze(0)
                outputs = non_max_suppression(outputs, 0.1, 0.5)[0]

                H = config['img_size']
                W = config['img_size']

                outputs[:, 0] /= W
                outputs[:, 1] /= H
                outputs[:, 2] /= W
                outputs[:, 3] /= H
        
                for output in outputs:
                    if (int(output[5].item()) + 1) in black_list:
                        continue
                    #tl_x = round(output[0].item(), 3)
                    #tl_y = round(output[1].item(), 3)
                    #br_x = round(output[2].item(), 3)
                    #br_y = round(output[3].item(), 3)
                    #conf = round(output[4].item(), 3)
                    #writer.writerow([video_name, sec, tl_x, tl_y, br_x, 
                        #br_y, int(output[5].item()) + 1, conf])
                        
                    results.append([video_name, sec, round(output[0].item(), 3), 
                        round(output[1].item(), 3), round(output[2].item(), 3), 
                        round(output[3].item(), 3), int(output[5].item()) + 1, 
                        round(output[4].item(), 3)])


        with open(ava_result_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(results)

        get_ava_performance.eval(config)