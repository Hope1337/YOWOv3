from model.TSN.YOWOv3 import build_yowov3
from cus_datasets.build_dataset import build_dataset
from utils.box import non_max_suppression
import onnxruntime

import torch
from utils.box import draw_bounding_box
import cv2
import numpy as np

def export2onnx(config):
    model   = build_yowov3(config) 
    model.eval()

    dummy_input = torch.randn(1, 3, 16, 224, 224)

    torch.onnx.export(model,
                    dummy_input,
                    "yowov3.onnx",
                    verbose=False,
                    input_names=['clip'],
                    output_names=['image'],
                    export_params=True)
    
    
    mapping = config['idx2name']
    onnx_model_path = "yowov3.onnx"
    ort_session = onnxruntime.InferenceSession(onnx_model_path)

    dataset = build_dataset(config, phase='test')

    for idx in range(dataset.__len__()):
        origin_image, clip, bboxes, labels = dataset.__getitem__(idx, get_origin_image=True)

        clip = clip.unsqueeze(0)       

        input_data = {ort_session.get_inputs()[0].name: clip.numpy()}
        outputs = torch.tensor(ort_session.run(None, input_data))
        #outputs = torch.from_numpy(outputs[0])
        outputs = non_max_suppression(outputs[0], conf_threshold=0.3, iou_threshold=0.5)[0]
        origin_image = cv2.resize(origin_image, (config['img_size'], config['img_size']))
        draw_bounding_box(origin_image, outputs[:, :4], outputs[:, 5], outputs[:, 4], mapping)

        flag = 1 
        if flag:
            cv2.imshow('img', origin_image)
            k = cv2.waitKey(1)
            if k == ord('q'):
                return
        else:
            cv2.imwrite(r"H:\detect_images\_" + str(idx) + r".jpg", origin_image)

            print("ok")
            print("image {} saved!".format(idx))