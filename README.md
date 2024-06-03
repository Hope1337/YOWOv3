# YOWOv3: An Efficient and Generalized Framework for Human Action Detection and Recognition

## Temporary Instructions

If you encounter any difficulties or have any questions, feel free to ask in Issues section. I'm here to answer everyone's questions. Thank you sincerely :3.

Currently, the **YOWOv3** model and paper are still being finalized, and experiments are continuously being conducted. Therefore, we are unable to provide complete official instructions at this time. However, we will provide a temporary guide, and the comprehensive official instructions will be supplemented and completed in the near future, soon :3. For now, you can:

- Prepare UCF101-24 dataset.
- Prepare AVAv2.2 dataset.
- Detect on UCF101-24 and AVAv2.2 dataset.
- Detect using your own camera.

Note: In the **config** folder, there are two files: **ucf_config.yaml** and **ava_config.yaml**. These are the configuration files for the corresponding datasets. We read information from these config files to build the model and specify hyperparameters and related details. To decide which file to use, simply go to **utils/build_config.py** and modify the default path in the ```build_config``` function (see code below) to the desired file. This can be handled more easily using an argument parser, but as mentioned, the code is currently being used for research purposes, and everything is set up for convenience during experimentation.

```python
def build_config(config_file='config/ucf_config.yaml'):
    with open(config_file, "r") as file:
        config = yaml.load(file, Loader=yaml.SafeLoader)

    if config['active_checker']:
        pass
    
    return config
```

### Prepare UCF101-24 dataset
- Download from (as in YOWOv2): https://drive.google.com/file/d/1Dwh90pRi7uGkH5qLRjQIFiEmMJrAog5J/view
- Then go to the config file and change **data_root** to the path to the folder path of UCF101-24 dataset.
### Prepare AVAv2.2 dataset
- Follow the instructions at: https://github.com/yjh0410/AVA_Dataset.
- If you find that video downloading takes too long, you can consider downloading them from my Hugging Face repository and then proceed with the remaining steps as instructed above: https://huggingface.co/datasets/manh6054/avav2.2_train_set/tree/main.
### Detect on UCF101-24 and AVAv2.2 dataset
- Specify which config file will be read (as mentioned above).
- Download the checkpoint for AVAv2.2 (**Medium-2 model**, checkpoint file name: ema_epoch_7_88.24_mAP.pth) from: https://huggingface.co/manh6054/Project_VU/tree/main.
- Download the checkpoint for UCF101-24 (**Medium-2 model**, checkpoint file name: ema_epoch_9.pth) from: https://huggingface.co/manh6054/Project_VU/tree/main.
- Open the config file and modify **backbone3D** to **i3d** and **pretrain_path** to the path of the downloaded checkpoint.
- Run ```python detect.py```
### Detect using your own camera
Follow the instructions as mentioned above and run ```python live.py```.

## Experimental results:

Note: The **Medium-1**, **Medium-2**, and **Large** models correspond to the 3D backbone models shufflenetv2, i3d, and resnext101, respectively.

### UCF101-24
![image](https://github.com/AakiraOtok/Project_VU/assets/120596914/45f27332-ac36-4d2b-beb9-4cbf6c35cff5)

### AVAv2.2
![image](https://github.com/AakiraOtok/Project_VU/assets/120596914/6658d501-aa2d-47ef-af4b-e1b2d672b709)

## Dataset

## Pretrained Model 

## Instruction

Some notes:
- In the commit history, there are commits named after the best checkpoints. The code in that commit is what I used to train the model, but it's not the config file! This is because during the training process, I opened another terminal window to experiment with a few things, so the config file changed and I didn't revert it back to the original. The original configs are saved in my notes, not in the commit files.


## References

I would like to express my sincere gratitude to the following amazing repositories/codes, which were the primary sources I heavily relied on during the development of this project:

- [A neat implementation of YOLOv8 using PyTorch](https://github.com/jahongir7174/YOLOv8-pt)
- [3D CNN backbones: MobileNet/v2, ShuffleNet/v2, ResNet, ResNeXt](https://github.com/okankop/Efficient-3DCNNs)
- [Implementation of the I3D model on PyTorch](https://github.com/piergiaj/pytorch-i3d)
- [YOWO model](https://github.com/wei-tim/YOWO?tab=readme-ov-file)
- [YOWOv2 model](https://github.com/yjh0410/YOWOv2)
- [AVAv2.2 evaluation code from the organizers](https://github.com/activitynet/ActivityNet/tree/master/Evaluation)
