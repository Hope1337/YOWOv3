##from model.backbone2D.build_backbone2D import build_backbone2D
##from utils.build_config import build_config
##import torch
##from PIL import Image
##from datasets.ucf.transforms import UCF_transform
##import matplotlib.pyplot as plt


##def show_image_from_tensor(image_tensor):
    ##image_np = image_tensor.cpu().detach().squeeze(0).permute(1, 2, 0).numpy()
    ##image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())
    ##plt.imshow(image_np)
    ##plt.axis('off')  # Tắt trục x, y
    ##plt.show(block=False)
    ##plt.pause(0.1) 

##img_size = 640

##transform = UCF_transform(img_size)
##config = build_config('config/ucf_config.yaml')
##model = build_backbone2D(config)
##model.load_pretrain()
##for param in model.parameters():
    ##param.requires_grad = False

##targets = torch.randn((1, 4))
##image = torch.ones((1, 3, img_size, img_size))
##image.requires_grad = True

##img1 = Image.open('459580930_1089993085880734_4229300939237798211_n.jpg').convert('RGB')
##img1 = transform([img1], targets)[0].squeeze(1).unsqueeze(0)
###print(img1.requires_grad)


##img2 = Image.open('starry.jpg').convert('RGB')
##img2 = transform([img2], targets)[0].squeeze(1).unsqueeze(0)
##print(img2.requires_grad)

##feature1 = list(model(img1))
##feature2 = list(model(img2))

##max_iter = 10000

##def l2_loss(tensor1, tensor2):
    ##loss = torch.sum((tensor1 - tensor2) ** 2)
    ###loss = torch.sum((tensor1 - tensor2) ** 2)
    ##return loss

##def gram_matrix(tensor):
    ##B, C, H, W = tensor.shape
    ##tensor = tensor.view(B*C, H*W)
    ##return torch.mm(tensor, tensor.T)
    ###return torch.einsum('tchw,tcab -> thwab', tensor, tensor)

##def content_loss(source, target):
    ##return l2_loss(source, target) / 2

##def style_loss(source, target):
    ##S = gram_matrix(source)
    ##T = gram_matrix(target)
    ##B, C, H, W = target.shape
    
    ##return l2_loss(S, T) / ((C * ((H * W))))


##optimizer     = torch.optim.AdamW([image], lr=5e-3, weight_decay=5e-4)
#from model.backbone2D.build_backbone2D import build_backbone2D
#from utils.build_config import build_config
#import torch
#from PIL import Image
#from datasets.ucf.transforms import UCF_transform
#import matplotlib.pyplot as plt


#def show_image_from_tensor(image_tensor):
    #image_np = image_tensor.cpu().detach().squeeze(0).permute(1, 2, 0).numpy()
    #image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())
    #plt.imshow(image_np)
    #plt.axis('off')  # Tắt trục x, y
    #plt.show(block=False)
    #plt.pause(0.1) 

#img_size = 640

#transform = UCF_transform(img_size)
#config = build_config('config/ucf_config.yaml')
#model = build_backbone2D(config)
#model.load_pretrain()
#for param in model.parameters():
    #param.requires_grad = False

#targets = torch.randn((1, 4))
#image = torch.ones((1, 3, img_size, img_size))
#image.requires_grad = True

#img1 = Image.open('459580930_1089993085880734_4229300939237798211_n.jpg').convert('RGB')
#img1 = transform([img1], targets)[0].squeeze(1).unsqueeze(0)
##print(img1.requires_grad)


#img2 = Image.open('starry.jpg').convert('RGB')
#img2 = transform([img2], targets)[0].squeeze(1).unsqueeze(0)
#print(img2.requires_grad)

#feature1 = list(model(img1))
#feature2 = list(model(img2))

#max_iter = 10000

#def l2_loss(tensor1, tensor2):
    #loss = torch.sum((tensor1 - tensor2) ** 2)
    ##loss = torch.sum((tensor1 - tensor2) ** 2)
    #return loss

#def gram_matrix(tensor):
    #B, C, H, W = tensor.shape
    #tensor = tensor.view(B*C, H*W)
    #return torch.mm(tensor, tensor.T)
    ##return torch.einsum('tchw,tcab -> thwab', tensor, tensor)

#def content_loss(source, target):
    #return l2_loss(source, target) / 2

#def style_loss(source, target):
    #S = gram_matrix(source)
    #T = gram_matrix(target)
    #B, C, H, W = target.shape
    
    #return l2_loss(S, T) / ((C * ((H * W))))


#optimizer     = torch.optim.AdamW([image], lr=5e-3, weight_decay=5e-4)


#for i in range(max_iter):
    #feature = list(model(image))

    #loss1 = 0.
    #loss2 = 0.

    #for (x1, x2) in zip(feature[:1], feature1[:1]):
        #loss1 = loss1 + content_loss(x1, x2)
        
    #for (x1, x2) in zip(feature[:1], feature2[:1]):
        #loss2 = loss2 + style_loss(x1, x2)

    #loss1 = loss1 * 0.0
    #loss2 = loss2 * 1.0

    ##loss3           = loss2
    #loss3           = loss1 + loss2
      
    #print("iter : {}, loss1  : {}, loss2 : {}".format(i, loss1, loss2))
    
    #loss3.backward()
    #optimizer.step()
    #optimizer.zero_grad()
    
    
    ##if i in decay_schedule:
        ##scheduler.step()

    #if (i + 1)%10 == 0:
        #show_image_from_tensor(image)



##for i in range(max_iter):
    ##feature = list(model(image))

    ##loss1 = 0.
    ##loss2 = 0.

    ##for (x1, x2) in zip(feature[:1], feature1[:1]):
        ##loss1 = loss1 + content_loss(x1, x2)
        
    ##for (x1, x2) in zip(feature[:1], feature2[:1]):
        ##loss2 = loss2 + style_loss(x1, x2)

    ##loss1 = loss1 * 0.0
    ##loss2 = loss2 * 1.0

    ###loss3           = loss2
    ##loss3           = loss1 + loss2
      
    ##print("iter : {}, loss1  : {}, loss2 : {}".format(i, loss1, loss2))
    
    ##loss3.backward()
    ##optimizer.step()
    ##optimizer.zero_grad()
    
    
    ###if i in decay_schedule:
        ###scheduler.step()

    ##if (i + 1)%10 == 0:
        ##show_image_from_tensor(image)

import torch
import torch.utils.data as data
import os
import cv2
import pickle
import numpy as np
from datasets.ucf.transforms import Augmentation, UCF_transform
from PIL import Image
from torchvision import transforms
import random
import torchvision.transforms.functional as F

def collate_fn(batch_data):
    clips  = []
    labels = []
    for b in batch_data:
        clips.append(b[0])
        labels.append(b[1])
    
    clips = torch.stack(clips, dim=0) # [batch_size, num_frame, C, H, W]
    labels = torch.stack(labels, dim=0)
    return clips, labels

class RWFtransfrom():
    def __init__(self, img_size):
        self.img_size = img_size
        self.transform = transforms.Compose([
                    transforms.Resize((img_size, img_size)),  
                    transforms.ToTensor(), 
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

    def __call__(self, clip):
        result = []
        for frame in clip:
            result.append(self.transform(frame))

        return result
    
class RWFAugmentation(object):
    def __init__(self, img_size=224, jitter=0.2, hue=0.1, saturation=1.5, exposure=1.5):
        self.img_size = img_size
        self.jitter = jitter
        self.hue = hue
        self.saturation = saturation
        self.exposure = exposure


    def rand_scale(self, s):
        scale = random.uniform(1, s)

        if random.randint(0, 1): 
            return scale

        return 1./scale


    def random_distort_image(self, video_clip):
        dhue = random.uniform(-self.hue, self.hue)
        dsat = self.rand_scale(self.saturation)
        dexp = self.rand_scale(self.exposure)
        
        video_clip_ = []
        for image in video_clip:
            image = image.convert('HSV')
            cs = list(image.split())
            cs[1] = cs[1].point(lambda i: i * dsat)
            cs[2] = cs[2].point(lambda i: i * dexp)
            
            def change_hue(x):
                x += dhue * 255
                if x > 255:
                    x -= 255
                if x < 0:
                    x += 255
                return x

            cs[0] = cs[0].point(change_hue)
            image = Image.merge(image.mode, tuple(cs))

            image = image.convert('RGB')

            video_clip_.append(image)

        return video_clip_


    def random_crop(self, video_clip, width, height):
        dw =int(width * self.jitter)
        dh =int(height * self.jitter)

        pleft  = random.randint(-dw, dw)
        pright = random.randint(-dw, dw)
        ptop   = random.randint(-dh, dh)
        pbot   = random.randint(-dh, dh)

        swidth =  width - pleft - pright
        sheight = height - ptop - pbot

        sx = float(swidth)  / width
        sy = float(sheight) / height
        
        dx = (float(pleft) / width)/sx
        dy = (float(ptop) / height)/sy

        # random crop
        cropped_clip = [img.crop((pleft, ptop, pleft + swidth - 1, ptop + sheight - 1)) for img in video_clip]

        return cropped_clip, dx, dy, sx, sy
        
    def to_tensor(self, video_clip):
        return [F.to_tensor(image) for image in video_clip]
    
    def normalization(self, video_clip):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)

        video_clip = (video_clip - mean)/std
        return video_clip


    def __call__(self, video_clip):
        oh = video_clip[-1].height  
        ow = video_clip[-1].width
        
        # random crop
        #video_clip, dx, dy, sx, sy = self.random_crop(video_clip, ow, oh)

        # resize
        video_clip = [img.resize([self.img_size, self.img_size]) for img in video_clip]

        # random flip
        flip = random.randint(0, 1)
        if flip:
            video_clip = [img.transpose(Image.Transpose.FLIP_LEFT_RIGHT) for img in video_clip]

        # distort
        video_clip = self.random_distort_image(video_clip)
            
        # to tensor
        video_clip = self.to_tensor(video_clip)
        video_clip = torch.stack(video_clip, dim=1)
        
        video_clip = self.normalization(video_clip)
        
        return video_clip 

class RWF2000(data.Dataset):
    def __init__(self, root_path, phase, clip_length, 
                 sampling_rate, img_size, transform):
        
        self.root_path     = os.path.join(root_path, phase)
        self.phase         = phase
        self.clip_length   = clip_length
        self.sampling_rate = sampling_rate
        self.img_size      = img_size
        self.transform     = transform
        self.frame_list    = []

        self.build_video_list()
        self.data_len      = len(self.frame_list)

    def build_video_list(self):

        for classes in ["Fight", "NonFight"]:
            path = os.path.join(self.root_path, classes)
            for video in os.listdir(path):
                video_folder = os.path.join(path, video)
                for frame in os.listdir(video_folder):
                    if int(frame.split('.')[0]) >= 15:
                        self.frame_list.append(os.path.join(video_folder, frame))
    
    def read_clip(self, path):
        clip          = []
        path_template = '/'.join(path.split('/')[:-1])
        key_idx       = int(path.split('/')[-1].split('.')[0])

        for idx in reversed(range(self.clip_length)):
            cur_frame_idx = key_idx - idx * self.sampling_rate
            path_w_idx    = os.path.join(path_template, '{:05d}.jpg'.format(cur_frame_idx))
            cur_frame     = Image.open(path_w_idx).convert('RGB')
            clip.append(cur_frame)
        return clip
    
    def read_label(self, path):
        label = path.split('/')[-3]
        if label == 'Fight':
            label = torch.zeros(1)
        elif label == 'NonFight':
            label = torch.ones(1)
        return label
    
    def read_key_frame(self, path):
        return Image.open(path).convert("RGB")

    def __len__(self):
        return self.data_len
    
    def __getitem__(self, idx, get_origin_image=False):
        path  = self.frame_list[idx]
        clip  = self.read_clip(path)
        label = self.read_label(path)

        if self.transform is not None:
            clip = self.transform(clip)
        
        #clip  = torch.stack(clip, dim=0)
        
        if get_origin_image == True:
            origin_image = self.read_key_frame(path)
            return origin_image, clip, label
        else:
            return clip, label

if __name__ == "__main__":
    root_path     = '/home/manh/Datasets/RWF-2000'
    phase         = 'train'
    clip_length   = 16
    sampling_rate = 1
    img_size      = 224
    transform     = RWFAugmentation(224)

    dataset = RWF2000(root_path, phase, clip_length, sampling_rate, img_size, transform)
    dataloader = data.DataLoader(dataset, 2, 1, collate_fn=collate_fn, pin_memory=True)

    for clip, label in dataloader:
        print(clip.shape)
        print(label.shape)
        pass