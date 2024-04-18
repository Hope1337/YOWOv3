import torch

def collate_fn(batch_data):
    clips  = []
    boxes  = []
    labels = []
    for b in batch_data:
        clips.append(b[0])
        boxes.append(b[1])
        labels.append(b[2])
    
    clips = torch.stack(clips, dim=0) # [batch_size, num_frame, C, H, W]
    return clips, boxes, labels