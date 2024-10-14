from datasets.jhmdb.load_data import JHMDB_dataset
from datasets.ucf.transforms import Augmentation, UCF_transform
import cv2

if __name__ == '__main__':
    root_path = '/home/manh/Datasets/jhmdb'
    clip_length   = 16
    sampling_rate = 1
    img_size      = 224

    dataset = JHMDB_dataset(root_path, 'test', 'trainlist.txt', clip_length,
                             sampling_rate, img_size, 
                             transform=UCF_transform(img_size=img_size))


    for i in range(dataset.__len__()):
        origin_image, clip, boxes, labels = dataset.__getitem__(i, True)
        print(clip.shape)
        print(boxes.shape)
        print(labels.shape)
        cv2.imshow('img', origin_image)
        cv2.waitKey()