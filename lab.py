#import pickle

## Đọc dữ liệu từ file .pkl
#with open('/home/manh/Datasets/UCF-Crime/Annotations/Test_annotation.pkl', 'rb') as file:
    #data = pickle.load(file)

## Sử dụng dữ liệu đã đọc
#for x in data:
    #for t in data[x]:
        #print(t)
        #import sys
        #sys.exit()

#name2idx = {
    #'Abuse'         : 0,
    #'Arrest'        : 1,
    #'Arson'         : 2,
    #'Assault'       : 3,
    #'Burglary'      : 4,
    #'Explosion'     : 5,
    #'Fighting'      : 6,
    #'RoadAccidents' : 7,
    #'Robbery'       : 8,
    #'Shooting'      : 9,
    #'Shoplifting'   : 10,
    #'Stealing'      : 11,
    #'Vandalism'     : 12
#}

#from datasets.ucf_crime.load_data import UCFCrime_dataset
#from datasets.ucf.transforms import Augmentation, UCF_transform
#from PIL import Image
#import cv2

#root_path  = '/home/manh/Datasets/UCF-Crime'
#phase      = 'test'
#split_path = 'Test_annotation.pkl'
#clip_length = 16
#sampling_rate = 1
#img_size = 224
#dataset = UCFCrime_dataset(root_path, phase, split_path,
                           #clip_length, sampling_rate, img_size,
                           #name2idx, transform=UCF_transform(img_size=img_size))

#for i in range(dataset.__len__()):
    #origin_image, clip, boxes, label = dataset.__getitem__(i, True)
    #print(i)
    #print(boxes)
    #H, W, C = origin_image.shape
    #print(origin_image.shape)
    #cv2.rectangle(origin_image, (int(boxes[0, 0]*W), int(boxes[0, 1]*H)), (int(boxes[0, 2]*W), int(boxes[0, 3]*H)), (0, 0, 255), 1)
    #cv2.imshow('img', origin_image)
    #cv2.waitKey()

#import os
#import cv2

#videos_folder = '/home/manh/Datasets/UCF-Crime/Anomaly/Videos/'
#images_folder = '/home/manh/Datasets/UCF-Crime/Anomaly/Images/'

## Duyệt qua tất cả các thư mục con trong thư mục Videos
#for class_name in os.listdir(videos_folder):
    #class_path = os.path.join(videos_folder, class_name)
    #if os.path.isdir(class_path):
        ## Tạo thư mục class_name trong thư mục Images nếu chưa tồn tại
        #images_class_path = os.path.join(images_folder, class_name)
        #os.makedirs(images_class_path, exist_ok=True)
        
        ## Duyệt qua tất cả các file video trong thư mục class
        #for video_name in os.listdir(class_path):
            #video_path = os.path.join(class_path, video_name)
            #if video_name.endswith('.mp4'):
                ## Tạo thư mục video_name trong thư mục class_name nếu chưa tồn tại
                #images_video_path = os.path.join(images_class_path, os.path.splitext(video_name)[0])
                #os.makedirs(images_video_path, exist_ok=True)
                
                ## Đọc video với frame rate là 30 fps
                #cap = cv2.VideoCapture(video_path)
                #cap.set(cv2.CAP_PROP_FPS, 30)
                
                #frame_count = 1
                
                ## Đọc từng frame từ video
                #while cap.isOpened():
                    #ret, frame = cap.read()
                    #if not ret:
                        #break
                    
                    ## Lưu frame thành hình ảnh với định dạng 04d.jpg
                    #if frame_count <= 9999:
                        #frame_path = os.path.join(images_video_path, 'image_{:04d}.jpg'.format(frame_count))
                    #else:
                        #frame_path = os.path.join(images_video_path, 'image_{:05d}.jpg'.format(frame_count))
                    #cv2.imwrite(frame_path, frame)
                    #frame_count += 1

                #cap.release()

#print('Quá trình tạo hình ảnh từ video đã hoàn tất.')


#import pickle

## Đọc dữ liệu từ file .pkl
#with open('/home/manh/Datasets/UCF-Crime/Annotations/Train_annotation.pkl', 'rb') as file:
    #train_data = pickle.load(file)

#with open('/home/manh/Datasets/UCF-Crime/Annotations/Test_annotation.pkl', 'rb') as file:
    #test_data = pickle.load(file)

#train = train_data.keys()
#test  = test_data.keys()

#common_keys = set(train) & set(test)

##for x in train_data:
    ##print(train_data[x])
    ##break

#data = {**train_data, **test_data}

#train1 = {}
#test1  = {}

#for x in data:
    #train1[x]  = []
    #test1[x]   = []

    #num = len(data[x])
    #sorted_data = sorted(data[x], key=lambda x: x[0])
    
    #train_len = int(0.7 * num)

    #for i in range(0, train_len):
        #train1[x].append(sorted_data[i])
    
    #for i in range(train_len, num):
        #test1[x].append(sorted_data[i])

#with open('/home/manh/Datasets/UCF-Crime/Annotations/Train2_annotation.pkl', 'wb') as file:
    #pickle.dump(train1, file)


#with open('/home/manh/Datasets/UCF-Crime/Annotations/Test2_annotation.pkl', 'wb') as file:
    #pickle.dump(test1, file)

##print(train1)

## Đọc dữ liệu từ file .pkl
#with open('/home/manh/Datasets/UCF-Crime/Annotations/Train2_annotation.pkl', 'rb') as file:
    #train_data = pickle.load(file)

#print(train_data)

import os 

t = os.environ['LOCAL_RANK']
print(t)