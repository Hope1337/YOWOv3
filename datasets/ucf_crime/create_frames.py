import os
import cv2

videos_folder = '/home/manh/Datasets/UCF-Crime/Anomaly/Videos/'
images_folder = '/home/manh/Datasets/UCF-Crime/Anomaly/Images/'

# Duyệt qua tất cả các thư mục con trong thư mục Videos
for class_name in os.listdir(videos_folder):
    class_path = os.path.join(videos_folder, class_name)
    if os.path.isdir(class_path):
        # Tạo thư mục class_name trong thư mục Images nếu chưa tồn tại
        images_class_path = os.path.join(images_folder, class_name)
        os.makedirs(images_class_path, exist_ok=True)
        
        # Duyệt qua tất cả các file video trong thư mục class
        for video_name in os.listdir(class_path):
            video_path = os.path.join(class_path, video_name)
            if video_name.endswith('.mp4'):
                # Tạo thư mục video_name trong thư mục class_name nếu chưa tồn tại
                images_video_path = os.path.join(images_class_path, os.path.splitext(video_name)[0])
                os.makedirs(images_video_path, exist_ok=True)
                
                # Đọc video với frame rate là 30 fps
                cap = cv2.VideoCapture(video_path)
                cap.set(cv2.CAP_PROP_FPS, 30)
                
                frame_count = 0
                
                # Đọc từng frame từ video
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Lưu frame thành hình ảnh với định dạng 04d.jpg
                    frame_path = os.path.join(images_video_path, 'image_{:04d}.jpg'.format(frame_count))
                    cv2.imwrite(frame_path, frame)
                    frame_count += 1

                cap.release()

print('Quá trình tạo hình ảnh từ video đã hoàn tất.')