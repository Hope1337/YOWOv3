import os
import cv2

def extract_frames(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    frame_count = 0
    ff          = 0
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Trích xuất frame mỗi 1/30 giây (30fps)
        if frame_count % int(fps/30) == 0:
            cv2.imwrite(os.path.join(output_folder, "{:05d}.jpg".format(ff)), frame)
            ff += 1
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()

root_folder = '/home/manh/Datasets/RWF-2000'
for dataset_type in ['train', 'val']:
    dataset_path = os.path.join(root_folder, dataset_type)
    for label in ['Fight', 'NonFight']:
        label_path = os.path.join(dataset_path, label)
        for video_file in os.listdir(label_path):
            video_path = os.path.join(label_path, video_file)

            t = video_file.split('.')[0]
            output_folder = os.path.join(label_path, t)
            os.makedirs(output_folder, exist_ok=True)
            extract_frames(video_path, output_folder)