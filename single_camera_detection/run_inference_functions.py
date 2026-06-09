from ultralytics import YOLO
from tqdm import tqdm
import os 
import cv2

def save_detections_to_txt(detections, txt_path, normed=True):
    with open(txt_path, 'w') as f:
        for det in detections[0].boxes:
            cls = det.cls.item()
            box = det.xywhn.tolist()[0] if normed else det.xywh.tolist()[0]
            conf = det.conf.item()

            line = str(int(cls))
            for b in box:
                line += ' ' + str(round(b, 6))
            line += ' ' + str(round(conf, 6)) + '\n'

            f.write(line)

def process_images_with_object_detection(images_folder, output_folder, model_path, image_size):
    model = YOLO(model_path)

    for image in tqdm(os.listdir(images_folder)):
        image_path = os.path.join(images_folder, image)
        img = cv2.imread(image_path)
        results = model(img, imgsz=image_size)

        label = image.replace('.jpg', '.txt')

        save_path = os.path.join(output_folder, label)
        save_detections_to_txt(results, save_path)

def process_video_with_object_detection(video_path, output_folder, model_path, image_size):
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_number = 1
    for i in tqdm(range(frame_count)):
        ret, frame = cap.read()
        if not ret:
            print(f"Error reading video: {video_path}")
            break

        results = model(frame, imgsz=image_size)

        label_name = "%06d.txt" % frame_number
        label = os.path.join(output_folder, label_name)

        save_detections_to_txt(results, label, normed=False)

        frame_number += 1

    cap.release()