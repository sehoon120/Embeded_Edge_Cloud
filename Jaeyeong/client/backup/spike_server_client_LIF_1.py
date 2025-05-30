import requests
import time
from picamera2 import Picamera2
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import os

# server address setting
URL = 'http://52.79.154.43:8000/inference'

###########################################################################################################################################

# spiking method base setting
thresh = 0.2
delta_t = 2**(-10)
tau1 = 5e-3
decay1 = np.exp(-delta_t/tau1)

mem = 0
spike = 0

def act_fun(thresh, mem):
	if mem > thresh:
		return 1
        #return torch.tensor(1)
	else:
		return 0
        #return torch.tensor(0)

def mem_update(x, mem, spike):
    mem = mem * decay1 * (1. - spike) + 0.1 * x
    spike = act_fun(thresh, mem)
    return mem, spike
###########################################################################################################################################

rtt_list = []
count = 0

# YOLOv8n model load
model = YOLO('yolov8n.pt')  # Download only on first time
model.fuse()  # Model Optimizing

# Haarcascades classifier load
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
#picam2.configure(picam2.create_preview_configuration(main={"size": (320, 240)}))
picam2.start()

empty_img = np.zeros((112, 112, 1), dtype=np.uint8)
_, empty_encoded = cv2.imencode('.jpg', empty_img)
default_files = {
    'image': ('default.jpg', empty_encoded.tobytes(), 'image/jpeg')
}

if not os.path.exists('spike_frames'):
    os.makedirs('spike_frames')

def detect_person_and_prepare_file(frame, model, face_cascade, default_files, mem, spike):
    # YOLOv8 Inference

    filename = None
    files = default_files
    results = model.predict(frame, imgsz=640, conf=0.3, verbose=False)
    #results = model.predict(frame, imgsz=320, conf=0.3, verbose=False)
    x = 0 # x means person_detected
    for result in results:
        classes = result.boxes.cls.cpu().numpy()
        
        if 0 in classes:
            x = 1
        else:
            x = 0
        mem, spike = mem_update(x, mem , spike)
        
        if spike == 1:
            print("Spike occurs")
            boxes = result.boxes.xyxy.cpu().numpy()
            
            timestamp = int(time.time() * 1000)
            filename = f"spike_frames/frame_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            person_indices = np.where(classes == 0)[0]
            if len(person_indices) > 0:
                areas = []
                # Select only the biggest box
                for idx in person_indices:
                    x1, y1, x2, y2 = boxes[idx]
                    area = (x2 - x1) * (y2 - y1)
                    areas.append(area)
                max_area_idx = person_indices[np.argmax(areas)]
                
                #person_boxes = boxes[person_indices]
                #areas = (person_boxes[:, 2] - person_boxes[:, 0]) * (person_boxes[:, 3] - person_boxes[:, 1])
                #max_area_idx = person_indices[np.argmax(areas)]
                
                # crop the biggest person
                x1, y1, x2, y2 = map(int, boxes[max_area_idx])
                person_crop = frame[y1:y2, x1:x2]
                gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                if len(faces) > 0:
                    (fx, fy, fw, fh) = faces[0]
                    face_img = person_crop[fy:fy+fh, fx:fx+fw]
                    face_img = cv2.resize(face_img, (112, 112))

                    _, img_encoded = cv2.imencode('.jpg', face_img)
                    files = {'image': ('face.jpg', img_encoded.tobytes(), 'image/jpeg')}

            break
    return files, mem, spike, filename

while True:
    frame = picam2.capture_array()
    files, mem, spike, filename = detect_person_and_prepare_file(
        frame, model, face_cascade, default_files, mem, spike)

    if spike == 1:
        start = time.time()
        try:
            print("Try Connection!")
            response = requests.post(URL, files=files)
            if response.status_code == 200:
                end = time.time()
                rtt = end - start
                rtt_list.append(rtt)
                count += 1
                print(f"[{count}] RTT: {rtt:.4f} seconds")
                
                try:
                    server_result = response.json()
                    print(f"[{count}] Server Result: {server_result['result']}")
                except Exception as e:
                    print(f"Error parsing server response: {e}")
                
                if filename is not None:
                    img = cv2.imread(filename)
                    if img is not None:
                        cv2.imshow('Spike Frame', img)
                time.sleep(0.5)        
                
            else:
                print(f"Request failed with status code: {response.status_code}")
        except Exception as e:
            print(f"Request exception: {e}")

    # cv2 end code
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    #time.sleep(0.5)
    time.sleep(0.1)

picam2.stop()
cv2.destroyAllWindows()
