import time
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import os

# 기본 설정
thresh = 0.15
delta_t = 2**(-10)
tau1, tau2 = 25e-3, 2.5e-3
const = tau1 / (tau1 - tau2)
decay1 = np.exp(-delta_t/tau1)
decay2 = np.exp(-delta_t/tau2)

mem1 = 0
mem2 = 0.01
mem_list = []
spike = 0
x_list = []
spike_list = []

def act_fun(thresh, mem):
    if mem > thresh:
        return 1
    else:
        return 0

def mem_update(x, mem1, mem2, spike):
    mem1 = mem1 * decay1 * (1. - spike) + const * 0.1 * x
    mem2 = mem2 * decay2 * (1. - spike) + const * 0.1 * x
    mem = mem1 - mem2
    spike = act_fun(thresh, mem)
    return mem, mem1, mem2, spike

# Webcam 초기화
cap = cv2.VideoCapture(0)  # 0번 카메라 (노트북 기본 카메라)

empty_img = np.zeros((112, 112, 1), dtype=np.uint8)
_, empty_encoded = cv2.imencode('.jpg', empty_img)
default_files = {
    'image': ('default.jpg', empty_encoded.tobytes(), 'image/jpeg')
}

if not os.path.exists('spike_frames'):
    os.makedirs('spike_frames')

# YOLOv8n 모델 로드
model = YOLO('yolov8n.pt')
model.fuse()

# Haar Classifier 로드
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_person_and_prepare_file(frame, model, face_cascade, default_files, mem1, mem2, spike, mem_list, x_list, spike_list):
    filename = None
    files = default_files
    results = model.predict(frame, imgsz=640, conf=0.3, verbose=False)
    x = 0
    for result in results:
        classes = result.boxes.cls.cpu().numpy()
        boxes = result.boxes.xyxy.cpu().numpy()

        if 0 in classes:
            x = 1
        else:
            x = 0
        mem, mem1, mem2, spike = mem_update(x, mem1, mem2, spike)
        mem_list.append(mem)
        x_list.append(x)
        spike_list.append(spike)

        if spike == 1:
            timestamp = int(time.time() * 1000)
            filename = f"spike_frames/frame_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            person_indices = np.where(classes == 0)[0]
            if len(person_indices) > 0:
                areas = []
                for idx in person_indices:
                    x1, y1, x2, y2 = boxes[idx]
                    area = (x2 - x1) * (y2 - y1)
                    areas.append(area)

                max_area_idx = person_indices[np.argmax(areas)]

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
    return files, mem1, mem2, spike, filename

while True:
    ret, frame = cap.read()
    if ret:
        files, mem1, mem2, spike, filename = detect_person_and_prepare_file(
            frame, model, face_cascade, default_files, mem1, mem2, spike, mem_list, x_list, spike_list
        )
        #cv2.imshow('Live', frame)
    else:
        print("카메라에서 프레임을 가져오지 못했습니다.")
        continue
    
    if spike == 1:
        if filename is not None:
            img = cv2.imread(filename)
            if img is not None:
                cv2.imshow('Spike Frame', img)
                #cv2.waitKey(1)
        time.sleep(0.5)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    #time.sleep(0.5)
    time.sleep(0.1)
    
cap.release()
cv2.destroyAllWindows()