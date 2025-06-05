# pip install mediapipe opencv-python tflite-runtime numpy

# project/
# │
# ├── main.py
# ├── models/
# │   └── mobile_face_net.tflite
# └── registered_faces/
#     ├── sehoon.jpg
#     └── jaeyoung.jpg



import cv2
import mediapipe as mp
import numpy as np
import tflite_runtime.interpreter as tflite
from numpy.linalg import norm
import os
from picamera2 import Picamera2
import time

# ---------- MobileFaceNet TFLite 모델 로드 ----------
interpreter = tflite.Interpreter(model_path="models/mobile_face_net.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_face(face_img):
    img = cv2.resize(face_img, (112, 112))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def get_embedding(face_img):
    img_input = preprocess_face(face_img)
    interpreter.set_tensor(input_details[0]['index'], img_input)
    interpreter.invoke()
    embedding = interpreter.get_tensor(output_details[0]['index'])
    return embedding[0]

# ---------- 등록된 얼굴 임베딩 ----------
REGISTER_DIR = 'registered_faces'
registered_faces = {}

def initialize_registered_faces():
    for filename in os.listdir(REGISTER_DIR):
        name, _ = os.path.splitext(filename)
        path = os.path.join(REGISTER_DIR, filename)
        img = cv2.imread(path)
        if img is None:
            continue
        emb = get_embedding(img)
        registered_faces[name] = emb
        print(f"[INFO] 등록됨: {name}")

def recognize_face(face_img):
    emb = get_embedding(face_img)
    min_dist = float('inf')
    identity = "Unknown"

    for name, reg_emb in registered_faces.items():
        dist = norm(emb - reg_emb)
        if dist < min_dist and dist < 0.8:
            min_dist = dist
            identity = name

    return identity

# ---------- 얼굴 탐지기 (BlazeFace via MediaPipe) ----------
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6)

# ---------- Picamera2 초기화 ----------
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()
time.sleep(2)  # Warm-up time

# ---------- 얼굴 인식 루프 ----------
initialize_registered_faces()

while True:
    frame = picam2.capture_array()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb)

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x1 = int(bboxC.xmin * w)
            y1 = int(bboxC.ymin * h)
            x2 = x1 + int(bboxC.width * w)
            y2 = y1 + int(bboxC.height * h)

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            face_crop = frame[y1:y2, x1:x2]

            identity = recognize_face(face_crop)
            label = f"{identity}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Face Recognition (Picamera2)", frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

cv2.destroyAllWindows()
picam2.stop()
