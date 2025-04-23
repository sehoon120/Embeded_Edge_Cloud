# rpi_client.py
import requests
import time
from picamera2 import Picamera2
import cv2
import numpy as np

EC2_URL = 'http://52.79.154.43:8000/inference'

rtt_list = []
count = 0

# 얼굴 검출을 위한 Haar cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size" : (640, 480)}))
picam2.start()

empty_img = np.zeros((112, 112, 1), dtype=np.uint8)
_, empty_encoded = cv2. imencode('.jpg', empty_img)
default_files = {
    'image': ('default.jpg', empty_encoded.tobytes(), 'image/jpeg')
}

files = default_files

while True:
    
    frame =picam2.capture_array()

# 이미지 grayscale로 변환 후 얼굴 검출
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # 얼굴이 검출되면 첫 번째 얼굴만 전처리
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        face_img = frame[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (112, 112))  # InsightFace 입력 크기

        _, img_encoded = cv2.imencode('.jpg', face_img)
        files = {'image': ('face.jpg', img_encoded.tobytes(), 'image/jpeg')}

        # try:
        #     response = requests.post(server_url, files=files)
        #     print("서버 응답:", response.json())
        # except Exception as e:
        #     print("서버 요청 실패:", e)

    # 디버깅용: 얼굴 표시
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow('Face Detection - Raspberry Pi', frame)

    # with open('768_432.jpg', 'rb') as f:
        # files = {'image': ('768_432.jpg', f, 'image/jpeg')}

    start = time.time()
    try:
        response = requests.post(EC2_URL, files=files, timeout=5)
        end = time.time()

        rtt = (end - start) * 1000  # ms 단위
        rtt_list.append(rtt)
        count += 1

        avg_rtt = sum(rtt_list) / count

        print("Response:", response.json())
        print(f"RTT: {rtt:.2f} ms | 평균 RTT: {avg_rtt:.2f} ms")

    except requests.exceptions.RequestException as e:
        print("전송 실패:", e)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    time.sleep(0.5)

picam2.stop()
cv2.destroyAllWindows()