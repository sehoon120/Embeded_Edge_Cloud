import requests
import time
from picamera2 import Picamera2
import cv2
import numpy as np
import mediapipe as mp

# pip install mediapipe opencv-python

# EC2 서버 URL
EC2_URL = 'http://52.79.154.43:8000/inference'

# BlazeFace 설정
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6)

# Picamera2 초기화
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
picam2.start()

# 얼굴 없을 경우 기본 전송 이미지
empty_img = np.zeros((112, 112, 3), dtype=np.uint8)
_, empty_encoded = cv2.imencode('.jpg', empty_img)
default_files = {
    'image': ('default.jpg', empty_encoded.tobytes(), 'image/jpeg')
}

rtt_list = []
count = 0

try:
    while True:
        # 루프마다 초기화
        files = default_files

        # 이미지 캡처
        frame = picam2.capture_array()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(frame_rgb)

        if results.detections:
            # 첫 번째 얼굴만 처리
            detection = results.detections[0]
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x = int(bboxC.xmin * iw)
            y = int(bboxC.ymin * ih)
            w = int(bboxC.width * iw)
            h = int(bboxC.height * ih)

            # 바운딩 박스 자르기 및 resize
            face_img = frame[y:y+h, x:x+w]
            if face_img.size > 0:
                face_img = cv2.resize(face_img, (112, 112))
                _, img_encoded = cv2.imencode('.jpg', face_img)
                files = {'image': ('face.jpg', img_encoded.tobytes(), 'image/jpeg')}
        else:
            print("[INFO] 얼굴 미검출 - 기본 이미지 전송")

        # 디버깅용 얼굴 박스 표시
        if results.detections:
            for det in results.detections:
                bboxC = det.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x = int(bboxC.xmin * iw)
                y = int(bboxC.ymin * ih)
                w = int(bboxC.width * iw)
                h = int(bboxC.height * ih)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow('BlazeFace Detection - Raspberry Pi', frame)

        # 서버 전송
        start = time.time()
        try:
            response = requests.post(EC2_URL, files=files, timeout=5)
            end = time.time()
            rtt = (end - start) * 1000
            rtt_list.append(rtt)
            count += 1

            # 결과 출력
            if response.ok:
                result = response.json()
                print(f"[RESULT] 이름: {result.get('identified', 'unknown')} | 유사도: {result.get('score', 0):.3f}")
            else:
                print("[ERROR] 서버 응답 오류:", response.status_code)

            print(f"RTT: {rtt:.2f} ms | 평균 RTT: {sum(rtt_list)/count:.2f} ms")

        except requests.exceptions.RequestException as e:
            print("[ERROR] 전송 실패:", e)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(0.5)

except KeyboardInterrupt:
    print("[EXIT] 사용자 종료")

finally:
    face_detection.close()
    picam2.stop()
    cv2.destroyAllWindows()