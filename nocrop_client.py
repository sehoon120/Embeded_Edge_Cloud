import cv2
import requests
import time

# 얼굴 검출을 위한 Haar cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# CSI 카메라를 사용한 VideoCapture 초기화
cap = cv2.VideoCapture(0)  # PiCam 사용 시 0으로 두어도 됨

server_url = "http://<노트북_IP>:5000/infer"  # 서버 주소를 여기에 입력

while True:
    ret, frame = cap.read()
    if not ret:
        continue

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

        try:
            response = requests.post(server_url, files=files)
            print("서버 응답:", response.json())
        except Exception as e:
            print("서버 요청 실패:", e)

    # 디버깅용: 얼굴 표시
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow('Face Detection - Raspberry Pi', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

