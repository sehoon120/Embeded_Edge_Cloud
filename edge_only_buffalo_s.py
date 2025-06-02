import cv2
import numpy as np
import insightface
import time
import os
from picamera2 import Picamera2

# 등록된 얼굴 경로
REGISTER_DIR = 'registered_faces'

registered_faces = {
    'sehoon': None,
    'jaeyoung': None,
}

# 모델 로드 (경량 모델 추천)
print("[INFO] 모델 로드 중...")
model = insightface.app.FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
model.prepare(ctx_id=0)
print("[INFO] 모델 준비 완료!")

# 등록된 얼굴 임베딩 추출
def extract_faces(img):
    return model.get(img)

def initialize_registered_faces():
    for name in registered_faces:
        path = os.path.join(REGISTER_DIR, f'{name}.jpg')
        if os.path.exists(path):
            img = cv2.imread(path)
            faces = extract_faces(img)
            if len(faces) > 0:
                registered_faces[name] = faces[0].embedding
                print(f"[INFO] 등록 완료: {name}")
            else:
                print(f"[WARNING] 얼굴 감지 실패: {name}")
        else:
            print(f"[WARNING] 등록 이미지 없음: {name}")

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

initialize_registered_faces()

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
picam2.start()

face_prev = False

while True:
    frame = picam2.capture_array()
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = extract_faces(img_rgb)

    face_now = len(faces) > 0
    if face_now and not face_prev:
        print("[EVENT] 얼굴 인식됨 → 추론 및 결과 출력")
        display_frame = frame.copy()

        for face in faces:
            embedding = face.embedding
            bbox = face.bbox.astype(int).tolist()

            best_score = -1
            best_name = 'unknown'

            for name, ref in registered_faces.items():
                if ref is not None:
                    score = cosine_similarity(embedding, ref)
                    if score > best_score:
                        best_score = score
                        best_name = name

            if best_score < 0.4:
                best_name = 'unknown'

            x1, y1, x2, y2 = bbox
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display_frame, f"{best_name} ({best_score:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            print(f"[RESULT] 이름: {best_name}, 유사도: {best_score:.3f}, bbox: {bbox}")

        cv2.imshow("Recognition Result", display_frame)
        cv2.waitKey(1)
        time.sleep(0.5)
    else:
        time.sleep(0.1)

    face_prev = face_now

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


picam2.stop()
cv2.destroyAllWindows()
