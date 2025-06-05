import cv2
import numpy as np
import insightface
import time
import os
from glob import glob

# 폴더 설정
REGISTER_DIR = 'registered_faces'
TEST_DIR = 'test_images'  # A, E, G, H, J 등의 폴더가 여기에 있어야 함

registered_faces = {'sehoon': None, 'jaeyoung': None}

print("[INFO] 모델 로드 중...")
model = insightface.app.FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
model.prepare(ctx_id=0)
print("[INFO] 모델 준비 완료!")

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

# 테스트용 이미지 파일들 불러오기
image_paths = glob(os.path.join(TEST_DIR, '*', '*.jpg'))  # ex) test_images/A/*.jpg
image_paths.sort()

for path in image_paths:
    print(f"\n[TEST] 이미지 파일: {path}")
    frame = cv2.imread(path)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faces = extract_faces(img_rgb)

    if len(faces) == 0:
        print("[INFO] 얼굴 없음")
        continue

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

        x1, y1, x2, y2 = bbox
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(display_frame, f"{best_name} ({best_score:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        print(f"[RESULT] 이름: {best_name}, 유사도: {best_score:.3f}, bbox: {bbox}")

    # 이미지 띄우기 (키 입력 대기 후 다음 이미지 진행)
    cv2.imshow("Test Image", display_frame)
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()
