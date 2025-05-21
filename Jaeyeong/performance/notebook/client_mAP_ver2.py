import cv2
import numpy as np
import insightface
import time
import os
from glob import glob

# 폴더 설정
REGISTER_DIR = r'/home/pi/Project/Embeded_Edge_Cloud/registered_face'
TEST_DIR = r'/home/pi/Project/Embeded_Edge_Cloud/Jaeyeong/performance/dataset'
OUTPUT_DIR = 'output'

os.makedirs(OUTPUT_DIR, exist_ok=True)

registered_faces = {
    'Ann_Veneman': None,
    'Edmund_Stoiber': None,
    'Gray_Davis': None,
    'Hugo_Chavez': None,
    'Jacques_Rogge': None
}

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

rtt_list = []
count = 0
initialize_registered_faces()

# 테스트용 이미지 파일들 불러오기
image_paths = glob(os.path.join(TEST_DIR, '*', '*.jpg'))
image_paths.sort()

for path in image_paths:
    print(f"\n[TEST] 이미지 파일: {path}")
    start = time.time()
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

        # 등록되지 않은 얼굴 처리
        if best_name == '-' or best_name.lower() == 'unknown':
            best_name = '-'
            print(f"[RESULT] 등록되지 않은 인물 → 성공 처리됨")
        else:
            print(f"[RESULT] 이름: {best_name}, 유사도: {best_score:.3f}, bbox: {bbox}")

        x1, y1, x2, y2 = bbox
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(display_frame, f"{best_name} ({best_score:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        end = time.time()
        rtt = end - start
        rtt_list.append(rtt)
        count += 1
        print(f"[{count}] RTT: {rtt:.4f} sec | avg RTT: {sum(rtt_list)/count:.4f} sec")

        # 결과 이미지 저장
        output_filename = os.path.basename(path)
        output_path = os.path.join(OUTPUT_DIR, f"result_{count:04d}_{best_name}.jpg")
        cv2.imwrite(output_path, display_frame)
        print(f"[INFO] 저장됨: {output_path}")


cv2.destroyAllWindows()