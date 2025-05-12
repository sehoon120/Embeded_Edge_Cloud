# batch_test.py
import os
import cv2
import numpy as np
import insightface
import mediapipe as mp

# 1) InsightFace 모델 로드
print("[INFO] 모델 로드 중...")
model = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
model.prepare(ctx_id=0)
print("[INFO] 모델 준비 완료!")

# # 모델 로드 직후에 추가
# print(">>> ONNX Runtime providers:", model.session.get_providers())

# 2) 등록된 얼굴 임베딩 불러오기
REGISTER_DIR = r"C:\Embeded_Project\registered_faces"
registered = {}
for fname in os.listdir(REGISTER_DIR):
    name, ext = os.path.splitext(fname)
    if ext.lower() in ('.jpg', '.png'):
        img = cv2.imread(os.path.join(REGISTER_DIR, fname))
        if img is None:
            print(f"[WARN] '{fname}' 읽기 실패")
            continue
        faces = model.get(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if faces:
            registered[name] = faces[0].embedding
            print(f"[REGISTERED] {name}")
        else:
            print(f"[WARN] '{fname}'에서 얼굴 미검출")

# 3) (선택) Mediapipe 얼굴 감지 준비
mp_fd          = mp.solutions.face_detection
face_detect   = mp_fd.FaceDetection(model_selection=0, min_detection_confidence=0.6)

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 4) 테스트 이미지 순회
TEST_DIR = r"C:\Embeded_Project\img"
print("\n[START] Batch inference")
for fname in sorted(os.listdir(TEST_DIR)):
    if not fname.lower().endswith(('.jpg', '.png')):
        continue

    path = os.path.join(TEST_DIR, fname)
    img  = cv2.imread(path)
    if img is None:
        print(f"[ERROR] '{fname}' 읽기 실패")
        continue

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # (선택) Mediapipe로 먼저 얼굴이 있는지 빠르게 확인
    det = face_detect.process(rgb)
    if not det.detections:
        print(f"{fname:12} | 얼굴 미검출 → 스킵")
        continue

    # InsightFace로 추론
    faces = model.get(rgb)
    if not faces:
        print(f"{fname:12} | InsightFace도 얼굴 미검출")
        continue

    emb = faces[0].embedding
    # 등록자 중 최고 유사도 찾기
    best_name, best_score = 'unknown', -1.0
    for name, ref in registered.items():
        score = cosine_sim(emb, ref)
        if score > best_score:
            best_score, best_name = score, name

    print(f"{fname:12} | → {best_name:<8} ({best_score:.3f})")

print("[DONE] Batch inference")
