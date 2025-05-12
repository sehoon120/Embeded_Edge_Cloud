# local_blaze_inference.py

import time
import cv2
import numpy as np
import mediapipe as mp
import insightface
import os

# ── 1) 스파이킹 설정 ─────────────────────────────────────────────
thresh   = 0.2
delta_t  = 2**(-10)
tau1     = 5e-3
decay1   = np.exp(-delta_t/tau1)
mem, spike = 0.0, 0

def act_fun(thresh, mem):
    return 1 if mem > thresh else 0

def mem_update(x, mem, spike):
    mem   = mem * decay1 * (1. - spike) + 0.1 * x
    spike = act_fun(thresh, mem)
    return mem, spike

# ── 2) Mediapipe 얼굴 감지 ─────────────────────────────────────
mp_fd         = mp.solutions.face_detection
face_detection = mp_fd.FaceDetection(model_selection=0, min_detection_confidence=0.6)

# ── 3) InsightFace 모델 로드 & 등록 얼굴 임베딩 ────────────────────
print("[INFO] InsightFace 모델 로드 중...")
model = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
model.prepare(ctx_id=0)
print("[INFO] 모델 준비 완료!")

REGISTER_DIR = 'registered_faces'
registered_faces = {}
for fname in os.listdir(REGISTER_DIR):
    name, ext = os.path.splitext(fname)
    if ext.lower() in ('.jpg', '.png'):
        img = cv2.imread(os.path.join(REGISTER_DIR, fname))
        faces = model.get(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if faces:
            registered_faces[name] = faces[0].embedding
            print(f"[REGISTER] {name}")
        else:
            print(f"[WARN] 얼굴 미검출: {fname}")

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ── 4) 카메라 열기 (웹캠 또는 Pi 카메라) ─────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("카메라 열기 실패")

frame_count       = 0
fps_start_time    = time.time()
inference_counter = 0

# ── 5) 메인 루프 ───────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] 프레임 읽기 실패")
        break

    # Mediapipe 얼굴 감지 신호값 x
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb)
    x       = 1.0 if results.detections else 0.0

    # 스파이킹 업데이트
    mem, spike = mem_update(x, mem, spike)
    # if spike:
    if True:    # ── 스파이크 로직 일단 무시, 매 프레임마다 바로 추론 ──
        inference_counter += 1
        print(f"\n[{inference_counter}] Spike 발생 → Inference 시작")

        # InsightFace 추론
        faces = model.get(rgb)
        if not faces:
            print("  얼굴 미검출")
        else:
            emb = faces[0].embedding
            best_name, best_score = 'unknown', -1.0
            for name, ref in registered_faces.items():
                score = cosine_similarity(emb, ref)
                if score > best_score:
                    best_score, best_name = score, name
            print(f"  결과: {best_name} ({best_score:.3f})")

        # FPS 측정
        frame_count += 1
        if frame_count >= 10:
            elapsed = time.time() - fps_start_time
            print(f"  [FPS] {frame_count/elapsed:.2f}")
            frame_count    = 0
            fps_start_time = time.time()

        time.sleep(0.5)   # spike 후 딜레이

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
