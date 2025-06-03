import cv2
import numpy as np
import os
import time
import onnxruntime as ort
import mediapipe as mp
from picamera2 import Picamera2

# pip install onnxruntime

# 경로 설정     에지에서 재설정
REGISTER_DIR = r"C:\Embeded_Project\registered_faces"
ONNX_PATH = r'C:\Embeded_Project\Embeded_Edge_Cloud\Sehoon\models\buffalo_s_recognition_fp16.onnx'

registered_faces = {
    'sehoon': None,
    'jaeyoung': None
    # 'A': None,
    # 'B': None,
    # 'C': None
}

# Spiking parameter
thresh = 0.2
delta_t = 2**(-10)
tau1 = 5e-3
decay1 = np.exp(-delta_t / tau1)
mem = 0
spike = 0

def act_fun(thresh, mem):
    return 1 if mem > thresh else 0

def mem_update(x, mem, spike):
    mem = mem * decay1 * (1. - spike) + 0.1 * x
    spike = act_fun(thresh, mem)
    return mem, spike

# 얼굴 인식 ONNX 모델 로드
try:
    onnx_session = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
    input_name = onnx_session.get_inputs()[0].name
    print(f"[INFO] ONNX 모델 로딩 성공: 입력 이름 = {input_name}")
except Exception as e:
    print(f"[ERROR] ONNX 모델 로딩 실패: {e}")

def preprocess_face(crop_img):
    img = cv2.resize(crop_img, (112, 112))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img.astype(np.float16)

def extract_embedding(crop_img):
    input_tensor = preprocess_face(crop_img)
    input_tensor = input_tensor.astype(np.float16)
    return onnx_session.run(None, {input_name: input_tensor})[0][0]

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# 얼굴 등록 초기화
print("[INFO] 등록된 얼굴 로딩 중...")
mp_face_detection = mp.solutions.face_detection
model = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6)

try:
    for name in registered_faces:
        path = os.path.join(REGISTER_DIR, f"{name}.jpg")
        if os.path.exists(path):
            img = cv2.imread(path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = model.process(img_rgb)

            if result.detections:
                bbox = result.detections[0].location_data.relative_bounding_box
                h, w, _ = img.shape
                x1 = max(int(bbox.xmin * w), 0)
                y1 = max(int(bbox.ymin * h), 0)
                x2 = min(int(x1 + bbox.width * w), w)
                y2 = min(int(y1 + bbox.height * h), h)
                face_crop = img[y1:y2, x1:x2]
                if face_crop.size == 0:
                    print(f"[WARNING] 잘못된 crop: {name}")
                    continue
                registered_faces[name] = extract_embedding(face_crop)
                print(f"[INFO] 등록 완료: {name}")
            else:
                print(f"[WARNING] 얼굴 감지 실패: {name}")
        else:
            print(f"[WARNING] 등록 이미지 없음: {name}")
except Exception as e:
    print(f"[ERROR] 예외 발생: {e}")

model.close()

# 확인용 출력
# print("\n[RESULT] 등록된 임베딩:")
# for name, embedding in registered_faces.items():
#     if embedding is not None:
#         print(f"- {name}: 임베딩 생성 완료 (shape={embedding.shape})")
#     else:
#         print(f"- {name}: 생성 실패")


# 실시간 처리 시작
print("[INFO] 얼굴 인식 시작")
mp_fd = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6)
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
picam2.start()

rtt_list = []
frame_count = 0
fps_start_time = time.time()
inference_count = 0

while True:
    frame = picam2.capture_array()
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_fd.process(img_rgb)
    x = 1.0 if results.detections else 0.0
    mem, spike = mem_update(x, mem, spike)

    display_frame = frame.copy()

    if spike and results.detections:
        start_time = time.time()

        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x1 = int(bbox.xmin * w)
            y1 = int(bbox.ymin * h)
            x2 = int(x1 + bbox.width * w)
            y2 = int(y1 + bbox.height * h)

            face_crop = frame[y1:y2, x1:x2]
            embedding = extract_embedding(face_crop)

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

            # 결과 표시
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display_frame, f"{best_name} ({best_score:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            print(f"[RESULT] 이름: {best_name}, 유사도: {best_score:.3f}")

        end_time = time.time()
        rtt = end_time - start_time
        rtt_list.append(rtt)
        inference_count += 1

        print(f"[{inference_count}] RTT: {rtt:.4f} sec | avg RTT: {sum(rtt_list)/len(rtt_list):.4f} sec")

        frame_count += 1
        if frame_count >= 10:
            elapsed = time.time() - fps_start_time
            fps = frame_count / elapsed
            print(f"[FPS] avg FPS: {fps:.2f}")
            frame_count = 0
            fps_start_time = time.time()

        time.sleep(0.5)

    else:
        time.sleep(0.1)

    cv2.imshow("Face Recognition", display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

picam2.stop()
mp_fd.close()
cv2.destroyAllWindows()
