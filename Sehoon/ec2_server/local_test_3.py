import cv2
import numpy as np
import insightface
import mediapipe as mp
import os
import onnxruntime as ort

# 폴더 경로
REGISTER_DIR = r'C:\Embeded_Project\Embeded_Edge_Cloud\Sehoon\ec2_server\registered_faces'
TEST_DIR = r'C:\Embeded_Project\Embeded_Edge_Cloud\Sehoon\ec2_server\img'

registered_faces = {'sehoon': None, 'jaeyoung': None}

# BlazeFace (mediapipe)
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6)

# InsightFace (GPU)
model = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
model.prepare(ctx_id=0)
recognition_model = model.models['recognition']

print("ONNX Runtime available providers:", ort.get_available_providers())

# 임베딩 추출 함수
def extract_embedding(img):
    if img.shape[0] != 112 or img.shape[1] != 112:
        img = cv2.resize(img, (112, 112))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # BGR → RGB
    img = img.astype(np.float32)                # uint8 → float32
    print('00')
    embedding = recognition_model.get(img)
    print('11')
    return embedding


def extract_embedding_init(img):
    faces = model.get(img)
    if len(faces) == 0:
        return None
    return faces[0].embedding


# 등록 얼굴 임베딩 초기화
# def initialize_registered_faces():
#     for name in registered_faces:
#         img_path = os.path.join(REGISTER_DIR, f'{name}.jpg')
#         if os.path.exists(img_path):
#             img = cv2.imread(img_path)
#             if img is None:
#                 print(f"[WARNING] 등록 이미지 로드 실패: {img_path}")
#                 continue
#             embedding = extract_embedding_init(img)
#             registered_faces[name] = embedding
#             print(f"[INFO] 등록 완료: {name}")
#         else:
#             print(f"[WARNING] 등록 이미지 없음: {img_path}")


def initialize_registered_faces():
    for name in registered_faces:
        img_path = os.path.join(REGISTER_DIR, f'{name}.jpg')
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            emb = extract_embedding_init(img)
            registered_faces[name] = emb
            print(f"[INFO] 등록 완료: {name}")

initialize_registered_faces()

# cosine similarity 계산
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 테스트 폴더 이미지 순회
for fname in sorted(os.listdir(TEST_DIR)):
    img_path = os.path.join(TEST_DIR, fname)
    img = cv2.imread(img_path)
    if img is None:
        print(f"[WARNING] 이미지 로드 실패: {img_path}")
        continue
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detection.process(img_rgb)

    if results.detections:
        detection = results.detections[0]
        ih, iw, _ = img.shape
        bboxC = detection.location_data.relative_bounding_box
        x = int(bboxC.xmin * iw)
        y = int(bboxC.ymin * ih)
        w = int(bboxC.width * iw)
        h = int(bboxC.height * ih)

        # padding 추가
        pad = 20
        x1 = max(x - pad, 0)
        y1 = max(y - pad, 0)
        x2 = min(x + w + pad, iw)
        y2 = min(y + h + pad, ih)
        face_img = img[y1:y2, x1:x2]
        print('0')
        if face_img.size > 0:
            embedding = extract_embedding(face_img)
            print('1')

            if embedding is not None:
                best_score = -1
                best_name = 'unknown'
                for name, ref_embedding in registered_faces.items():
                    if ref_embedding is not None:
                        score = cosine_similarity(embedding, ref_embedding)
                        if score > best_score:
                            best_score = score
                            best_name = name
                print(f"[RESULT] {fname} → 최종 매칭: {best_name} | 유사도: {best_score:.3f}")
            else:
                print(f"[WARNING] 얼굴 임베딩 추출 실패: {fname}")
        else:
            print(f"[WARNING] 얼굴 crop 실패: {fname}")
    else:
        print(f"[WARNING] 얼굴 미검출: {fname}")

face_detection.close()
