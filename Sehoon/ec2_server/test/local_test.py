import cv2
import numpy as np
import os
import insightface
from insightface.app import FaceAnalysis
import onnxruntime as ort


# GPU용 providers 설정
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

# 모델 로드
model = FaceAnalysis(name='buffalo_l', providers=providers)
model.prepare(ctx_id=0)  # ctx_id=0 → GPU 사용

print("ONNX Runtime available providers:", ort.get_available_providers())

# 등록 얼굴 폴더
registered_faces = {}
reg_folder = r'C:\Embeded_Project\registered_faces'

def extract_embedding(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"[WARNING] 이미지 로드 실패: {img_path}")
        return None
    faces = model.get(img)
    if len(faces) == 0:
        print(f"[WARNING] 얼굴 미검출: {img_path}")
        return None
    return faces[0].embedding

# 등록된 얼굴 임베딩 생성
for fname in os.listdir(reg_folder):
    if fname.lower().endswith(('.jpg', '.png')):
        name = os.path.splitext(fname)[0]
        emb = extract_embedding(os.path.join(reg_folder, fname))
        if emb is not None:
            registered_faces[name] = emb
            print(f"[INFO] 등록 완료: {name}")

# 테스트 이미지 폴더 및 이름 패턴
test_folder = r'C:\Embeded_Project\img'
test_pattern = 'test_face_'

# 테스트 이미지 0 ~ 8번 순회
for i in range(9):
    test_img_path = os.path.join(test_folder, f'{test_pattern}{i}.jpg')
    test_emb = extract_embedding(test_img_path)

    if test_emb is None:
        continue

    # cosine similarity 계산
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    # 유사도 비교
    best_score = -1
    best_name = 'unknown'
    for name, ref_emb in registered_faces.items():
        score = cosine_similarity(test_emb, ref_emb)
        print(f"[{test_pattern}{i}.jpg] {name} similarity: {score:.3f}")
        if score > best_score:
            best_score = score
            best_name = name

    # 결과 출력
    print(f"[RESULT] → 최종 매칭: {best_name} (유사도: {best_score:.3f})\n")
