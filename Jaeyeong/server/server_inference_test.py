from flask import Flask, request, jsonify
import cv2
import numpy as np
import insightface
import os
import time
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

# 모델 준비
print("[INFO] 모델 로드 중...")

# 얼굴 인코딩 모델 (112x112 얼굴 입력용)
face_encoder = get_model('buffalo_l', download=False)
face_encoder.prepare(ctx_id=0)

print("[INFO] 모델 준비 완료!")

# 등록된 얼굴 사전 임베딩
REGISTER_DIR = 'registered_faces'
registered_faces = {
    'Ann_Veneman': None,
    'Edmund_Stoiber': None,
    'Gray_Davis' : None,
    'Hugo_Chavez' : None,
    'Jacques_Rogge' : None
}

# 등록 얼굴 임베딩 초기화
def initialize_registered_faces():
    for name in registered_faces:
        img_path = os.path.join(REGISTER_DIR, f'{name}.jpg')
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None and img.shape[:2] == (112, 112):
                embedding = face_encoder.get_feat(img)
                registered_faces[name] = embedding
                print(f"[INFO] 등록 완료: {name}")
            else:
                print(f"[WARNING] 등록 이미지 오류 (112x112 아님): {img_path}")
        else:
            print(f"[WARNING] 등록 이미지 없음: {img_path}")

initialize_registered_faces()

# 유사도 계산
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Flask 서버 설정
app = Flask(__name__)

@app.route('/inference', methods=['POST'])
def inference():
    recv_time = time.time()

    if 'image' not in request.files:
        return jsonify({'status': 'error', 'message': 'No image part'}), 400

    file = request.files['image']
    img_array = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({'status': 'error', 'message': 'Image decoding failed'}), 400

    if img.shape[:2] != (112, 112):
        return jsonify({'status': 'error', 'message': 'Image must be 112x112'}), 400

    embedding = face_encoder.get_feat(img)

    best_score = -1
    best_name = 'unknown'

    for name, ref_embedding in registered_faces.items():
        if ref_embedding is not None:
            score = cosine_similarity(embedding, ref_embedding)
            if score > best_score:
                best_score = score
                best_name = name

    print(f"[RESULT] 이름: {best_name} | 유사도: {best_score:.3f}")

    return jsonify({
        'status': 'success',
        'identified': best_name,
        'score': float(best_score),
        'recv_time': recv_time
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
