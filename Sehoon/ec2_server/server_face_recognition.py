from flask import Flask, request, jsonify
import cv2
import numpy as np
import insightface
import time
import os

# pip install insightface onnxruntime opencv-python-headless numpy flask


# 모델 로드 (ArcFace 모델)
model = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
model.prepare(ctx_id=0)

# 등록된 얼굴 (사전 임베딩)
registered_faces = {
    'alice': None,
    'bob': None,
}

# 이미지에서 얼굴 임베딩 추출 함수
def extract_embedding(img_bytes):
    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    faces = model.get(img)
    if len(faces) == 0:
        return None

    return faces[0].embedding  # 일단 가장 큰 얼굴 하나만

# 등록된 얼굴 임베딩 초기화 (서버 실행 시)
def initialize_registered_faces():
    for name in registered_faces:
        img_path = f'registered_faces/{name}.jpg'
        if os.path.exists(img_path):
            with open(img_path, 'rb') as f:
                embedding = extract_embedding(f.read())
                registered_faces[name] = embedding

initialize_registered_faces()

# 거리 계산
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

app = Flask(__name__)

@app.route('/inference', methods=['POST'])
def inference():
    recv_time = time.time()

    if 'image' not in request.files:
        return jsonify({'status': 'error', 'message': 'No image part'}), 400

    file = request.files['image']
    embedding = extract_embedding(file.read())

    if embedding is None:
        return jsonify({'status': 'fail', 'message': 'No face detected'}), 200

    best_score = -1
    best_name = 'unknown'

    for name, ref_embedding in registered_faces.items():
        if ref_embedding is None:
            continue
        score = cosine_similarity(embedding, ref_embedding)
        if score > best_score:
            best_score = score
            best_name = name

    return jsonify({
        'status': 'success',
        'identified': best_name,
        'score': float(best_score),
        'recv_time': recv_time
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
