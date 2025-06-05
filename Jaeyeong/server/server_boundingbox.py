# server.py
from flask import Flask, request, jsonify
import cv2
import numpy as np
import insightface
import os
import time
# pip install --upgrade pip setuptools wheel
# pip install insightface==0.7.3
# pip install flask flask-cors
# pip install onnxruntime-gpu --pre


# /home/ubuntu/server.py



# InsightFace 모델 준비 (GPU 사용)
print("[INFO] 모델 로드 중...")
model = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
model.prepare(ctx_id=0)
print("[INFO] 모델 준비 완료!")

# 등록된 얼굴 사전 임베딩
REGISTER_DIR = 'registered_faces'
registered_faces = {
    'sehoon': None,
    'jaeyoung': None,
}

# 임베딩 추출 함수
def extract_faces(img):
    return model.get(img)

# 등록 얼굴 임베딩 초기화
def initialize_registered_faces():
    for name in registered_faces:
        img_path = os.path.join(REGISTER_DIR, f'{name}.jpg')
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            faces = extract_faces(img)
            if len(faces) > 0:
                registered_faces[name] = faces[0].embedding
                print(f"[INFO] 등록 완료: {name}")
            else:
                print(f"[WARNING] 얼굴 감지 실패: {img_path}")
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

    faces = extract_faces(img)

    if len(faces) == 0:
        return jsonify({'status': 'fail', 'message': 'No face detected'}), 200

    results = []

    for face in faces:
        embedding = face.embedding
        bbox = face.bbox.astype(int).tolist()

        best_score = -1
        best_name = 'unknown'

        for name, ref_embedding in registered_faces.items():
            if ref_embedding is not None:
                score = cosine_similarity(embedding, ref_embedding)
                if score > best_score:
                    best_score = score
                    best_name = name

        results.append({
            'identified': best_name,
            'score': float(best_score),
            'bbox': bbox
        })

        print(f"[RESULT] 이름: {best_name} | 유사도: {best_score:.3f} | BBOX: {bbox}")

    return jsonify({
        'status': 'success',
        'num_faces': len(results),
        'results': results,
        'recv_time': recv_time
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
