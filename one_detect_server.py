from flask import Flask, request, jsonify
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
import os

# Flask 앱
app = Flask(__name__)

# 모델 준비
face_app = FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=0)  # GPU 사용, CPU는 -1

# 내 얼굴 임베딩 불러오기
my_embedding = np.load("han_embedding.npy")  # 사전 등록된 평균 임베딩
my_embedding = my_embedding.reshape(1, -1)

# 유사도 기준 threshold (조정 가능, 일반적으로 0.4~0.5 이상이면 같은 사람으로 판단)
THRESHOLD = 0.45

@app.route('/infer', methods=['POST'])
def infer():
    file = request.files['image']
    np_img = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    faces = face_app.get(img)
    if not faces:
        return jsonify({'result': 'No face detected'})

    # 첫 번째 얼굴만 사용
    face = faces[0]
    input_embedding = face.embedding.reshape(1, -1)

    # cosine similarity 계산
    similarity = cosine_similarity(my_embedding, input_embedding)[0][0]

    is_me = similarity >= THRESHOLD
    return jsonify({
        'result': 'me' if is_me else 'not me',
        'similarity': float(similarity)
    })

app.run(host='0.0.0.0', port=5000)

