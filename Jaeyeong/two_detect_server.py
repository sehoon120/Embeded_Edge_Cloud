from flask import Flask, request, jsonify
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from insightface.app import FaceAnalysis
import os

# Flask 앱
app = Flask(__name__)

# 모델 준비
face_app = FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=0)

# 사람 이름과 임베딩을 dict로 불러오기
face_db = {
    "jaeyoung": np.load("han_embedding.npy"),
    "sehoon": np.load("minji_embedding.npy")
}

# threshold 설정 (cosine similarity 기준)
THRESHOLD = 0.45

@app.route('/infer', methods=['POST'])
def infer():
    file = request.files['image']
    np_img = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    faces = face_app.get(img)
    if not faces:
        return jsonify({'result': 'No face detected'})

    face = faces[0]
    input_embedding = face.embedding.reshape(1, -1)

    best_match = None
    best_score = -1

    for name, db_embedding in face_db.items():
        db_embedding = db_embedding.reshape(1, -1)
        score = cosine_similarity(db_embedding, input_embedding)[0][0]
        if score > best_score:
            best_score = score
            best_match = name

    if best_score >= THRESHOLD:
        return jsonify({
            'result': best_match,
            'similarity': float(best_score)
        })
    else:
        return jsonify({
            'result': 'unknown',
            'similarity': float(best_score)
        })

app.run(host='0.0.0.0', port=5000)
