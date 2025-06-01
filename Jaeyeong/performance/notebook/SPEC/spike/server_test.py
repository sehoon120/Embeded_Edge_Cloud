from flask import Flask, request, jsonify
import cv2
import numpy as np
import insightface
import os
import time
from insightface.app import FaceAnalysis

lfw_embeddings = {}

# 모델 로딩
print("[INFO] 모델 로딩 중 (buffalo_l)...")
face_model = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
face_model.prepare(ctx_id=0)
print("[INFO] 모델 준비 완료!")


# 얼굴 추출 함수
def extract_faces(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return face_model.get(img_rgb)

# 유사도 계산 함수
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Threshold 설정
THRESHOLD = 0.4

# Flask 앱 설정
app = Flask(__name__)

@app.route('/inference', methods=['POST'])
def inference():
    recv_time = time.time()

    if 'image' not in request.files:
        return jsonify({'status': 'error', 'message': 'No image uploaded'}), 400

    file = request.files['image']
    img_array = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({'status': 'error', 'message': 'Invalid image data'}), 400

    faces = extract_faces(img)

    if not faces:
        return jsonify({'status': 'success', 'results': [], 'recv_time': recv_time})

    results = []
    for face in faces:
        emb = face.embedding
        best_score = -1.0
        best_name = '-'

        for name, emb_list in lfw_embeddings.items():
            for ref_emb in emb_list:
                score = cosine_similarity(emb, ref_emb)
                if score > best_score:
                    best_score = score
                    best_name = name

        if best_score < THRESHOLD:
            best_name = '-'

        # 응답용 (float), 그리기용 (int) bbox 모두 생성
        bbox = face.bbox.astype(int).tolist()

        results.append({
            'identified': best_name,
            'score': round(float(best_score), 4),
            'bbox': bbox
        })

    print(f"[RESULT] 총 {len(results)}명 인식됨")
    for r in results:
        print(f" - 이름: {r['identified']} | 유사도: {r['score']} | bbox: {r['bbox']}")

    return jsonify({
        'status': 'success',
        'results': results,
        'recv_time': recv_time
    })

if __name__ == '__main__':
    
    print("[INFO] 전체 LFW 인물 임베딩 로드 시작...")
    lfw_embeddings = np.load("lfw_embeddings.npy", allow_pickle=True).item()
    print(f"[INFO] 총 {len(lfw_embeddings)}명 임베딩 로드됨") 
    
    # Flask 실행
    app.run(host='0.0.0.0', port=5000)
