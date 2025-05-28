from flask import Flask, request, jsonify
import cv2
import numpy as np
import insightface
import os
import time
from insightface.app import FaceAnalysis

# 등록 이미지 경로
REGISTER_DIR = r'/home/han/ESD/Embeded_Edge_Cloud/Jaeyeong/performance/notebook/dataset/'

# 등록된 이름 리스트
registered_faces = {
    'Ann_Veneman': None,
    'Edmund_Stoiber': None,
    'Gray_Davis': None,
    'Hugo_Chavez': None,
    'Jacques_Rogge': None
}

# 모델 로딩
print("[INFO] 모델 로딩 중 (buffalo_l)...")
face_model = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
face_model.prepare(ctx_id=0)
print("[INFO] 모델 준비 완료!")

# 얼굴 추출 함수
def extract_faces(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return face_model.get(img_rgb)

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

# 유사도 계산 함수
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Threshold 설정
THRESHOLD = 0.4

# 평균 계산용 변수
total_bytes = 0
num_images = 0

# Flask 앱 설정
app = Flask(__name__)

@app.route('/inference', methods=['POST'])
def inference():
    global total_bytes, num_images
    recv_time = time.time()

    if 'image' not in request.files:
        return jsonify({'status': 'error', 'message': 'No image uploaded'}), 400

    file = request.files['image']
    file_bytes = file.read()
    byte_size = len(file_bytes)

    # 누적 통계 계산
    total_bytes += byte_size
    num_images += 1
    avg_bytes = total_bytes / num_images

    print(f"[INFO] 수신 이미지 크기: {byte_size} bytes | 평균: {avg_bytes:.2f} bytes")

    img_array = np.frombuffer(file_bytes, np.uint8)
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

        for name, ref_emb in registered_faces.items():
            if ref_emb is not None:
                score = cosine_similarity(emb, ref_emb)
                if score > best_score:
                    best_score = score
                    best_name = name

        if best_score < THRESHOLD:
            best_name = '-'

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
    app.run(host='0.0.0.0', port=5000)
