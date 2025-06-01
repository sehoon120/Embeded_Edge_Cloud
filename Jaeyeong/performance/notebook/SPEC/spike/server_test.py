from flask import Flask, request, jsonify
import cv2
import numpy as np
import insightface
import os
import time
from insightface.app import FaceAnalysis

# 등록 이미지 경로
REGISTER_DIR = r"/home/han/lfw-deepfunneled/lfw-deepfunneled/"

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

def extract_all_lfw_embeddings(lfw_root_dir):
    lfw_embeddings = {}

    for person_name in os.listdir(lfw_root_dir):
        person_dir = os.path.join(lfw_root_dir, person_name)
        if not os.path.isdir(person_dir):
            continue

        embeddings = []
        for img_file in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue

            faces = extract_faces(img)
            if faces:
                embeddings.append(faces[0].embedding)

        if embeddings:
            lfw_embeddings[person_name] = np.array(embeddings)
            print(f"[INFO] {person_name} - {len(embeddings)} 임베딩 추출됨")

    return lfw_embeddings

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

        for name, ref_emb in registered_faces.items():
            if ref_emb is not None:
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
    # LFW 디렉토리 경로 예시: 압축 푼 폴더 경로를 넣으세요
    LFW_PATH = r"/home/han/lfw-deepfunneled/lfw-deepfunneled/"
    
    print("[INFO] 전체 LFW 인물 임베딩 추출 시작...")
    all_embeddings = extract_all_lfw_embeddings(LFW_PATH)
    
    # numpy 파일로 저장
    np.save("lfw_embeddings.npy", all_embeddings)
    print("[INFO] 임베딩 저장 완료 (lfw_embeddings.npy)")
    
    # Flask 실행
    app.run(host='0.0.0.0', port=5000)
