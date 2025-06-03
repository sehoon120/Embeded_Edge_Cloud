import os
import cv2
import numpy as np
from sklearn.metrics import roc_curve, auc
import onnxruntime as ort

# 모델 불러오기
ONNX_PATH = r'C:\Embeded_Project\Embeded_Edge_Cloud\Sehoon\models\buffalo_s_recognition_fp16.onnx'
onnx_session = ort.InferenceSession(ONNX_PATH, providers=['CUDAExecutionProvider'])     # CPUExecutionProvider
input_name = onnx_session.get_inputs()[0].name

# 얼굴 전처리 함수
def preprocess(img):
    img = cv2.resize(img, (112, 112))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    img = np.transpose(img, (2, 0, 1))[np.newaxis, ...]
    return img.astype(np.float16)

# 임베딩 추출
def get_embedding(img):
    pre = preprocess(img)
    emb = onnx_session.run(None, {input_name: pre})[0][0]
    return emb / np.linalg.norm(emb)

# cosine similarity
def cos_sim(a, b):
    return np.dot(a, b)

# LFW pairs.txt 로드
def load_pairs(pairs_path, lfw_root):
    pairs = []
    labels = []

    with open(pairs_path, 'r') as f:
        lines = f.readlines()[1:]  # skip header
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 3:
                # 같은 사람
                name, idx1, idx2 = parts
                img1 = os.path.join(lfw_root, name, f"{name}_{int(idx1):04d}.jpg")
                img2 = os.path.join(lfw_root, name, f"{name}_{int(idx2):04d}.jpg")
                label = 1
            else:
                # 다른 사람
                name1, idx1, name2, idx2 = parts
                img1 = os.path.join(lfw_root, name1, f"{name1}_{int(idx1):04d}.jpg")
                img2 = os.path.join(lfw_root, name2, f"{name2}_{int(idx2):04d}.jpg")
                label = 0

            if os.path.exists(img1) and os.path.exists(img2):
                pairs.append((img1, img2))
                labels.append(label)

    return pairs, labels

# 전체 평가 수행
def evaluate_lfw(pairs_path, lfw_root):
    pairs, labels = load_pairs(pairs_path, lfw_root)
    similarities = []

    for idx, (img1_path, img2_path) in enumerate(pairs):
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        emb1 = get_embedding(img1)
        emb2 = get_embedding(img2)

        sim = cos_sim(emb1, emb2)
        similarities.append(sim)

        if idx % 100 == 0:
            print(f"[INFO] 처리 중: {idx}/{len(pairs)}")

    # ROC Curve 및 AUC
    fpr, tpr, thresholds = roc_curve(labels, similarities)
    roc_auc = auc(fpr, tpr)

    # Accuracy 측정
    best_acc = 0
    for t in thresholds:
        preds = [1 if s > t else 0 for s in similarities]
        acc = np.mean(np.array(preds) == np.array(labels))
        best_acc = max(best_acc, acc)

    print(f"[RESULT] AUC: {roc_auc:.4f}")
    print(f"[RESULT] Best Accuracy: {best_acc:.4f}")

    return fpr, tpr, roc_auc



# 예시 경로
pairs_txt_path = "/home/pi/lfw/pairs.txt"
lfw_image_root = "/home/pi/lfw/lfw-deepfunneled"

fpr, tpr, auc_score = evaluate_lfw(pairs_txt_path, lfw_image_root)
