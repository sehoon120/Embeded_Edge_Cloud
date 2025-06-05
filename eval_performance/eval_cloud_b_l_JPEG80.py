import os
import cv2
import numpy as np
import onnxruntime as ort
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from insightface.app import FaceAnalysis

REGISTER_DIR = r"C:\Embeded_Project\registered_faces"
# ONNX 모델 경로
ONNX_PATH = r'C:\Embeded_Project\Embeded_Edge_Cloud\Sehoon\models\buffalo_s_recognition_fp16.onnx'

# 모델 로드
model = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])  # or CPUExecutionProvider
model.prepare(ctx_id=0)  # GPU: 0, CPU: -1

# === cosine similarity ===
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# === LFW pairs 파일 읽기 ===
def load_pairs(pairs_path, lfw_root):
    pairs = []
    labels = []
    with open(pairs_path, 'r') as f:
        lines = f.readlines()[1:]
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 3:
                name, idx1, idx2 = parts
                img1 = os.path.join(lfw_root, name, f"{name}_{int(idx1):04d}.jpg")
                img2 = os.path.join(lfw_root, name, f"{name}_{int(idx2):04d}.jpg")
                label = 1
            else:
                name1, idx1, name2, idx2 = parts
                img1 = os.path.join(lfw_root, name1, f"{name1}_{int(idx1):04d}.jpg")
                img2 = os.path.join(lfw_root, name2, f"{name2}_{int(idx2):04d}.jpg")
                label = 0

            if os.path.exists(img1) and os.path.exists(img2):
                pairs.append((img1, img2))
                labels.append(label)
    return pairs, labels

# === InsightFace 임베딩 추출 ===
def extract_embedding(img):
    faces = model.get(img)
    if not faces:
        return None
    return faces[0].embedding / np.linalg.norm(faces[0].embedding)

# === 평가 ===
def evaluate_insightface(pairs_path, lfw_root):
    pairs, labels = load_pairs(pairs_path, lfw_root)
    sims = []
    valid_labels = []

    for i, (img1_path, img2_path) in enumerate(pairs):
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        # ▶ 추가: JPEG 품질 80%로 메모리 상 압축/해제
        _, buf1 = cv2.imencode(".jpg", img1, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        img1 = cv2.imdecode(buf1, cv2.IMREAD_COLOR)

        _, buf2 = cv2.imencode(".jpg", img2, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        img2 = cv2.imdecode(buf2, cv2.IMREAD_COLOR)

        emb1 = extract_embedding(img1)
        emb2 = extract_embedding(img2)

        if emb1 is None or emb2 is None:
            continue

        sim = cosine_sim(emb1, emb2)
        sims.append(sim)
        valid_labels.append(labels[i])

        if i % 500 == 0:
            print(f"[INFO] {i}/{len(pairs)}")

    sims = np.array(sims)
    labels = np.array(valid_labels)

    # ROC / AUC
    fpr, tpr, thresholds = roc_curve(labels, sims)
    roc_auc = auc(fpr, tpr)

    # Best Accuracy & Threshold
    best_acc, best_thresh = 0, 0
    for t in thresholds:
        preds = (sims > t).astype(int)
        acc = np.mean(preds == labels)
        if acc > best_acc:
            best_acc = acc
            best_thresh = t

    # TPR@FAR=1e-3
    target_far = 1e-3
    idx = np.where(fpr <= target_far)[0]
    tpr_at_far = tpr[idx[-1]] if len(idx) > 0 else 0.0

    # 결과 출력
    print(f"\n[RESULT] AUC: {roc_auc:.4f}")
    print(f"[RESULT] Best Accuracy: {best_acc:.4f} (Threshold: {best_thresh:.4f})")
    print(f"[RESULT] TPR@FAR=1e-3: {tpr_at_far:.4f}")

    # ROC Curve
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - buffalo_l (InsightFace)")
    plt.grid(True)
    plt.legend()
    plt.show()

pairs_path = r"C:\Embeded_Project\lfw\pairs.txt"
lfw_root = r"C:\Embeded_Project\lfw\lfw-funneled.tgz\lfw-funneled\lfw_funneled"

evaluate_insightface(pairs_path, lfw_root)
