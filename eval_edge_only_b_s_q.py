import os
import cv2
import numpy as np
import onnxruntime as ort
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

REGISTER_DIR = r"C:\Embeded_Project\registered_faces"
# ONNX 모델 경로
ONNX_PATH = r'C:\Embeded_Project\Embeded_Edge_Cloud\Sehoon\models\buffalo_s_recognition_fp16.onnx'

# 모델 로드
onnx_session = ort.InferenceSession(ONNX_PATH, providers=['CUDAExecutionProvider'])
input_name = onnx_session.get_inputs()[0].name

def preprocess(img):
    img = cv2.resize(img, (112, 112))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    img = np.transpose(img, (2, 0, 1))[np.newaxis, ...]
    return img.astype(np.float16)

def get_embedding(img):
    emb = onnx_session.run(None, {input_name: preprocess(img)})[0][0]
    return emb / np.linalg.norm(emb)

def cosine_sim(a, b):
    return np.dot(a, b)

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

def evaluate_lfw(pairs_path, lfw_root):
    pairs, labels = load_pairs(pairs_path, lfw_root)
    similarities = []

    for i, (img1_path, img2_path) in enumerate(pairs):
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        emb1 = get_embedding(img1)
        emb2 = get_embedding(img2)
        sim = cosine_sim(emb1, emb2)
        similarities.append(sim)
        if i % 500 == 0:
            print(f"[INFO] Processing {i}/{len(pairs)}")

    labels = np.array(labels)
    similarities = np.array(similarities)

    # ROC & AUC
    fpr, tpr, thresholds = roc_curve(labels, similarities)
    roc_auc = auc(fpr, tpr)

    # Best Accuracy + Threshold
    best_acc = 0
    best_thresh = 0
    for t in thresholds:
        preds = (similarities > t).astype(int)
        acc = np.mean(preds == labels)
        if acc > best_acc:
            best_acc = acc
            best_thresh = t

    # TPR @ FAR = 1e-3
    target_far = 1e-3
    idx = np.where(fpr <= target_far)[0]
    tpr_at_far = tpr[idx[-1]] if len(idx) > 0 else 0.0

    print(f"\n[RESULT] AUC: {roc_auc:.4f}")
    print(f"[RESULT] Best Accuracy: {best_acc:.4f} (Threshold: {best_thresh:.4f})")
    print(f"[RESULT] TPR @ FAR=1e-3: {tpr_at_far:.4f}")

    # Plot ROC
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Face Verification")
    plt.grid(True)
    plt.legend(loc="lower right")
    plt.show()

    return roc_auc, best_acc, best_thresh, tpr_at_far

pairs_path = r"C:\Embeded_Project\lfw\pairs.txt"
lfw_root = r"C:\Embeded_Project\lfw\lfw-funneled.tgz\lfw-funneled\lfw_funneled"

evaluate_lfw(pairs_path, lfw_root)
