
import os
import cv2
import requests
import random
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    average_precision_score, precision_recall_curve
)

# Spiking method base setting
thresh = 0.2
delta_t = 2**(-10)
tau1 = 5e-3
decay1 = np.exp(-delta_t / tau1)

mem = 0
spike = 0

def act_fun(thresh, mem):
    return 1 if mem > thresh else 0

def mem_update(x, mem, spike):
    mem = mem * decay1 * (1. - spike) + 0.1 * x
    spike = act_fun(thresh, mem)
    return mem, spike

# 서버 주소 및 경로 설정
EC2_URL = 'http://52.79.154.43:8000/inference'
lfw_dir = '/path/to/lfw'  # ← 실제 경로로 수정하세요

# 등록된 이름 리스트
registered_names = {
    'Ann_Veneman',
    'Edmund_Stoiber',
    'Gray_Davis',
    'Hugo_Chavez',
    'Jacques_Rogge'
}

# 평가용 변수 초기화
y_true = []
y_pred = []
y_scores = []
label_set = set()

per_class_scores = defaultdict(list)
per_class_truths = defaultdict(list)

correct = 0
total = 0

# 데이터 전송 및 결과 수집
for person in os.listdir(lfw_dir):
    person_dir = os.path.join(lfw_dir, person)
    if not os.path.isdir(person_dir):
        continue

    label_set.add(person)

    for img_file in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_file)
        x = 0.0
        img = cv2.imread(img_path)

        if img is not None:
            x = 1.0
            img = cv2.resize(img, (400, 400), interpolation=cv2.INTER_AREA)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
            _, img_encoded = cv2.imencode('.jpg', img, encode_param)
            files = {'image': ('image.jpg', img_encoded.tobytes(), 'image/jpeg')}
            mem, spike = mem_update(x, mem, spike)

            if spike:
                try:
                    response = requests.post(EC2_URL, files=files, timeout=5)
                    result = response.json()

                    if result.get('status') == 'success':
                        face_results = result.get('results', [])

                        if not face_results:
                            print(f"[{total+1}] GT: {person} | FAIL - No face")
                            y_true.append(person)
                            y_pred.append('-')
                            y_scores.append(0.0)
                            for label in label_set:
                                per_class_scores[label].append(0.0)
                                per_class_truths[label].append(1 if person == label else 0)
                            total += 1
                            continue

                        best_face = max(face_results, key=lambda x: x['score'])
                        identified = best_face.get('identified', '-')
                        score = best_face.get('score', 0.0)

                        print(f"[{total+1}] GT: {person} | Predicted: {identified} | Score: {score:.3f}")

                        y_true.append(person)
                        y_pred.append(identified)
                        y_scores.append(score)

                        for label in label_set:
                            per_class_scores[label].append(score if identified == label else 0.0)
                            per_class_truths[label].append(1 if person == label else 0)

                        is_correct = (
                            (person in registered_names and identified == person) or
                            (person not in registered_names and identified == '-')
                        )
                        if is_correct:
                            correct += 1

                    else:
                        print(f"[{total+1}] GT: {person} | FAIL - Server error")

                except Exception as e:
                    print(f"Error for {img_path}: {e}")

                total += 1

            elif x == 1.0 and person in registered_names:
                print(f"[{total+1}] GT: {person} | FAIL - Detected but spike==0")
                y_true.append(person)
                y_pred.append('-')
                y_scores.append(0.0)
                for label in label_set:
                    per_class_scores[label].append(0.0)
                    per_class_truths[label].append(1 if person == label else 0)
                total += 1

# 성능 지표 계산
accuracy = correct / total if total > 0 else 0.0
precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

print("\nClassification Metrics")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

# mAP 계산
ap_list = []
for label in sorted(label_set):
    y_true_bin = per_class_truths[label]
    y_score_cls = per_class_scores[label]

    if sum(y_true_bin) == 0:
        continue

    ap = average_precision_score(y_true_bin, y_score_cls)
    ap_list.append(ap)
    print(f"AP for class '{label}': {ap:.4f}")

mean_ap = sum(ap_list) / len(ap_list) if ap_list else 0.0
print(f"\nMean Average Precision (mAP): {mean_ap:.4f}")

# PR Curve 시각화
labels_to_plot = random.sample(list(label_set), min(5, len(label_set)))

plt.figure(figsize=(10, 7))
for label in labels_to_plot:
    y_true_bin = per_class_truths[label]
    y_score_cls = per_class_scores[label]

    if sum(y_true_bin) == 0:
        continue

    precision_vals, recall_vals, _ = precision_recall_curve(y_true_bin, y_score_cls)
    ap = average_precision_score(y_true_bin, y_score_cls)

    plt.plot(recall_vals, precision_vals, label=f"{label} (AP={ap:.2f})")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (Sample Classes)")
plt.legend(loc='lower left')
plt.grid(True)
plt.tight_layout()
plt.show()
