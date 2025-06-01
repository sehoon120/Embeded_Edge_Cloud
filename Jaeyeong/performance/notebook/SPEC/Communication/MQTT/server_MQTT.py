import cv2
import numpy as np
import insightface
import os
import time
import json
import base64
import paho.mqtt.client as mqtt
from insightface.app import FaceAnalysis

# 전역 변수
lfw_embeddings = {}

# MQTT 설정
MQTT_BROKER = "0.0.0.0"   # 로컬 또는 외부 브로커 IP
MQTT_PORT = 1883
MQTT_TOPIC_SUB = "face/infer"
MQTT_TOPIC_PUB = "face/result"

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

# MQTT 메시지 수신 콜백 함수
def on_message(client, userdata, msg):
    recv_time = time.time()
    try:
        payload = json.loads(msg.payload.decode())
        img_b64 = payload.get("image")

        if not img_b64:
            print("[ERROR] No image data in MQTT payload")
            return

        img_bytes = base64.b64decode(img_b64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            print("[ERROR] 이미지 디코딩 실패")
            return

        faces = extract_faces(img)
        if not faces:
            client.publish(MQTT_TOPIC_PUB, json.dumps({
                'status': 'success',
                'results': [],
                'recv_time': recv_time
            }))
            return

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

            bbox = face.bbox.astype(int).tolist()

            results.append({
                'identified': best_name,
                'score': round(float(best_score), 4),
                'bbox': bbox
            })

        print(f"[RESULT] 총 {len(results)}명 인식됨")
        for r in results:
            print(f" - 이름: {r['identified']} | 유사도: {r['score']} | bbox: {r['bbox']}")

        client.publish(MQTT_TOPIC_PUB, json.dumps({
            'status': 'success',
            'results': results,
            'recv_time': recv_time
        }))

    except Exception as e:
        print(f"[ERROR] 메시지 처리 중 예외 발생: {e}")

# MQTT 설정 및 연결
client = mqtt.Client()
client.on_message = on_message

if __name__ == '__main__':
    print("[INFO] 전체 LFW 인물 임베딩 로드 시작...")
    lfw_embeddings = np.load("lfw_embeddings.npy", allow_pickle=True).item()
    print(f"[INFO] 총 {len(lfw_embeddings)}명 임베딩 로드됨") 

    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.subscribe(MQTT_TOPIC_SUB)
    print(f"[INFO] MQTT 브로커에 연결됨, 토픽 '{MQTT_TOPIC_SUB}' 구독 중...")

    client.loop_forever()

