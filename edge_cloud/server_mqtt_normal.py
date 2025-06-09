import base64
import cv2
import numpy as np
import os
import time
import json
import paho.mqtt.client as mqtt
from insightface.app import FaceAnalysis

# 등록된 얼굴 이미지 디렉토리
REGISTER_DIR = "registered_faces"
registered_faces = {
    'jaeyoung': None,
    'sehoon': None
}

# 모델 준비
print("[INFO] 모델 로딩 중 (buffalo_l)...")
face_model = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
face_model.prepare(ctx_id=0)
print("[INFO] 모델 준비 완료!")

# 얼굴 임베딩 추출 함수
def extract_faces(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return face_model.get(img_rgb)

# 등록된 얼굴 초기화
def initialize_registered_faces():
    for name in registered_faces:
        img_path = os.path.join(REGISTER_DIR, f'{name}.jpg')
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            faces = extract_faces(img)
            if len(faces) > 0:
                registered_faces[name] = faces[0].embedding
                print(f"[INFO] {name} 얼굴 등록 완료")
            else:
                print(f"[WARN] {name} 얼굴 인식 실패")
        else:
            print(f"[WARN] {img_path} 존재하지 않음")

initialize_registered_faces()

# 코사인 유사도
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 임계값
THRESHOLD = 0.4

# MQTT 설정
MQTT_BROKER = "ipadr"  # 실제 브로커 IP로 수정하세요
MQTT_PORT = 1883
REQUEST_TOPIC = "face/infer/request"

# 메시지 수신 시 동작
def on_message(client, userdata, msg):
    print("[INFO] MQTT message received")

    try:
        payload = json.loads(msg.payload.decode())
        print("[DEBUG] payload keys:", payload.keys())

        image_data = base64.b64decode(payload['image'])
        img_array = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        faces = extract_faces(img)
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

        # 응답 토픽 지정 (client_id 기반)
        response_topic = f"face/infer/response/{payload['client_id']}"
        client.publish(response_topic, json.dumps(results))
        print(f"[INFO] Published results to {response_topic}")

    except Exception as e:
        print(f"[ERROR] Exception in on_message: {e}")

# MQTT 연결 및 시작
client = mqtt.Client(protocol=mqtt.MQTTv311)
client.on_message = on_message
client.connect(MQTT_BROKER, MQTT_PORT)
print(f"[INFO] Connected to broker at {MQTT_BROKER}:{MQTT_PORT}")
client.subscribe(REQUEST_TOPIC)
print(f"[INFO] Subscribed to topic: {REQUEST_TOPIC}")
client.loop_forever()
