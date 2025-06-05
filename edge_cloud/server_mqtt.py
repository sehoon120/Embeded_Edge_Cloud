import base64
import cv2
import numpy as np
import os
import time
import json
import ssl
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

# AWS IoT MQTT 설정
MQTT_BROKER = "a33h6h6h7kxrjl-ats.iot.ap-northeast-2.amazonaws.com"
MQTT_PORT = 8883
REQUEST_TOPIC = "face/infer/request"

# 인증서 경로 (Linux용으로 / 슬래시 사용)
CA_PATH = "/home/ubuntu/test/AmazonRootCA1.pem"
CERT_PATH = "/home/ubuntu/test/fad08c99cac46ad365f3e3e657902745e89e2436f541c65904c5ac0b783e7e44-certificate.pem.crt"  # 실제 이름 입력
KEY_PATH = "/home/ubuntu/test/fad08c99cac46ad365f3e3e657902745e89e2436f541c65904c5ac0b783e7e44-private.pem.key"      # 실제 이름 입력

# MQTT 수신 콜백
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

# MQTT 클라이언트 생성 및 연결
client = mqtt.Client()
client.tls_set(
    ca_certs=CA_PATH,
    certfile=CERT_PATH,
    keyfile=KEY_PATH,
    tls_version=ssl.PROTOCOL_TLSv1_2
)
client.on_message = on_message
client.connect(MQTT_BROKER, MQTT_PORT)

# 구독 및 시작
print(f"[INFO] Connected to broker at {MQTT_BROKER}:{MQTT_PORT}")
client.subscribe(REQUEST_TOPIC)
print(f"[INFO] Subscribed to topic: {REQUEST_TOPIC}")
client.loop_forever()
