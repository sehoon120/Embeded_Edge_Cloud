
import base64
import cv2
import numpy as np
import os
import time
import json
import paho.mqtt.client as mqtt
from insightface.app import FaceAnalysis

REGISTER_DIR = "/home/han/ESD/Embeded_Edge_Cloud/Jaeyeong/performance/notebook/dataset/"
registered_faces = {
    'Ann_Veneman': None,
    'Edmund_Stoiber': None,
    'Gray_Davis': None,
    'Hugo_Chavez': None,
    'Jacques_Rogge': None
}

print("[INFO] 모델 로딩 중 (buffalo_l)...")
face_model = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
face_model.prepare(ctx_id=0)
print("[INFO] 모델 준비 완료!")

def extract_faces(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return face_model.get(img_rgb)

def initialize_registered_faces():
    for name in registered_faces:
        img_path = os.path.join(REGISTER_DIR, f'{name}.jpg')
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            faces = extract_faces(img)
            if len(faces) > 0:
                registered_faces[name] = faces[0].embedding

initialize_registered_faces()

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

THRESHOLD = 0.4

# MQTT 설정
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
REQUEST_TOPIC = "face/infer/request"
RESPONSE_TOPIC = "face/infer/response"

def on_message(client, userdata, msg):
    payload = json.loads(msg.payload.decode())
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

    client.publish(RESPONSE_TOPIC, json.dumps(results))

client = mqtt.Client()
client.on_message = on_message
client.connect(MQTT_BROKER, MQTT_PORT)
client.subscribe(REQUEST_TOPIC)
client.loop_forever()
