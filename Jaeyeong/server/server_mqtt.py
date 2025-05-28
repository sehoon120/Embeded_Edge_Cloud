
import cv2
import base64
import numpy as np
import paho.mqtt.client as mqtt
import json
from insightface.app import FaceAnalysis

face_model = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
face_model.prepare(ctx_id=0)

registered_faces = {
    'sehoon': None,
    'jaeyoung': None
}
REGISTER_DIR = '/path/to/registered_images'
THRESHOLD = 0.4

def extract_faces(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return face_model.get(img_rgb)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def initialize_registered_faces():
    for name in registered_faces:
        img = cv2.imread(f"{REGISTER_DIR}/{name}.jpg")
        faces = extract_faces(img)
        if faces:
            registered_faces[name] = faces[0].embedding

initialize_registered_faces()

def on_message(client, userdata, msg):
    payload = json.loads(msg.payload.decode())
    client_id = payload['client_id']
    img_data = base64.b64decode(payload['image'])
    img_array = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    results = []
    for face in extract_faces(img):
        emb = face.embedding
        best_score, best_name = -1.0, '-'
        for name, ref in registered_faces.items():
            if ref is not None:
                score = cosine_similarity(emb, ref)
                if score > best_score:
                    best_score = score
                    best_name = name
        if best_score < THRESHOLD:
            best_name = '-'
        results.append({
            'identified': best_name,
            'score': round(float(best_score), 4),
            'bbox': list(map(int, face.bbox))
        })

    response_topic = f"face/infer/response/{client_id}"
    client.publish(response_topic, json.dumps({"results": results}))

client = mqtt.Client()
client.on_message = on_message
client.connect("localhost", 1883)
client.subscribe("face/infer/request")
client.loop_forever()
