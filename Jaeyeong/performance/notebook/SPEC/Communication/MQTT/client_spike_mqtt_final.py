
import os
import cv2
import time
import base64
import json
import numpy as np
import mediapipe as mp
import paho.mqtt.client as mqtt

# Spiking method
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

# Mediapipe face detector
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6)

# MQTT 설정
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
REQUEST_TOPIC = "face/infer/request"
RESPONSE_TOPIC = "face/infer/response"

# LFW 데이터 경로
lfw_dir = "/path/to/lfw"
registered_names = {'Ann_Veneman', 'Edmund_Stoiber', 'Gray_Davis', 'Hugo_Chavez', 'Jacques_Rogge'}

# MQTT 응답 콜백
results = {}
def on_message(client, userdata, msg):
    global results
    payload = json.loads(msg.payload.decode())
    results = payload

client = mqtt.Client()
client.on_message = on_message
client.connect(MQTT_BROKER, MQTT_PORT)
client.subscribe(RESPONSE_TOPIC)
client.loop_start()

for person in os.listdir(lfw_dir):
    person_dir = os.path.join(lfw_dir, person)
    if not os.path.isdir(person_dir):
        continue

    for img_file in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_file)
        img = cv2.imread(img_path)
        x = 0.0

        if img is not None:
            resized = cv2.resize(img, (400, 400))
            result = face_detection.process(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
            x = 1.0 if result.detections else 0.0
            mem, spike = mem_update(x, mem, spike)

            if spike:
                _, img_encoded = cv2.imencode('.jpg', resized)
                b64_image = base64.b64encode(img_encoded).decode('utf-8')
                payload = json.dumps({'image': b64_image, 'filename': img_file})
                client.publish(REQUEST_TOPIC, payload)

                time.sleep(1)
                print(f"[{img_file}] 결과: {results}")

client.loop_stop()
client.disconnect()
