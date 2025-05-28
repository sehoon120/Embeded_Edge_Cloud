
import cv2
import time
import paho.mqtt.client as mqtt
import base64
import uuid
import json
from picamera2 import Picamera2
import mediapipe as mp
import numpy as np

CLIENT_ID = str(uuid.uuid4())[:8]
BROKER = 'localhost'
PORT = 1883
REQ_TOPIC = 'face/infer/request'
RESP_TOPIC = f'face/infer/response/{CLIENT_ID}'

thresh = 0.2
delta_t = 2**(-10)
tau1 = 5e-3
decay1 = np.exp(-delta_t/tau1)
mem, spike = 0, 0

def act_fun(thresh, mem):
    return 1 if mem > thresh else 0

def mem_update(x, mem, spike):
    mem = mem * decay1 * (1. - spike) + 0.1 * x
    spike = act_fun(thresh, mem)
    return mem, spike

def on_message(client, userdata, msg):
    data = json.loads(msg.payload.decode())
    results = data.get("results", [])

    frame = userdata['last_frame']
    for person in results:
        name = person.get("identified", "unknown")
        score = person.get("score", 0.0)
        x1, y1, x2, y2 = map(int, person.get("bbox", [0, 0, 0, 0]))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{name} ({score:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("MQTT Face Recognition", frame)
    cv2.waitKey(1)

mp_face_detection = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6)
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
picam2.start()

client = mqtt.Client(client_id=CLIENT_ID, userdata={'last_frame': None})
client.on_message = on_message
client.connect(BROKER, PORT)
client.subscribe(RESP_TOPIC)
client.loop_start()

while True:
    frame = picam2.capture_array()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_face_detection.process(frame_rgb)

    x = 1.0 if results.detections else 0.0
    mem, spike = mem_update(x, mem, spike)

    if spike:
        _, buffer = cv2.imencode('.jpg', frame)
        img_base64 = base64.b64encode(buffer).decode()

        message = {
            "client_id": CLIENT_ID,
            "image": img_base64
        }
        client.user_data_set({'last_frame': frame.copy()})
        client.publish(REQ_TOPIC, json.dumps(message))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

picam2.stop()
cv2.destroyAllWindows()
client.loop_stop()
client.disconnect()
