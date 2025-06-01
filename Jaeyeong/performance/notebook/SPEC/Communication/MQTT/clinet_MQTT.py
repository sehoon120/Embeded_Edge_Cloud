import cv2
import base64
import json
import os
import paho.mqtt.client as mqtt
import time

# MQTT 설정
MQTT_BROKER = "Server IP Address"  
MQTT_PORT = 1883
MQTT_TOPIC_SUB = "face/result"
MQTT_TOPIC_PUB = "face/infer"

# 이미지 폴더 설정
IMAGE_DIR = "./images"  # ← 여기에 여러 이미지가 있다고 가정 (jpg/png 등)

# 응답 저장용
received_results = {}

# MQTT 수신 콜백
def on_message(client, userdata, msg):
    try:
        response = json.loads(msg.payload.decode())
        recv_time = response.get('recv_time')
        results = response.get('results', [])
        print(f"\n[📥 SERVER RESPONSE] recv_time: {recv_time}")
        for r in results:
            print(f" - 이름: {r['identified']}, 유사도: {r['score']}, bbox: {r['bbox']}")
        received_results[recv_time] = results
    except Exception as e:
        print(f"[ERROR] 응답 처리 중 예외 발생: {e}")

# MQTT 클라이언트 설정
client = mqtt.Client()
client.on_message = on_message
client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.subscribe(MQTT_TOPIC_SUB)
client.loop_start()

# 이미지 파일 전송
image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.png'))])

for filename in image_files:
    image_path = os.path.join(IMAGE_DIR, filename)
    img = cv2.imread(image_path)
    if img is None:
        print(f"[WARNING] 이미지 로드 실패: {image_path}")
        continue

    # base64 인코딩
    _, buffer = cv2.imencode(".jpg", img)
    img_b64 = base64.b64encode(buffer).decode()

    payload = {
        "image": img_b64
    }

    print(f"[🚀 PUBLISH] {filename} 전송 중...")
    client.publish(MQTT_TOPIC_PUB, json.dumps(payload))

    # 결과 수신을 위한 시간 확보
    time.sleep(2)  # 서버 응답 시간에 따라 조절

# MQTT 종료
time.sleep(2)  # 마지막 응답까지 대기
client.loop_stop()
client.disconnect()
