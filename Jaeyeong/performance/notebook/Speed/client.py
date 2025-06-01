import requests
import time
from picamera2 import Picamera2
import cv2
import numpy as np
import mediapipe as mp

EC2_URL = 'http://52.79.154.43:8000/inference'

###########################################################################################################################################

# spiking method base setting
thresh = 0.2
delta_t = 2**(-10)
tau1 = 5e-3
decay1 = np.exp(-delta_t/tau1)

mem = 0
spike = 0

def act_fun(thresh, mem):
    return 1 if mem > thresh else 0

def mem_update(x, mem, spike):
    mem = mem * decay1 * (1. - spike) + 0.1 * x
    spike = act_fun(thresh, mem)
    return mem, spike

###########################################################################################################################################

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6)

rtt_list = []
count = 0
inference_count = 0  # ★ EC2 요청 횟수 카운터

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
picam2.start()

frame_count = 0
fps_start_time = time.time()

# ★ 전체 실행 시간 설정
duration_sec = 30
start_time = time.time()

while True:
    # ★ 시간 초과되면 종료
    if time.time() - start_time > duration_sec:
        print(f"\n[INFO] Total EC2 inference count: {inference_count}")
        break

    frame = picam2.capture_array()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(frame_rgb)
    x = 1.0 if results.detections else 0.0
    mem, spike = mem_update(x, mem, spike)
    
    if spike:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80] 
        _, img_encoded = cv2.imencode('.jpg', frame, encode_param)
        files = {'image': ('full.jpg', img_encoded.tobytes(), 'image/jpeg')}
        
        try:
            print("Try Connection!")
            start_req = time.time()
            response = requests.post(EC2_URL, files=files, timeout=5)
            
            if response.status_code == 200:
                end_req = time.time()
                rtt = end_req - start_req
                rtt_list.append(rtt)
                count += 1
                inference_count += 1  # ★ 카운트 증가
                print(f"[{count}] RTT: {rtt:.4f} sec | avg RTT: {sum(rtt_list)/count:.4f} sec")

                frame_count += 1
                if frame_count >= 10:
                    elapsed = time.time() - fps_start_time
                    fps = frame_count / elapsed
                    print(f"[FPS] average FPS: {fps:.2f}")
                    frame_count = 0
                    fps_start_time = time.time()
                    
                try:
                    server_result = response.json()
                    results = server_result.get('results', [])

                    for person in results:
                        name = person.get('identified', 'unknown')
                        score = person.get('score', 0.0)
                        bbox = person.get('bbox', [0, 0, 0, 0])

                        x1, y1, x2, y2 = bbox
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{name} ({score:.2f})", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    print(f"[{count}] Server Result: {results}")

                except Exception as e:
                    print(f"Error parsing server response: {e}")
                    
                time.sleep(0.5)
            
            else:
                print(f"Request failed with status code: {response.status_code}")
                
        except Exception as e:
            print(f"Request exception: {e}")
    
    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print(f"\n[INFO] Interrupted manually. Total EC2 inference count: {inference_count}")
        break

    time.sleep(0.1)

picam2.stop()
face_detection.close()
cv2.destroyAllWindows()
