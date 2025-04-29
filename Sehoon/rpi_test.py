# rpi_client.py
import requests
import time

EC2_URL = 'http://52.79.154.43:8000/inference'

rtt_list = []
count = 0

while True:
    with open('768_432.jpg', 'rb') as f:
        files = {'image': ('768_432.jpg', f, 'image/jpeg')}

        start = time.time()
        try:
            response = requests.post(EC2_URL, files=files, timeout=5)
            end = time.time()

            rtt = (end - start) * 1000  # ms 단위
            rtt_list.append(rtt)
            count += 1

            avg_rtt = sum(rtt_list) / count

            print("Response:", response.json())
            print(f"RTT: {rtt:.2f} ms | 평균 RTT: {avg_rtt:.2f} ms")

        except requests.exceptions.RequestException as e:
            print("전송 실패:", e)

    time.sleep(0.5)
