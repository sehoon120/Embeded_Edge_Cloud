# ec2_server.py
from flask import Flask, request, jsonify
import time

app = Flask(__name__)

@app.route('/inference', methods=['POST'])
def inference():
    # 수신 시각 기록 (선택 사항)
    recv_time = time.time()
    
    # 데이터 처리 (간단히 echo 처리)
    data = request.get_json()
    
    # 간단한 응답 반환
    return jsonify({
        'status': 'received',
        'message': data,
        'recv_time': recv_time  # EC2에서 받은 시각을 같이 보내줄 수도 있음
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
