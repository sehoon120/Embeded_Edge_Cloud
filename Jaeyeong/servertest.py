# simple_server.py
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return "✅ 서버가 잘 열렸습니다!"

@app.route('/infer', methods=['POST'])
def infer():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']
    filename = image_file.filename

    # 예시: 파일 이름만 리턴
    return jsonify({'result': f'{filename} received!'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
