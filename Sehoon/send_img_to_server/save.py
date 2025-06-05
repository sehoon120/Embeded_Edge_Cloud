from flask import Flask, request, jsonify
import os
from datetime import datetime

app = Flask(__name__)  # ✅ Flask 앱 정의

SAVE_DIR = 'registered_faces'
os.makedirs(SAVE_DIR, exist_ok=True)

@app.route('/inference', methods=['POST'])
def inference():
    if 'image' not in request.files:
        return jsonify({'status': 'error', 'message': 'No image part'}), 400

    file = request.files['image']
    filename = 'sehoon' + '.jpg'
    # 'jaeyoung'
    save_path = os.path.join(SAVE_DIR, filename)
    file.save(save_path)

    return jsonify({'status': 'success', 'saved_path': save_path})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
