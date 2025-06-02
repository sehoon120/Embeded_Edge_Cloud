import insightface
import onnx

# 모델 준비
app = insightface.app.FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0)

# ArcFace 인식 모델 가져오기
arcface_model = app.models['arcface']

# ONNX 모델로 저장
arcface_model.save('buffalo_s_arcface.onnx')
print("[INFO] ONNX 파일 저장 완료!")


