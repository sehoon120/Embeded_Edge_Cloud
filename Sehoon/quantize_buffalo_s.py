from onnxconverter_common.float16 import convert_float_to_float16
import onnx

model_fp32 = onnx.load(r'C:\Embeded_Project\Embeded_Edge_Cloud\Sehoon\models\buffalo_s_recognition.onnx')
model_fp16 = convert_float_to_float16(model_fp32)
onnx.save(model_fp16, r'C:\Embeded_Project\Embeded_Edge_Cloud\Sehoon\models\buffalo_s_recognition_fp16.onnx')





