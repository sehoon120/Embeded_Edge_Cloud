from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    model_input=r'C:\Embeded_Project\Embeded_Edge_Cloud\Sehoon\models\buffalo_s_recognition.onnx',
    model_output=r'C:\Embeded_Project\Embeded_Edge_Cloud\Sehoon\models\buffalo_s_recognition_int8.onnx',
    weight_type=QuantType.QInt8
)
print("[INFO] INT8 양자화 완료!")
