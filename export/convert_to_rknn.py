# convert_to_rknn.py
from rknn.api import RKNN

onnx_path = 'weights/yolo11n-pose.onnx'
rknn_path = 'weights/yolo11n-pose.rknn'


TARGET = 'rk3588'
QUANT = False

rknn = RKNN()

# 可选预处理配置；也可保持默认在推理端自行处理
rknn.config(
    target_platform=TARGET,
    mean_values=[[0, 0, 0]],
    std_values=[[255, 255, 255]],
    quantized_algorithm='normal'
)

print('Loading ONNX...')
ret = rknn.load_onnx(model=onnx_path)
if ret != 0:
    raise RuntimeError('load_onnx failed')

print('Building RKNN...')
ret = rknn.build(do_quantization=QUANT, dataset=None)
if ret != 0:
    raise RuntimeError('build failed')

print('Exporting RKNN...')
ret = rknn.export_rknn(rknn_path)
if ret != 0:
    raise RuntimeError('export failed')

rknn.release()
print('Done ->', rknn_path)
