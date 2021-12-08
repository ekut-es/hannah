import onnx
import os
file_path = '/../trained_models/svd_rank3/test/conv_net_trax/model.onnx'
model = onnx.load(os.path.dirname(__file__) + file_path)
[print(t.name) for t in model.graph.initializer]