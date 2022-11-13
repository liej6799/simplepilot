import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import transform_frames 
import onnx
import onnxruntime as rt

# single frame test

#read frame
img = cv2.imread('sample/road.png') 

#initiate frame resize to 1164, 874
dim = (1164, 874)

# preprocess image
seq_len = 1

yuv_frames = np.zeros((seq_len + 1, dim[1]*3//2, dim[0]), dtype=np.uint8)
stacked_frames = np.zeros((seq_len, 12, 128, 256), dtype=np.uint8)
recurrent_state = np.zeros((1, 512), dtype=np.float32)

# resize image
# orig image from comma10k already 1164, 874, but other source might need to resize
img = cv2.resize(img, dim)
assert img.shape == ((874, 1164, 3))

# convert bgr to yuv
img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV_I420)
assert img.shape  == ((874*3//2, 1164))

yuv_frames[0] = img
transformmed_frames = transform_frames(yuv_frames)

for i in range(seq_len):
    stacked_frames[i] = np.vstack(transformmed_frames[i:i+2])[None].reshape(12, 128, 256)


# end preprocess image

# model
path_to_model = '/Users/joes/source/openpilot-pipeline/common/models/supercombo.onnx'
onnx_graph = onnx.load(path_to_model)
output_names = [node.name for node in onnx_graph.graph.output]
model = rt.InferenceSession(path_to_model, providers=['CPUExecutionProvider'])

for t_idx in range(seq_len):
    inputs = {
        'input_imgs': stacked_frames[t_idx:t_idx+1].astype(np.float32),
        'desire': np.zeros((1, 8), dtype=np.float32),
        'traffic_convention': np.array([0, 1], dtype=np.float32).reshape(1, 2),
        'initial_state': recurrent_state,
    }

    outs =  model.run(output_names, inputs)[0]
    recurrent_state = outs[:, -512:]
    print(outs.shape)

#end model




#output frame
cv2.imshow('image2',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
