import importlib
import numpy as np
import onnx
importlib
import onnxruntime as rt
import cv2
import torch
from utils import bgr_to_yuv, transform_frames, printf, FULL_FRAME_SIZE, create_image_canvas, PATH_TO_CACHE  # noqa

plot_img_height, plot_img_width = 480, 640 # can be reduced for wandb
seq_len = 1
#load fake image
def get_random_input_tensors():
  np_inputs = {
    'input_img': np.zeros((1, 1382400), dtype=np.float32),
    'calib': np.zeros((1, 3), dtype=np.float32),
  }
  return np_inputs


    #return imgs_med_model
def get_real_input():
    img = cv2.imread("sample/135.png")
    img = cv2.resize(img, (1164, 874), interpolation = cv2.INTER_AREA)
 
    rgb_frames = np.zeros((seq_len, plot_img_height, plot_img_width, 3), dtype=np.uint8)
    yuv_frames = np.zeros((seq_len + 1, FULL_FRAME_SIZE[1]*3//2, FULL_FRAME_SIZE[0]), dtype=np.uint8)
    stacked_frames = np.zeros((seq_len, 12, 128, 256), dtype=np.uint8)

    yuv_frames = np.zeros((1, FULL_FRAME_SIZE[1]*3//2, FULL_FRAME_SIZE[0]), dtype=np.uint8)
   
    yuv_frame2 = bgr_to_yuv(img)
    yuv_frames[0] = yuv_frame2
    prepared_frames = transform_frames(yuv_frames)

    return torch.from_numpy(prepared_frames).float()
    #cv2.imshow('awd', img)
    #cv2.waitKey(0) 




def load_inference_model(path_to_model):

    onnx_graph = onnx.load(path_to_model)
    output_names = [node.name for node in onnx_graph.graph.output]
    model = rt.InferenceSession(path_to_model, providers=['CPUExecutionProvider'])

    def run_model(inputs):
        outs =  model.run(output_names, inputs)[0]
        return outs

    return model, run_model


# run on the model
onnx_model, run_model = load_inference_model("models/supercombo.onnx")

#inference on the model
# inputs = get_random_input_tensors()
# print(inputs["input_img"])
# print(inputs["calib"])
# outs = run_model(inputs)
# print(outs.shape)

recurrent_state = np.zeros((1, 512), dtype=np.float32)
input_frame = get_real_input()
print(input_frame[0:1].numpy().astype(np.float32).shape)

for t_idx in range(seq_len):
    inputs = {
        'input_imgs': input_frame[t_idx:t_idx+1].numpy().astype(np.float32),
        'big_input_imgs': input_frame[t_idx:t_idx+1].numpy().astype(np.float32),
        
        'desire': np.zeros((1, 8), dtype=np.float32),
        'traffic_convention': np.array([0, 1], dtype=np.float32).reshape(1, 2),
        'initial_state': recurrent_state,
    }

    outs, recurrent_state = run_model(inputs)