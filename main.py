
import numpy as np
import onnx
import os
import time
import io
from extra.onnx import get_run_onnx
from tinygrad.tensor import Tensor

#driver monitoring
def get_random_input_tensors():
  np_inputs = {

    # uncomment for dmonitoring
    #"input_img": np.random.randn(*(1, 1382400)),
    #"calib": np.random.randn(*(1,3))
    
    #
    "input_imgs": np.random.randn(*(1, 12, 128, 256)),
    "big_input_imgs": np.random.randn(*(1, 12, 128, 256)),
    "desire": np.zeros((1, 8)),
    "traffic_convention": np.array([[1., 0.]]),
    "initial_state": np.zeros((1, 512))
    #"initial_state": np.zeros((1, 768))
  }

  #import picklexaxwxwxw
  #frames, big_frames, last_state, frame_inputs,xx policy_outs = pickle.load(open("openpilot/test/frame_0.pkl", "rb"))
  #np_inputs["input_imgs"] = frames
  #np_inputs["big_input_imgs"] = big_frames
  #np_inputs["initial_state"] = last_state[0]

  #for i,k in enumerate(np_inputs.keys()):
  #  dat = open("/home/batman/openpilot/xx/ml_tools/snpe/compile_test_data/dlc_input_%d" % i, "rb").read()
  #  np_inputs[k] = np.frombuffer(dat, np.float32).reshape(np_inputs[k].shape)

  np_inputs = {k:v.astype(np.float32) for k,v in np_inputs.items()}
  inputs = {k:Tensor(v.astype(np.float32), requires_grad=False) for k,v in np_inputs.items()}
  for _,v in inputs.items(): v.realize()
  return inputs, np_inputs


inputs, _ = get_random_input_tensors()

onnx_model = onnx.load("models/supercombo.onnx")
run_onnx = get_run_onnx(onnx_model)
tinygrad_out = run_onnx(inputs)["outputs"]
print(tinygrad_out)