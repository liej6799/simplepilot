import importlib
import numpy as np
import onnx
importlib
import onnxruntime as rt

MODEL_WIDTH = 1440;
MODEL_HEIGHT = 960;

#load fake image
def get_random_input_tensors():
  np_inputs = {
    'input_img': np.zeros((1, 1382400), dtype=np.float32),
    'calib': np.zeros((1, 3), dtype=np.float32),
  }
  return np_inputs

def load_inference_model(path_to_model):

    onnx_graph = onnx.load(path_to_model)
    output_names = [node.name for node in onnx_graph.graph.output]
    model = rt.InferenceSession(path_to_model, providers=['CPUExecutionProvider'])

    def run_model(inputs):
        outs =  model.run(output_names, inputs)[0]
        return outs

    return model, run_model


# run on the model
onnx_model, run_model = load_inference_model("models/dmonitoring_model.onnx")

#inference on the model
inputs = get_random_input_tensors()
print(inputs["input_img"])
print(inputs["calib"])
outs = run_model(inputs)
print(outs.shape)

