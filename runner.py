# read the onnx model
import onnxruntime as ort
import numpy as np
import sys
import os

def read(sz, tf8=False):
  dd = []
  gt = 0
  szof = 1 if tf8 else 4
  while gt < sz * szof:
    st = os.read(0, sz * szof - gt)
    assert(len(st) > 0)
    dd.append(st)
    gt += len(st)
  r = np.frombuffer(b''.join(dd), dtype=np.uint8 if tf8 else np.float32).astype(np.float32)
  if tf8:
    r = r / 255.
  return r

def write(d):
  os.write(1, d.tobytes())

def run_loop(m, tf8_input=False):
  ishapes = [[1]+ii.shape[1:] for ii in m.get_inputs()]
  keys = [x.name for x in m.get_inputs()]
  print("ready to run onnx model", keys, ishapes, file=sys.stderr)
  while 1:
    inputs = []
    for k, shp in zip(keys, ishapes):
      ts = np.product(shp)
      #print("reshaping %s with offset %d" % (str(shp), offset), file=sys.stderr)
      inputs.append(read(ts, (k=='input_img' and tf8_input)).reshape(shp))
    ret = m.run(None, dict(zip(keys, inputs)))
    print(ret, file=sys.stderr)
    for r in ret:
      write(r)


options = ort.SessionOptions()
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
options.intra_op_num_threads = 2
options.inter_op_num_threads = 8
options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
provider = 'CPUExecutionProvider'

# onnx_model is an in-memory ModelProto
ort_session = ort.InferenceSession("models/dmonitoring_model.onnx", options, providers=[provider])
run_loop(ort_session)

print(ort_session)

