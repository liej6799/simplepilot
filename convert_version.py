# does not work on mac m1

import onnx

# Load the model
model = onnx.load("models/dmonitoring_model.onnx")

# Check that the IR is well formed
onnx.checker.check_model(model)

from onnx import version_converter

# Convert to version 8
converted_model = version_converter.convert_version(model, 11)

# Save model
onnx.save(converted_model, "path_to/resnet18_v8.onnx")