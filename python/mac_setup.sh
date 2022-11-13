
# install forge
brew install miniforge
conda init zsh

# setup python env
conda create -n simplepilot -y python=3.8.6
conda activate simplepilot

conda install -c conda-forge -y opencv
conda install -c conda-forge -y numpy
conda install -c conda-forge -y matplotlib

conda install -c conda-forge -y onnx
conda install -c conda-forge -y onnxruntime