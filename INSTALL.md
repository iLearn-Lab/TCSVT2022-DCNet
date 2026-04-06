## Installation

This repo only requires pycocotools and Scene-Graph-Benchmark, with the original Apex dependency removed for CUDA 12 compatibility.

### Requirements:
- PyTorch >= 2.2 (Mine 2.2.2 (CUDA 12.4))
- cocoapi
- yacs
- matplotlib
- GCC >= 4.9
- OpenCV


### Step-by-step installation

```bash

conda create -n sgg python=3.10 -y
conda activate sgg

# PyTorch installation
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia


# this installs the right pip and dependencies for the fresh python
conda install ipython
conda install scipy
conda install h5py

# scene_graph_benchmark and coco api dependencies
pip install ninja yacs cython matplotlib tqdm opencv-python overrides


export INSTALL_DIR=$PWD

# install pycocotools
cd $INSTALL_DIR
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install

# install PyTorch Detection
cd $INSTALL_DIR
git clone https://github.com/hanxjing/DCNet.git
cd DCNet

# re-build it
python -m pip install -e . --no-build-isolation


unset INSTALL_DIR

