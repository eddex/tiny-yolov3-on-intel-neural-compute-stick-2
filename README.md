# Train a custom YOLOv3-tiny model and run it with a Raspberry Pi on an Intel Neural Compute Stick 2 
yes, this is a long title.

# Environment
Hardware:
- PC (with CUDA enabled GPU)
- Raspberry Pi 3 B+
- Intel Neural Compute Stick 2

Software:
- Ubuntu 18.04.02 LTS
- Nvidia CUDA 10.1
- OpenVINO Toolkit 2019 R1.01
- Darknet 61c9d02 - Sep 14, 2018
- Raspbian Stretch Lite - version April 2019
- Python 3.5


# Clone this repository

To simplify things, this repository contains all other necessary git repositories as submodules. To clone them together with this repository use:

```
git clone --recurse-submodules https://github.com/eddex/tiny-yolov3-on-intel-neural-compute-stick-2.git
```

Note: You can also download the repositories seperately in later steps.


# Install CUDA

To use your graphics card when training the custom YOLOv3-tiny model, install CUDA.

How to: 
[Install CUDA 10.1 on Ubuntu 18.04](https://gist.github.com/eddex/707f9cbadfaec9d419a5dfbcc2042611#file-install-cuda-10-1-on-ubuntu-18-04-md)


# Download darknet

If you haven't downloaded the submodules, clone https://github.com/pjreddie/darknet

`git clone https://github.com/pjreddie/darknet.git`


# Train custom YOLOv3-tiny model with darknet

navigate to the darknet repo: `cd darknet`

**Enable training on GPU (requires CUDA)**

open the file `Makefile` in your prefered text editor and set `GPU=1` and `OPENMP=1`.

```
GPU=1
CUDNN=0
OPENCV=0
OPENMP=1
DEBUG=0
```

**Build darknet**

```
cd darknet
make
```

Requires build-essential:
```
sudo apt-get update
sudo apt-get install build-essential
```

**Train a custom model based on YOLOv3-tiny**
- TODO

# Download and install OpenVINO

Note: You need to register to download the OpenVINO Toolkit. In my case the registration form was broken. Here's a direct download link for Linux (version 2019 R1.0.1):
http://registrationcenter-download.intel.com/akdlm/irc_nas/15461/l_openvino_toolkit_p_2019.1.133.tgz

Then follow the official installation guide:

Note: When using `fish` instead of `bash`, setting the environment variables might not work. This is not a problem. We'll use absolute paths where needed in the steps below.

https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_linux.html#install-openvino


# Convert YOLOv3-tiny model to Intermediate Representation (IR)

To run the model on the Intel Neural Compute Stick 2, we need to convert it to an "Intermediate Representation".

There's no need to follow the official guide if you use the instructions below. But for reference, it can be found here:
https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_YOLO_From_Tensorflow.html

## Option 1: using the convert.sh script

Instead of running the 2 steps below, you can run the script in this repo: 

```
sh convert.sh
```

## Option 2: Convert model manually

**Dump YOLOv3 TensorFlow Model:**

For this step you need to install tensorflow:
```
pip3 install tensorflow
```

then run:

```
python3 tensorflow-yolo-v3/convert_weights_pb.py --class_names signals.names --data_format NHWC --weights_file yolov3-tiny-signals.weights --tiny
```

The TensorFlow model is saved as `frozen_darknet_yolov3_model.pb`

**Convert YOLOv3 TensorFlow Model to the IR:**

```
python3 /opt/intel/openvino_2019.1.133/deployment_tools/model_optimizer/mo_tf.py --input_model frozen_darknet_yolov3_model.pb --tensorflow_use_custom_operations_config yolo_v3_tiny_custom.json --input_shape [1,416,416,3]
```
The `--input_shape` parameter is needed as otherwise it blows up due to getting -1 for the mini-batch size. Forcing this to 1 solves the problem.

The IR is generated and saved as `frozen_darknet_yolov3_model.xml` and `frozen_darknet_yolov3_model.bin`.


# Setup Raspberry Pi

This section decribes how to setup and configure a Raspberry Pi 3 B+ to run the YOLOv3-tiny model on the Intel Neural Compute Stick 2.

## Install and configure Raspbian Stretch Lite
- TODO


## Setup OpenVINO Toolkit on Raspberry Pi

Follow the instruction in the official guide (v 2019 R1.01): https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_raspbian.html


## Copy IR model to Raspberry Pi and run it using Python

Install dependencies (can take quite long):
```
apt update
apt install python3-pip
pip3 install opencv-python
apt install libgtk-3-dev
```

At this point there should be an **Intel Neural Compute Stick 2** and a **camera** connected to the Raspberry Pi. The camera can be a PyCam or any USB Webcam that can be detected by OpenCV.

Copy the files `openvino_tiny-yolov3_test.py`, `frozen_darknet_yolov3_model.xml` and `frozen_darknet_yolov3_model.bin` to the Raspberry Pi:
```
scp openvino_tiny-yolov3_test.py pi@192.168.1.162:~/openvino-python/
scp frozen_darknet_yolov3_model.xml pi@192.168.1.162:~/openvino-python/
scp frozen_darknet_yolov3_model.bin pi@192.168.1.162:~/openvino-python/
```

And run the python script **on the Raspberry Pi**:
```
python3 openvino_tiny-yolov3_test.py -d MYRIAD
```