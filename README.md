# Train a custom YOLOv3-tiny model and run it with a Raspberry Pi on an Intel Neural Compute Stick 2 
yes, this is a long title.

# Environment
Hardware:
- PC (with CUDA enabled GPU)
- Raspberry Pi 3 B+
- Intel Neural Compute Stick 2
- PyCam / webcam

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

If you haven't downloaded the submodules, clone https://github.com/AlexeyAB/darknet

`git clone git@github.com:AlexeyAB/darknet.git`


# Train custom YOLOv3-tiny model with darknet

navigate to the darknet repo: `cd darknet`

**Enable training on GPU (requires CUDA)**

open the file `Makefile` in your prefered text editor and set `GPU=1` and `OPENMP=1`.

```
GPU=1
CUDNN=0
CUDNN_HALF=0
OPENCV=0
AVX=0
OPENMP=1
LIBSO=0
ZED_CAMERA=0
```

**Build darknet**

Install build-essential:
```
sudo apt-get update
sudo apt-get install build-essential
```
Build:
```
make
```
Note: When using `fish` instead of `bash`, the build might fail. Just use bash in this case.

**Train a custom model based on YOLOv3-tiny**

Copy the config files to the `darknet/` directory:
```
cp signals.names darknet/
cp signals.data darknet/
cp yolov3-tiny-signals.cfg darknet/
```
The config files in this repo are altered to fit the signals-dataset. To train a model on another dataset, follow the instructions here:
https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects

Navigate to the signals-dataset folder and run the script to create a train/test split of the data.
```
cd signals-dataset
python3 create_train_test_split.py
```
This should generate a directory called `yolov3` along with two files `train.txt` and `test.txt`.

Copy the directory and the two text files to the darknet directory:
```
cp signals-dataset/train.txt darknet/
cp signals-dataset/test.txt darknet/
cp -r signals-dataset/yolov3 darknet/
```

Download the pre-trained weights file:
```
cd darknet
wget https://pjreddie.com/media/files/yolov3-tiny.weights
```
Get pre-trained weights for convolutional layers (`./darknet` is the binary inside the `darknet/` directory):
```
./darknet partial cfg/yolov3-tiny.cfg yolov3-tiny.weights yolov3-tiny.conv.15 15
```

Start training:

```
./darknet detector train signals.data yolov3-tiny-signals.cfg yolov3-tiny.conv.15
```
To calculate mean average precision after every 1000 batches (1 iteration) start training with the `-map` flag.
```
./darknet detector train signals.data yolov3-tiny-signals.cfg yolov3-tiny.conv.15 -map
```
The trained model is saved in `darknet/backup/` as `.weights` file for every iteration. When using `-map`, the model with the highets precision is saved as `..._best.weights`.


# Analyze mean average precision (mAP) for models
Run the script to export the mAP data:
```
python3 calculate_map.py
```
The script creates a `.map` file for each `.weights` file (excluding best, final and last since they would be redundant).

You can also manually check the mAP of a `.weights` file:
```
./darknet detector map signals.data yolov3-tiny-signals.cfg backup/yolov3-tiny-signals_final.weights
```

After exporting the data you can visualize it using the `create_diagrams_from_mAP_data.ipynb` jupyter notebook.


# Test the model visually
Run the following command. it will then prompt you for a path to an image. Enter the path of an image in the test-set (or any other image)
``` 
./darknet detector test signals.data yolov3-tiny-signals.cfg backup/yolov3-tiny-signals_best.weights
```


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

**Dump YOLOv3 TensorFlow Model:**

For this step you need to install tensorflow:
```
pip3 install tensorflow==1.12.2
```
IMPORTANT: The correct version of tensorflow has to be used, otherwise the conversion goes wrong! 1.12.2 works. >=1.13.0 does not work.
See: https://software.intel.com/en-us/forums/computer-vision/topic/807383

then run:

```
python3 tensorflow-yolo-v3/convert_weights_pb.py --class_names signals.names --data_format NHWC --weights_file yolov3-tiny-signals.weights --tiny
```

The TensorFlow model is saved as `frozen_darknet_yolov3_model.pb`

**Convert YOLOv3 TensorFlow Model to the IR:**

```
python3 /opt/intel/openvino_2019.1.133/deployment_tools/model_optimizer/mo_tf.py --input_model frozen_darknet_yolov3_model.pb --tensorflow_use_custom_operations_config yolo_v3_tiny_custom.json --input_shape [1,416,416,3] --data_type FP16
```
The `--input_shape` parameter is needed as otherwise it blows up due to getting -1 for the mini-batch size. Forcing this to 1 solves the problem.

To run the model on a MYRIAD processor (Intel Compute Stick 2), the parameter `--data_type FP16` has to be passed.

The original `yolo_v3_tiny.json` can be found in `<OPENVINO_INSTALL_DIR>/deployment_tools/model_optimizer/extensions/front/tf/`.

The IR is generated and saved as `frozen_darknet_yolov3_model.xml` and `frozen_darknet_yolov3_model.bin`.


# Setup Raspberry Pi

This section decribes how to setup and configure a Raspberry Pi 3 B+ to run the YOLOv3-tiny model on the Intel Neural Compute Stick 2.

## Install and configure Raspbian Stretch Lite
- Download Raspbian Stretch Lite from https://www.raspberrypi.org/downloads/raspbian/
- Install the OS on an SD card (using https://www.balena.io/etcher/)
- Create a file called `ssh` in the root of the `boot` partition of the SD card.
- Create a file called `wpa_supplicant.conf` in the `boot` partition of the SD card.
- Make sure to change the “End of Line” setting set to “UNIX” for both files!
- In `wpa_supplicant.conf` add the following content:
```
update_config=1
ctrl_interface=/var/run/wpa_supplicant

network={
  scan_ssid=1
  ssid="MyNetworkSSID"
  psk="MyNetworkPassword"
}
```


find Raspberry PI in local network:
```
sudo nmap -sP 192.168.1.0/24 | awk '/^Nmap/{ip=$NF}/B8:27:EB/{print ip}'
```


## Setup OpenVINO Toolkit on Raspberry Pi

Follow the instruction in the official guide (v 2019 R1.01): https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_raspbian.html

Maybe it's necessary to add the inference engine libraries to the PATH. Add this to the bottom of ~/.bashrc

```
export PATH=$PATH:/opt/intel/openvino/inference_engine/lib
```
Maybe you need to add this to the top of your python scripts:
```
sys.path.insert(0, '/opt/intel/openvino/python/python3.5')
```


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
or
```
python3 openvino_tiny-yolov3_MultiStick_test.py -numncs 1
```

For reference, the original python script can be found here: https://github.com/PINTO0309/OpenVINO-YoloV3

### Run examples on on CPU

To run the examples on the CPU on UBUNTU, add the following to your `~/.bashrc` file.
```
export PATH="/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/:$PATH"
```
If you get an error that `lib/libcpu_extension.so` if not found, change the line referencing the file to `libcpu_extension_sse4.so` (remove `lib/`!)
