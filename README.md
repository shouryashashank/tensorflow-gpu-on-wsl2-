# tensorflow-gpu-on-wsl2-
tensorflow gpu on wsl2 with tensor rt support
## Install TensorFlow GPU on WSL2 Ubuntu 24.04 (Windows 11) | CUDA, cuDNN, TensorRT &amp; PyTorch

##################
Cuda and cuDNN:
[Click here to visit the official TensorFlow website for GPU support installation](https://www.tensorflow.org/install/source#gpu_support_2)


TensorRT Support for TensorFlow 2.16.1:
[Click here to visit the GitHub issue for TensorRT support in TensorFlow 2.16.1](https://github.com/tensorflow/tensorflow/issues/61468)
##################


##################

```bash
sudo apt update

sudo apt upgrade

sudo apt install build-essential

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

bash ./Miniconda3-latest-Linux-x86_64.sh
```


install in the hom dir 
##################


##################
check nvidia driver version 

nvidia-smi

check for update in geforce now app

wget https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda_12.1.1_530.30.02_linux.run

sudo sh cuda_12.1.1_530.30.02_linux.run

accept the eula and
┌──────────────────────────────────────────────────────────────────────────────┐
│ CUDA Installer                                                               │
│ + [X] CUDA Toolkit 12.1                                                      │
│   [X] CUDA Demo Suite 12.1                                                   │
│   [X] CUDA Documentation 12.1                                                │
│ - [ ] Kernel Objects                                                         │
│      [ ] nvidia-fs                                                           │
│   Options                                                                    │
│   Install                                                                    │
│                                                                              │
│ Up/Down: Move | Left/Right: Expand | 'Enter': Select | 'A': Advanced options │
└──────────────────────────────────────────────────────────────────────────────┘

select these options and click on install 


code ~/.bashrc

copy and paste these line in the .bashrc file and save 

export PATH=/usr/local/cuda-12.1/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

source ~/.bashrc

sudo apt-get install gedit

sudo gedit /etc/ld.so.conf

copy pase this line and save 

/usr/local/cuda-12.1/lib64

run this command and pray you dont get any thing in response 
sudo ldconfig

echo $PATH

check in the responce we have cuda path 

echo $LD_LIBRARY_PATH

check in the responce we have cuda lib64 path 


sudo ldconfig -p | grep cuda

this will return some jargan with cuda in it 

nvcc --version

this should show some thing like 

nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Mon_Apr__3_17:16:06_PDT_2023
Cuda compilation tools, release 12.1, V12.1.105
Build cuda_12.1.r12.1/compiler.32688072_0
##################

##################
install cudnn
go to this web page

https://developer.nvidia.com/rdp/cudnn-archive

select v8.9.7 for cuda 12.x
log in and download the latest tar ball bofr linux

##################
extract the tar , go to that extracted forler and copy some stuff from there:

tar -xvf cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz


cd cudnn-linux-x86_64-8.9.7.29_cuda12-archive


sudo cp include/cudnn*.h /usr/local/cuda-12.1/include

sudo cp lib/libcudnn* /usr/local/cuda-12.1/lib64

sudo chmod a+r /usr/local/cuda-12.1/include/cudnn*.h /usr/local/cuda-12.1/lib64/libcudnn*

cd ..

now check if suff are coppied

ls -l /usr/local/cuda-12.1/lib64/libcudnn*


test cudnn:

code test_cudnn.c

// test_cudnn.c
#include <cudnn.h>
#include <stdio.h>

int main() {
    cudnnHandle_t handle;
    cudnnStatus_t status = cudnnCreate(&handle);
    if (status == CUDNN_STATUS_SUCCESS) {
        printf("cuDNN successfully initialized.\n");
    } else {
        printf("cuDNN initialization failed.\n");
    }
    cudnnDestroy(handle);
    return 0;
}

Compile the code.

gcc -o test_cudnn test_cudnn.c -I/usr/local/cuda-12.1/include -L/usr/local/cuda-12.1/lib64 -lcudnn

Run the code.

./test_cudnn

It should return "cuDNN successfully initialized."

##################
Now we install TensorRT.

[Click here to visit the NVIDIA TensorRT download page](https://developer.nvidia.com/tensorrt/download)

##################
Extract and move the folder to /usr/local/TensorRT-8.6.1.

tar -xzvf TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-12.0.tar.gz

sudo mv TensorRT-8.6.1.6 /usr/local/TensorRT-8.6.1

Update the path.

code ~/.bashrc

export PATH=/usr/local/cuda-12.1/bin:/usr/local/TensorRT-8.6.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:/usr/local/TensorRT-8.6.1/lib:$LD_LIBRARY_PATH

Source the .bashrc file.

source ~/.bashrc

Check if the paths are updated by running the following commands:

echo $PATH

echo $LD_LIBRARY_PATH

Create a symbolic link for the libcudnn_adv_infer.so.8 library.

sudo ln -s /usr/local/cuda-12.1/targets/x86_64-linux/lib/libcudnn_adv_infer.so.8.x.x /usr/local/cuda-12.1/targets/x86_64-linux/lib/libcudnn_adv_infer.so.8

Create a new conda environment for TensorFlow.

conda create --name tf python=3.10

Activate the conda environment.

conda activate tf

Install TensorFlow with CUDA support.

python -m pip install tensorflow[and-cuda]

Check if TensorFlow is using GPU.

python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

Ignore any warnings during the installation.

Go to the /usr/local/TensorRT-8.6.1/python directory.

cd /usr/local/TensorRT-8.6.1/python

Install the TensorRT Python bindings.

pip install tensorrt-8.6.1-cp310-none-linux_x86_64.whl

Install the TensorRT dispatch library.

pip install tensorrt_dispatch-8.6.1-cp310-none-linux_x86_64.whl

Install the TensorRT lean library.

pip install tensorrt_lean-8.6.1-cp310-none-linux_x86_64.whl

Import TensorFlow and check if it is using GPU.

import tensorflow as tf

tf.config.list_physical_devices('GPU')

Import TensorFlow and print its version.

import tensorflow as tf
print(tf.__version__)

Install PyTorch.
Go to the [PyTorch website](https://pytorch.org/get-started/locally/) and select the compatible version for your system.

For the above setup, it will be:

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

Validate the PyTorch installation.

import torch
torch.cuda.is_available()


