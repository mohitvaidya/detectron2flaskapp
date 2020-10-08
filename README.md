# flaskappdetectron2
IMAGE="thenewj.azurecr.io/detectflask:v2"

Flask API implementation for detectron2
Repository contains source code inspired from https://github.com/facebookresearch/detectron2. Please follow below directory structure for the installation.
* Source code:
  * This folder contains all relevant utility for exisiting API
  https://portal.azure.com/#@newjournalism.onmicrosoft.com/resource/subscriptions/840f06dd-b466-45be-bc1e-914a21b7742c/resourceGroups/Netra-  Mumbai/providers/Microsoft.ContainerInstance/containerGroups/d2flaskapp/overview
    * Custom build of detectron2, docker_files/detectron2
    * FFMPEG utils
    * Main inference script and supporting configuration(.yaml, predictor.py)
    * Pull Util
    * Python request util
    * Python dependencies, requirements.txt
    * TImer util
* Dockerfile:
  * docker build: 
    docker build -t ${IMAGE} -f Dockerfile .
  * docker run : 
    docker run --gpus all -it --rm --shm-size=8g ${IMAGE}


All the require docker images can be pulled from ACR named "thenewj", which is available at https://portal.azure.com/#@newjournalism.onmicrosoft.com/resource/subscriptions/840f06dd-b466-45be-bc1e-914a21b7742c/resourceGroups/kubeflow/providers/Microsoft.ContainerRegistry/registries/thenewj/overview
