# FROM nvidia/cuda:10.1-cudnn7-devel
FROM nvidia/cuda:10.1-devel-ubuntu18.04

ENV DEBIAN_FRONTEND noninteractive

RUN cd /var/lib/apt/lists/ && rm -fr * && cd /etc/apt/sources.list.d/ && rm -fr * && cd /etc/apt && \
         cp sources.list sources.list.old &&  cp sources.list sources.list.tmp && \
         sed 's/ubuntuarchive.hnsdc.com/us.archive.ubuntu.com/' sources.list.tmp | tee sources.list &&\
          rm sources.list.tmp* && apt-get clean && apt-get update

RUN apt-get update && apt-get install -y \
    apache2 \
    curl \
    git \
    python3.7 \
    python3-pip
	
RUN apt-get update && apt-get install -y \
	python3-opencv ca-certificates wget sudo  \
	cmake ninja-build protobuf-compiler libprotobuf-dev 
# RUN ln -sv /usr/bin/python3 /usr/bin/python

# create a non-root user
# ARG USER_ID=1000
# RUN useradd -m --no-log-init --system  --uid ${USER_ID} appuser -g sudo
# RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
# USER appuser
# WORKDIR /home/appuser

# ENV PATH="/home/appuser/.local/bin:${PATH}"
ENV PATH="/root/.local/bin:${PATH}"

RUN wget https://bootstrap.pypa.io/get-pip.py && \
	python3 get-pip.py --user && \
	rm get-pip.py
# RUN apt-get update
# RUN apt-get install libav-tools
RUN apt-get install ffmpeg -y

# install dependencies
# See https://pytorch.org/ for other options if you use a different version of CUDA
# RUN pip install --user tensorboard
# RUN pip install --user flask 

# RUN pip install --user ffmpeg-python
# RUN pip install --user azure-storage-blob==12.1.0 

# RUN pip install --user 'git+https://github.com/facebookresearch/fvcore'
# RUN pip install --user opencv-python
# RUN pip install --user azure-cosmos 
RUN mkdir /app
COPY ./source_code/requirements.txt /app/

RUN pip install -r /app/requirements.txt --ignore-installed

RUN pip install --user torch==1.6 torchvision==0.7 -f https://download.pytorch.org/whl/cu101/torch_stable.html

RUN pip install --user streamlit --use-feature=2020-resolver

# install detectron2
# RUN sudo mkdir -p /app

# RUN git clone https://github.com/facebookresearch/detectron2 detectron2_repo
# set FORCE_CUDA because during `docker build` cuda is not accessible
ENV FORCE_CUDA="1"
# This will by default build detectron2 for all common cuda architectures and take a lot more time,
# because inside `docker build`, there is no way to tell which architecture will be used.
ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

COPY ./source_code/ /app/

# ENV CUDA_HOME='/usr/local/cuda'


# Set a fixed model cache directory.
ENV FVCORE_CACHE="/tmp"
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# WORKDIR /app/detectron2


EXPOSE 5000
RUN pip install --user -e /app/docker_files/detectron2

# ENTRYPOINT ["tail", "-f", "/dev/null"]

WORKDIR /app
ENTRYPOINT ["python3"]
CMD ["flask_app.py"]

# Pull Blob storage video > process it locally (trimming)> Do OD > save output to Cosmos DB


# change name in setup.py
# build locally 
# track the name of pkg

# ENTRYPOINT ["python3"]
# CMD ["/app/flask_app.py"]


# run detectron2 under user "appuser":
# wget http://images.cocodataset.org/val2017/000000439715.jpg -O input.jpg
# python3 demo/demo.py  \
	#--config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
	#--input input.jpg --output outputs/ \
	#--opts MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl


