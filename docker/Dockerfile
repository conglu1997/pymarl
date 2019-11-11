FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
MAINTAINER Tabish Rashid, Christian Schroeder de Witt

# CUDA includes
ENV CUDA_PATH /usr/local/cuda
ENV CUDA_INCLUDE_PATH /usr/local/cuda/include
ENV CUDA_LIBRARY_PATH /usr/local/cuda/lib64

# Ubuntu Packages
RUN apt-get update -y && apt-get install software-properties-common -y && \
    add-apt-repository -y multiverse && apt-get update -y && apt-get upgrade -y && \
    apt-get install -y apt-utils nano vim man build-essential wget sudo && \
    rm -rf /var/lib/apt/lists/*

# Install curl and other dependencies
RUN apt-get update -y && apt-get install -y curl libssl-dev openssl libopenblas-dev \
    libhdf5-dev hdf5-helpers hdf5-tools libhdf5-serial-dev libprotobuf-dev protobuf-compiler git
#RUN curl -sk https://raw.githubusercontent.com/torch/distro/master/install-deps | bash && \
#    rm -rf /var/lib/apt/lists/*

# Deal with tzdata
RUN export DEBIAN_FRONTEND=noninteractive && ln -fs /usr/share/zoneinfo/America/New_York /etc/localtime && apt-get install -y tzdata && dpkg-reconfigure --frontend noninteractive tzdata

# Install python3 pip3
RUN apt-get update
RUN apt-get -y install python3
RUN apt-get -y install python3-pip
RUN pip3 install --upgrade pip

# Python packages we use (or used at one point...)
RUN pip3 install numpy scipy pyyaml matplotlib
RUN pip3 install imageio
RUN pip3 install tensorboard-logger
RUN pip3 install pygame

RUN mkdir /install
WORKDIR /install

RUN pip3 install jsonpickle==0.9.6
# install Sacred (from OxWhirl fork)
RUN pip3 install setuptools
RUN git clone https://github.com/oxwhirl/sacred.git /install/sacred && cd /install/sacred && python3 setup.py install

#### -------------------------------------------------------------------
#### install pytorch
#### -------------------------------------------------------------------
RUN pip3 install torch
RUN pip3 install torchvision snakeviz pytest probscale

## -- SMAC
RUN pip3 install git+https://github.com/oxwhirl/smac.git
ENV SC2PATH /pymarl/3rdparty/StarCraftII

## -- BRAINBALL
RUN apt-get install -y git cmake build-essential libgl1-mesa-dev libsdl2-dev libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev libboost-all-dev libdirectfb-dev libst-dev mesa-utils xvfb x11vnc libsdl-sge-dev python3-pip
RUN git clone https://github.com/google-research/football.git && cd football && git checkout experimental && pip3 install .

WORKDIR /pymarl
