# FROM nvidia/cuda:10.2-cudnn8-runtime-ubuntu18.04
ARG BASE_CONTAINER=nvcr.io/nvidia/pytorch:21.08-py3
FROM $BASE_CONTAINER

WORKDIR /app

ENV TZ=US
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub 
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub 


ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# CUDA, cuDNN, CUDArt
RUN apt-get update && apt-get install -y \
    software-properties-common \
    wget
RUN apt-get update \
    && apt-get install -y --no-install-recommends 

# Opencv related
RUN apt-get install ffmpeg libsm6 libxext6  -y
# Install Common Dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    libmlx4-1 \
    libmlx5-1 \
    librdmacm1 \
    libibverbs1 \
    libmthca1 \
    libdapl2 \
    dapl2-utils \
    openssh-client \
    build-essential \
    bzip2 \
    git \
    vim \
    cpio && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip3 install -r requirements.txt
RUN python -m spacy download en_core_web_lg
USER root

EXPOSE 8888