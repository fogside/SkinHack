FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04

MAINTAINER Alex Ozerin <m12sl@yandex.ru>

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        gfortran \
        git \
        libfreetype6-dev \
        libhdf5-dev \
        libjpeg-dev \
        liblapack-dev \
        libopenblas-dev \
        libpng12-dev \
        libyaml-dev \
        libzmq3-dev \
        pkg-config \
        python3 \
        python3-dev \
        python3-numpy \
        python3-pip \
        python3-scipy \
        rsync \
        software-properties-common \
        swig \
        unzip \
        vim \
        wget \
        zip \
        zlib1g-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install \
        setuptools \
        --upgrade pip \
    && pip3 install --pre \
        click \
        cython \
        jupyter \
        ipykernel \
        matplotlib \
        tqdm \
        https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.11.0rc2-cp35-cp35m-linux_x86_64.whl \
    && python3 -m ipykernel.kernelspec

RUN pip3 install \
    scikit-learn \
    scikit-image \
    keras

WORKDIR "/src"