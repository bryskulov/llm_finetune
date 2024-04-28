FROM nvidia/cuda:12.4.1-devel-ubuntu20.04

ENV LANG C.UTF-8

ARG PYTHON_VERSION=
ARG CONDA_ENV_NAME=

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y -qq --no-install-recommends \
    apt-utils \
    build-essential \
    sudo \
    cmake \
    git \
    curl \
    vim \
    ca-certificates \
    libglib2.0-0 \
    libjpeg-dev \
    libpng-dev \
    ffmpeg    \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ssh \
    wget \
    unzip \
    tmux \
    && rm -rf /var/lib/apt/lists/*
    
RUN curl -o /tmp/miniconda.sh -sSL http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash /tmp/miniconda.sh -bfp /usr/local && \
    rm -rf /tmp/miniconda.sh

RUN conda update -y conda && \
    conda create -n ${CONDA_ENV_NAME} python=${PYTHON_VERSION}

ENV PATH /usr/local/envs/$CONDA_ENV_NAME/bin:$PATH
RUN echo "source activate ${CONDA_ENV_NAME}" >> ~/.bashrc

SHELL ["/bin/bash", "-c"]

COPY docker/requirements.txt /tmp/requirements.txt
RUN source activate ${CONDA_ENV_NAME} && pip install --no-cache-dir -r /tmp/requirements.txt

RUN source activate ${CONDA_ENV_NAME} && pip install torch==2.2.1 transformers
