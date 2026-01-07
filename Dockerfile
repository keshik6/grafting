# Inspired by https://github.com/anibali/docker-pytorch/blob/master/dockerfiles/1.10.0-cuda11.3-ubuntu20.04/Dockerfile
# ARG COMPAT=0
ARG PERSONAL=0
# FROM nvidia/cuda:11.3.1-devel-ubuntu20.04 as base-0
FROM nvcr.io/nvidia/pytorch:22.12-py3 as base


ENV HOST docker
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
# https://serverfault.com/questions/683605/docker-container-time-timezone-will-not-reflect-changes
ENV TZ America/Los_Angeles
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone


RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    ca-certificates \
    sudo \
    less \
    htop \
    git \
    tzdata \
    wget \
    tmux \
    zip \
    unzip \
    zsh stow subversion fasd \
    && rm -rf /var/lib/apt/lists/*
    # openmpi-bin \

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN mkdir -p /home/user && chmod 777 /home/user
WORKDIR /home/user

# Set up personal environment
# FROM base-${COMPAT} as env-0
FROM base as env-0
FROM env-0 as env-1
# Use ONBUILD so that the dotfiles dir doesn't need to exist unless we're building a personal image
# https://stackoverflow.com/questions/31528384/conditional-copy-add-in-dockerfile
ONBUILD COPY dotfiles ./dotfiles
ONBUILD RUN cd ~/dotfiles && stow bash zsh tmux && sudo chsh -s /usr/bin/zsh $(whoami)
# nvcr pytorch image sets SHELL=/bin/bash
ONBUILD ENV SHELL=/bin/zsh

FROM env-${PERSONAL} as packages

# Disable pip cache: https://stackoverflow.com/questions/45594707/what-is-pips-no-cache-dir-good-for
ENV PIP_NO_CACHE_DIR=1

# General packages that we don't care about the version
RUN pip install pytest matplotlib jupyter ipython ipdb gpustat scikit-learn spacy munch einops opt_einsum fvcore gsutil cmake pykeops zstandard psutil h5py twine gdown \
    && python -m spacy download en_core_web_sm
# hydra
RUN pip install hydra-core==1.3.1 hydra-colorlog==1.2.0 hydra-optuna-sweeper==1.2.0 pyrootutils rich

# Core packages
RUN pip install transformers==4.25.1 datasets==2.8.0 triton==2.0.0 wandb==0.13.7 timm==0.6.12 torchmetrics==0.10.3

# For MLPerf
RUN pip install git+https://github.com/mlcommons/logging.git@2.1.0

# Install FlashAttention
RUN pip install flash-attn==2.5.9.post1

# Install CUDA extensions for fused dense
RUN pip install git+https://github.com/HazyResearch/flash-attention@v2.5.9.post1#subdirectory=csrc/fused_dense_lib


# Install Causal Conv1D library (Build from source)
RUN git clone https://github.com/Dao-AILab/causal-conv1d.git
WORKDIR /home/user/causal-conv1d
RUN git checkout f8c2467

# Run setup.py bdist_wheel with environment variables
RUN MAX_JOBS=2 CAUSAL_CONV1D_FORCE_BUILD="TRUE" python setup.py bdist_wheel --dist-dir=dist -v
RUN pip uninstall ninja
RUN pip install ninja
RUN pip install dist/*.whl

# Set HOME DIR
WORKDIR /home/user/

# Install additional packages
RUN pip install diffusers==0.27.2
RUN pip install accelerate==0.29.2
RUN pip install --force-reinstall typing-extensions==4.5.0
RUN pip install webdataset==0.2.86
RUN pip install rich==13.7.1
RUN pip install huggingface-hub==0.25.2