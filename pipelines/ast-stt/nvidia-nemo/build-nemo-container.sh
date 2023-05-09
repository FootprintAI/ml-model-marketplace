#!/usr/bin/env bash

git clone https://github.com/NVIDIA/NeMo.git --branch v1.13.0 --single-branch

# the following build may take huge chunk of memory.
# our latest build took 8vCPU and 40+G memory
cd NeMo && \
    DOCKER_BUILDKIT=1 docker build -t footprintai/nvidia-nemo:v1.13.0 -f Dockerfile .

# modelfile can be found here
# https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/results.html#english
