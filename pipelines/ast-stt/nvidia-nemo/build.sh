#!/usr/bin/env bash

TAG=v0.0.1
docker build . -t footprintai/kserve-nvidia-nemo-stt:${TAG} -f nemo.Dockerfile
docker push footprintai/kserve-nvidia-nemo-stt:${TAG}
