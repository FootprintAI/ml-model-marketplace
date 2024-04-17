#!/usr/bin/env bash

TAG=v0.0.2
docker build -t footprintai/kserve-whisper:${TAG} -f Dockerfile .
docker push footprintai/kserve-whisper:${TAG}
