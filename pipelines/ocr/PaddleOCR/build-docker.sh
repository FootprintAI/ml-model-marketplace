#!/usr/bin/env bash

TAG=v0.0.1
docker build -t footprintai/paddleocr:${TAG} -f Dockerfile .
docker push footprintai/paddleocr:${TAG}
