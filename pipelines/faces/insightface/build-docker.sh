#!/usr/bin/env bash

TAG=v0.0.2
docker build -t footprintai/insightface:${TAG} -f Dockerfile .
docker push footprintai/insightface:${TAG}
