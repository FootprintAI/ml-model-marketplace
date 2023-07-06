#!/usr/bin/env bash

TAG=v0.0.5
docker build -t footprintai/aix-aixexplainer:${TAG} -f Dockerfile .
docker push footprintai/aix-aixexplainer:${TAG}
