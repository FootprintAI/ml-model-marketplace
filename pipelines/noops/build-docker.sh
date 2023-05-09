#!/usr/bin/env bash

TAG ?= latest
docker build -t footprintai/kserve-noops:${TAG} -f Dockerfile .
docker push footprintai/kserve-noops:${TAG}
