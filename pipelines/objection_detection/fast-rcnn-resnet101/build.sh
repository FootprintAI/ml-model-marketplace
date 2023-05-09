#!/usr/bin/env bash

TAG ?= latest
docker build . -t footprintai/coco-object-detector:${TAG} -f transformer.Dockerfile
docker push footprintai/coco-object-detector:${TAG}
