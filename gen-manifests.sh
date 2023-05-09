#!/usr/bin/env sh


if [ -f /.dockerenv ]; then
    echo "inside a container, use /app as default working dir"

    cd /app
fi

python3 gen-manifests.py
