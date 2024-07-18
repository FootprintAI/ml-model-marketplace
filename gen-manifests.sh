#!/usr/bin/env sh


if [ -f /.dockerenv ]; then
    echo "inside a container, use /app as default working dir"

    cd /app-script && python3 gen-manifests.py --appdir=/app
else
    python3 gen-manifests.py --appdir=./
fi

