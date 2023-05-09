FROM footprintai/nvidia-nemo:v1.13.0

RUN apt-get update \
&& apt-get install -y --no-install-recommends git

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir kserve Pillow joblib tensorflow opencc-python-reimplemented

# use numpy with 1.22.4 (see issue: https://github.com/NVIDIA/NeMo/issues/3501)
RUN pip uninstall -y numpy && \
    pip install numpy==1.22.4

COPY . .
ENTRYPOINT ["python"]
