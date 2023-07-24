FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

RUN apt-get update && \
    apt-get install ffmpeg libsm6 libxext6 wget git -y

RUN wget http://security.ubuntu.com/ubuntu/pool/main/o/openssl/libssl1.1_1.1.1f-1ubuntu2.19_amd64.deb && \
    dpkg -i libssl1.1_1.1.1f-1ubuntu2.19_amd64.deb

WORKDIR /workspace

RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir kserve==v0.10.1 && \
    pip install paddlepaddle-gpu -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY requirements-paddleocr.txt requirements-paddleocr.txt
RUN pip install --no-cache-dir -r requirements-paddleocr.txt

COPY . .

ENTRYPOINT ["python"]
