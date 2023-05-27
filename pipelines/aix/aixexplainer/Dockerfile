FROM python:3.9-slim

RUN apt-get update && \
    apt-get install git ffmpeg libsm6 libxext6 -y

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir kserve==v0.10.1 && \
    pip install git+https://github.com/Trusted-AI/AIX360.git@b48a3be3befd4c365a7b189808a9c99257800c68

COPY aixserver/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["python", "-m", "aixserver"]
