FROM python:3.10-slim-bullseye

#RUN apt-get update && \
#    apt-get install git ffmpeg libsm6 libxext6 -y

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir kserve==v0.10.1

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["python"]
