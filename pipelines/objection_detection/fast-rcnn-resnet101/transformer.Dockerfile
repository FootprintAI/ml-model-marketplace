FROM python:3.7-slim

RUN apt-get update \
&& apt-get install -y --no-install-recommends git

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir kserve Pillow joblib tensorflow

COPY . .
ENTRYPOINT ["python", "-m", "image_transformer"]
