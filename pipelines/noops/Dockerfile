FROM python:3.9-slim-bullseye

RUN apt-get update
RUN apt-get install -y build-essential
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir kserve==0.9.0

COPY . .

ENTRYPOINT ["python", "-m", "kserve-container"]
