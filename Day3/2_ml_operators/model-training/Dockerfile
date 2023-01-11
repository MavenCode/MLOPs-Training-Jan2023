FROM python:3.10.4-slim-buster

WORKDIR /app

RUN pip3 install --user torch torchvision matplotlib --no-cache-dir

COPY model_train.py /app