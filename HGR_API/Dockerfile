FROM python:3.6

COPY requirements.txt /
RUN pip install -r requirements.txt 

COPY . /

RUN pip install --upgrade pip

RUN pip install -e shenasa-api-utils
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
