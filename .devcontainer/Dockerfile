ARG PYTHON_VERSION="3.9"
FROM python:${PYTHON_VERSION}

RUN apt-get -y update
RUN apt-get -y install git
RUN apt-get install ffmpeg libsm6 libxext6  -y
COPY requirements.txt .
RUN pip3 install -r requirements.txt
