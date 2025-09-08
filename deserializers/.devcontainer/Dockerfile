FROM ubuntu:20.04

RUN apt-get update

# eCAL
RUN apt-get update
RUN apt-get install -y software-properties-common
RUN apt-get update
RUN add-apt-repository ppa:ecal/ecal-latest
RUN apt-get update
RUN apt-get install -y ecal

# OpenCV
RUN apt-get install -y libopencv-dev

# packages
RUN apt-get install -y build-essential libssl-dev cmake
RUN apt-get install -y libprotobuf-dev protobuf-compiler libhdf5-dev

# XIMEA API
RUN apt-get install -y sudo git wget
WORKDIR /tmp
COPY xiapi_install.sh .
RUN ./xiapi_install.sh

# Livox API
RUN apt install -y cmake gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
WORKDIR /tmp
RUN git clone https://github.com/Livox-SDK/Livox-SDK.git
WORKDIR /tmp/Livox-SDK/build
RUN cmake .. \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc \
    -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++
RUN make
RUN make install