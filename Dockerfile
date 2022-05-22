FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update && apt install -y software-properties-common && add-apt-repository ppa:deadsnakes/ppa

RUN apt-get update && apt-get install -y --no-install-recommends \
    gnupg2 curl ca-certificates && \
    apt-get purge --autoremove -y curl \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libxext6 \
    libx11-6 \
    vim \
    apt-utils \
    build-essential \
    curl \
    xvfb \
    xorg-dev \
    libsdl2-dev \
    swig \
    cmake \
    tmux \
    wget \
    unzip \
    scrot \
    xauth \
    python3-pip \
    python3-tk

WORKDIR /root
RUN wget https://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip && unzip -P iagreetotheeula SC2.4.10.zip
RUN mkdir /root/StarCraftII/maps \
    && cd /root/StarCraftII/maps \
    && wget https://blzdistsc2-a.akamaihd.net/MapPacks/Ladder2019Season1.zip \
    && unzip -P iagreetotheeula Ladder2019Season1.zip
    && for i in Ladder2019Season1/*; do mv $i .; done

RUN pip3 install --upgrade numpy \
                           opencv-python \
                           stable-baselines3[extra] \
                           wandb \
                           burnysc2
