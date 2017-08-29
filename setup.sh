#!/usr/bin/env bash

# install basic utilities
sudo apt-get update
sudo apt-get -y install wget libav-tools software-properties-common python-software-properties

# install Anaconda
cd ~/
wget https://repo.continuum.io/archive/Anaconda2-4.4.0-Linux-x86_64.sh
bash Anaconda2-4.4.0-Linux-x86_64.sh

if [ x`lsb_release -rs | cut -f1 -d.`x = x14x ]; then
    # add the extra repo on ubuntu 14 for installing ffmpeg
    sudo add-apt-repository -y ppa:mc3man/trusty-media
    sudo apt-get update
    sudo apt-get -y dist-upgrade
fi
sudo apt-get -y install ffmpeg

# install tensorflow 0.11
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.11.0rc1-cp27-none-linux_x86_64.whl
sudo /home/$USER/anaconda2/bin/pip install --upgrade $TF_BINARY_URL --ignore-installed

pip install opencv-python

source ~/.profile
conda install libgcc
