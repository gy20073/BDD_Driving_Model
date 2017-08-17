# BDD_Driving_Model
## Project Introduction:

Within BDD Driving Project, we formulate the self driving task as future egomotion prediction.
To attack the task, we collected <a name="abcd">[Berkeley DeepDrive Video Dataset](#abcd)</a> with our partner [Nexar](https://www.getnexar.com/),
proposed a FCN+LSTM model and implement it using tensorflow.

## Reference:
If you want to cite our [**paper**](https://arxiv.org/pdf/1612.01079.pdf), here is the bibtex:
```    
    @article{DBLP:journals/corr/XuGYD16,
    author  = {Huazhe Xu and
               Yang Gao and
               Fisher Yu and
               Trevor Darrell},
    title     = {End-to-end Learning of Driving Models from Large-scale Video Datasets},
    journal   = {CoRR},
    volume    = {abs/1612.01079},
    year      = {2016},
    url       = {http://arxiv.org/abs/1612.01079},
    timestamp = {Wed, 07 Jun 2017 14:41:32 +0200},
    biburl    = {http://dblp.uni-trier.de/rec/bib/journals/corr/XuGYD16},
    bibsource = {dblp computer science bibliography, http://dblp.org}
}
```
Berkeley DeepDrive Video Dataset(BDD-V)
  

BDD-V dataset will be released here. Sign up for notification when we release the data

## Implementation:
### install requirement
- tensorflow 0.11
- ffmpeg
- Python packages:
    * IPYTHON
    * PIL
    * opencv
    * scipy
    * matplotlib
    * numpy
    * sklearn
    * ffprobe

```bash
bash Anaconda2-4.3.1-Linux-x86_64.sh
sudo apt-get update
sudo apt-get -y install htop tmux wget libav-tools software-properties-common python-software-properties screen
# install the ffmpeg
sudo add-apt-repository -y ppa:mc3man/trusty-media
sudo apt-get update
sudo apt-get -y dist-upgrade
sudo apt-get -y install ffmpeg
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.11.0rc1-cp27-none-linux_x86_64.whl
sudo /home/$USER/anaconda2/bin/pip install --upgrade $TF_BINARY_URL --ignore-installed
source ~/.profile
sudo cp /home/$USER/cudnn_v5_1/include/cudnn.h /usr/local/cuda/include
sudo cp -P  /home/$USER/cudnn_v5_1/lib64/* /usr/local/cuda/lib64
pip install opencv-python
conda install libgcc
```
### data setup:
Download [here](https://goo.gl/forms/7XThUcjpGALkqxFU2)
data prepare:
1. extract all the video file path into one single file.
2. run generate_index.sh(will be there) to generate a list of videos.
```bash
3. python prepare_tfrecords.py --video_index /path/to/your/file --output_directory /path/to/your/output/dir --low_res False
```
### evaluate model:
        Here are the models(url) we proposed in our paper.
        # cd ./ && python scripts/train_car_stop.py eval config_name
train model:
        To train model, we provide the pretrained model to accelerate the training stage:

        # make dir and copy the initial pretrained model
        # cd ./ && python scripts/train_car_stop.py train config_name
tensorboard:

        # cd ./ && python scripts/train_car_stop.py board config_name
        # ssh -N -L 7231:localhost:7231 leviathan.ist.berkeley.edu &
        # open the browser for tensorboard and record the experiment on excel


run
    cd /data/yang/si/ && python scripts/train_car_stop.py test camera2_speed_only
    tensorboard: do the port forwarding, open the browser for tensorboard





