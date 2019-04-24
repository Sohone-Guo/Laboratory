FROM ubuntu
MAINTAINER Sohone <878153077@qq.com>

RUN apt-get update
RUN apt-get install -y wget

# RUN wget https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh

# install Anaconda3
RUN wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-5.2.0-Linux-x86_64.sh -O ~/anaconda3.sh
RUN bash ~/anaconda3.sh -b -p /home/anaconda3 \
	&& rm ~/anaconda3.sh 
ENV PATH /home/anaconda3/bin:$PATH

RUN conda install torch 