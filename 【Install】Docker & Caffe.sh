sudo apt-get update

sudo apt-get install -y\
    apt-transport-https \
    ca-certificates \
    curl \
    software-properties-common

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

sudo apt-key fingerprint 0EBFCD88

sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"


sudo apt-get update

sudo apt-get install -y docker-ce

apt-cache madison docker-ce

sudo docker run hello-world

# Next is the caffe install
docker pull tleyden5iwx/caffe-cpu-master

run -i -t -v /root/caffe:/caffe tleyden5iwx/caffe-cpu-master


# ssd
git clone https://github.com/weiliu89/caffe.git
cd caffe
git checkout ssd

cp Makefile.config.example Makefile.config
# change the Make open the cpu_only
make -j8

"""
if -lopenblas:
apt install liblapack-dev liblapack3 libopenblas-base libopenblas-dev
"""

# name the env
export PYTHONPATH=/caffe/caffe/python
export HOME=/caffe/caffe
export CAFFE_ROOT=/caffe/caffe

make py
make test -j8

cd $CAFFE_ROOT
./data/VOC0712/create_list.sh

"""
if libdc1394 error: Failed to initialize libdc1394
ln -s /dev/null /dev/raw1394

"""
./data/VOC0712/create_data.sh

python examples/ssd/ssd_pascal.py

# evaluate
python examples/ssd/score_ssd_pascal.py

# test
python examples/ssd/ssd_pascal_webcam.py


