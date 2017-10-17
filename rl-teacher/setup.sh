echo "Running setup"
#pip install tensorflow-gpu


sudo apt-get install -y python3-pip

pip3 install --upgrade pip

sudo apt-get install -y  libopenmpi-dev
sudo apt-get update && sudo apt-get install -y \
    ffmpeg \
    libav-tools \
    libpq-dev \
    libjpeg-dev \
    cmake \
    swig \
    python-opengl \
    libboost-all-dev \
    libsdl2-dev \
    xpra

#pip3 install --upgrade tensorflow-gpu==1.2.1

# install cuDNN v6.0
CUDNN_TAR_FILE="cudnn-8.0-linux-x64-v6.0.tgz"
wget http://developer.download.nvidia.com/compute/redist/cudnn/v6.0/${CUDNN_TAR_FILE}
tar -xzvf ${CUDNN_TAR_FILE}
sudo cp -P cuda/include/cudnn.h /usr/local/cuda-8.0/include
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda-8.0/lib64/
sudo chmod a+r /usr/local/cuda-8.0/lib64/libcudnn*


pip3 install tensorflow-gpu==1.2.1


cd rl-teacher-atari
pip3 install -e .
pip3 install -e human-feedback-api
pip3 install -e agents/parallel-trpo[tf_gpu]
pip3 install -e agents/pposgd-mpi[tf_gpu]
pip3 install -e agents/ga3c[tf_gpu]

pip3 install gym[atari]==0.9.1


curl -o /usr/bin/Xdummy https://gist.githubusercontent.com/nottombrown/ffa457f020f1c53a0105ce13e8c37303/raw/ff2bc2dcf1a69af141accd7b337434f074205b23/Xdummy
chmod +x /usr/bin/Xdummy
export "CFLAGS=-Wl,--no-as-needed -ldl"
Xdummy &

tensorboard --logdir ~/tb/rl-teacher/ --port $TENSORBOARD_PORT &
alias python=python3

#python rl_teacher/teach.py -e Pong-v0 -n rl-test -p rl

#pip3 install --upgrade dask==0.14.3
#pip3 install keras==2.0.8

# pip install matplotlib
# pip install mujoco-py~=0.5.7
# pip install gym[mujoco]

# pip install virtualenv
# virtualenv -p python3 ~/py3venv
# source ~/py3venv/bin/activate


# pip install jupyter
# python -m pip install ipykernel
# ipython kernel install --name py3 --user






# echo "Setup step done"

# echo "Jupyter is on: http://${SIMCLOUD_HOSTNAME}:${JUPYTER_PORT}"
# jupyter notebook --port ${JUPYTER_PORT} --ip="*" --allow-root
