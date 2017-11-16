echo "Running setup"

sudo apt-get install -y python3-pip
pip3 install --upgrade pip

sudo apt-get install -y  libopenmpi-dev
sudo apt-get install -y \
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

# install cuDNN v6.0
CUDNN_TAR_FILE="cudnn-8.0-linux-x64-v6.0.tgz"
wget http://developer.download.nvidia.com/compute/redist/cudnn/v6.0/${CUDNN_TAR_FILE}
tar -xzvf ${CUDNN_TAR_FILE}
sudo cp -P cuda/include/cudnn.h /usr/local/cuda-8.0/include
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda-8.0/lib64/
sudo chmod a+r /usr/local/cuda-8.0/lib64/libcudnn*

pip3 install tensorflow-gpu==1.2.1

cd rl-teacher
pip3 install -e .
pip3 install -e human-feedback-api
pip3 install -e agents/parallel-trpo[tf_gpu]
pip3 install -e agents/pposgd-mpi[tf_gpu]
# pip3 install -e agents/ga3c[tf_gpu]

# pip3 install gym[atari]==0.9.1

curl -o /usr/bin/Xdummy https://gist.githubusercontent.com/nottombrown/ffa457f020f1c53a0105ce13e8c37303/raw/ff2bc2dcf1a69af141accd7b337434f074205b23/Xdummy
chmod +x /usr/bin/Xdummy
export "CFLAGS=-Wl,--no-as-needed -ldl"
Xdummy &
export DISPLAY=:0

#tensorboard --logdir ~/tb/rl-teacher/ --port $TENSORBOARD_PORT &
alias python=python3
export PYTHONPATH=../rl-teacher/rl-teacher/:../rl-teacher/rl-teacher/agents/parallel-trpo/:../rl-teacher/rl-teacher/agents/pposgd-mpi/:../rl-teacher/rl-teacher/agents/simple-trpo/:../rl-teacher/rl-teacher/human-feedback-api:../unity/unityagents/:../unity/
cd ../linux_bin

#python ../rl-teacher/rl_teacher/teach.py -V -a unity-pposgd-mpi -p rl -e unity-3dball-bounce1ball -n unity-base-rl

# pip install jupyter
# python -m pip install ipykernel
# ipython kernel install --name py3 --user

# echo "Jupyter is on: http://${SIMCLOUD_HOSTNAME}:${JUPYTER_PORT}"
# jupyter notebook --port ${JUPYTER_PORT} --ip="*" --allow-root

#rsync -e 'ssh -i /Users/cristian/.turibolt/bolt_ssh_key -p 34507' -avz linux_bin root@mr2-909-1213-15-srv.mr2.simcloud.apple.com:/task_runtime