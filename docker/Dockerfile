FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04

MAINTAINER Josh Fromm <jwfromm@uw.edu>

# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        python3 \
        python3-dev \
        rsync \
        software-properties-common \
        unzip \
        graphviz \
        protobuf-compiler \
        x11-apps \
        vim \
        git \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

RUN pip --no-cache-dir install \
        Pillow \
        h5py \
        ipykernel \
        jupyter \
        matplotlib \
        numpy \
        pandas \
        scipy \
        sklearn \
        mixpanel \
        graphviz \
        pydot \
        pydot_ng \
        pyyaml \
        scikit-learn \
        keras \
        && \
    python -m ipykernel.kernelspec

# --- DO NOT EDIT OR DELETE BETWEEN THE LINES --- #
# These lines will be edited automatically by parameterized_docker_build.sh. #
# COPY _PIP_FILE_ /
# RUN pip --no-cache-dir install /_PIP_FILE_
# RUN rm -f /_PIP_FILE_

# Install TensorFlow GPU version.
RUN pip --no-cache-dir install \
    tensorflow-gpu
    #http://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-0.0.0-cp27-none-linux_x86_64.whl
# --- ~ DO NOT EDIT OR DELETE BETWEEN THE LINES --- #

# Set up our notebook config.
COPY jupyter_notebook_config.py /root/.jupyter/

# Jupyter has issues with being run directly:
#   https://github.com/ipython/ipython/issues/7062
# We just add a little wrapper script.
COPY run_jupyter.sh /

# Set up some convenience stuff
RUN mkdir /root/.keras
COPY .vimrc /root
COPY .gitconfig /root
COPY keras.json /root/.keras/

# For CUDA profiling, TensorFlow requires CUPTI.
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH:/usr/local/cuda/lib64/stubs

# set up BinOps repo
RUN git clone https://github.com/jwfromm/BinOps.git

WORKDIR /BinOps
ENV TF_INC /usr/local/lib/python3.5/dist-packages/tensorflow/include
RUN /bin/bash gcc_build_command.sh
RUN mv *.so /usr/local/lib
RUN ldconfig

ENV DISPLAY 0

# TensorBoard
EXPOSE 6006
# IPython
EXPOSE 8888

WORKDIR /root

#CMD ["/run_jupyter.sh", "--allow-root"]
