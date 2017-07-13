#!/bin/bash

# If we don't have py.test, install that!
if [ -z $(which py.test) ]; then
    echo "py.test not found, installing..."
    pip install --user pytest
    export PATH=$PATH:~/.local/bin
fi

# If we don't have keras, install that too
if [ -z $(pip list --format=columns | grep keras) ]; then
    pip install --user keras
fi

# First, build the libraries
echo "Building libraries..."
./gcc_build_command.sh

# Instead of installing to /usr/local/lib and running `sudo ldconfig` we'll
# override `LD_LIBRARY_PATH`, but you should ONLY DO THIS TEMPORARILY, for real
# work, just bite the bullet and do things the proper way with `ldconfig`.
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(pwd)

# Next, run the tests on CPU
echo "Running CPU tests..."
CUDA_DEVICE_ORDER="PCI_BUS_ID" CUDA_VISIBLE_DEVICES="" py.test binarize_tests.py

# Next, unless we forbid it, run tests on GPU
if [ -z "$NO_GPU" ]; then
    echo "Running GPU tests..."
    py.test binarize_tests.py
fi