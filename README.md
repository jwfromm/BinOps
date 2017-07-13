TensorFlow Binarize Ops
=======================

This repository contains the binarized Dense/Conv2D ops develoepd by @jwfromm for TensorFlow and Keras.

## Building

There are two ways to build these custom ops:

* If you are developing on TensorFlow itself, you can mount this repository within the TensorFlow source tree and use `bazel` to build the ops.  Mount this repository at a location such as `<tensorflow root>/tensorflow/core/user_ops/BinOps`, then run `./tensorflow/core/user_ops/BinOps/bazel_build_command.sh` to invoke `bazel` to build the necessary libraries.

* If you have TensorFlow already installed through something like `pip`, use `./gcc_build_command.sh` to build the libraries.

Once the `.so` libraries are built, install them into `/usr/local/lib` and run `sudo ldconfig`.  You can then place the python files `binarize_layers.py` and `binarize_ops.py` wherever is convenient for python to pick them up from your own code.

## Testing

To test this code, use `run_tests.sh`, which will build the library, run a set of unit tests against the codebase.  It is highly encouraged to use `docker` to provide the codebase with a uniform environment already containing such necessities like `tensorflow` and `CUDA` preinstalled.

To run tests on CPU-only kernels:
```
docker run -e NO_GPU=1 -ti -v $(pwd):/src -w /src tensorflow/tensorflow:latest-devel-py3 ./run_tests.sh
```

To run tests on GPU and CPU kernels, run:
```
nvidia-docker run -ti -v $(pwd):/src -w /src tensorflow/tensorflow:latest-devel-gpu-py3 ./run_tests.sh
```

Of course, if you don't want to use docker and instead just want to run the tests directly on your host system, you can do so simply by running `./run_tests.sh`.