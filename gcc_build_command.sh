#!/bin/bash

# These commands will build libbinarize.so and libmultibit.so, assuming that
# TensorFlow is already installed and g++/nvcc are available on your system
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())' 2>/dev/null)
if [ ! -d "$TF_INC" ]; then
    TF_INC=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())' 2>/dev/null)
    if [ ! -d "$TF_INC" ]; then
        echo "ERROR: Could not auto-discover TF_INC; is TensorFlow actually installed?"
        exit 1
    fi
fi
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
mkdir -p ${DIR}/build

for l in binarize multibit; do
    CFLAGS="-fPIC -I ${TF_INC} -D_GLIBCXX_USE_CXX11_ABI=0"
    CUDAFLAGS="-x cu -gencode arch=compute_61,code=sm_61 -expt-relaxed-constexpr"
    LDFLAGS="-fPIC"
    
    # First, compile CPU code to .o
    g++ -std=c++11 -c ${DIR}/${l}.cc -o ${DIR}/build/${l}.o ${CFLAGS}

    OBJS="${DIR}/build/${l}.o"

    # Now, unless we're doing a NO_GPU build, nvcc it up!
    if [ -z "$NO_GPU" ]; then
        CFLAGS="${CFLAGS} -DGOOGLE_CUDA=1"

        # Compile GPU code to .cu.o (note we're super dirty here and are slipping
        # -Xcompiler in before ${CFLAGS} to overlap with -fPIC.  Yikes.)
        nvcc -std=c++11 -c ${DIR}/${l}.cu.cc -o ${DIR}/build/${l}.cu.o -Xcompiler ${CFLAGS} ${CUDAFLAGS}
        
        LDFLAGS="${LDFLAGS} -L/usr/local/cuda/lib64 -lcudart"
        OBJS="${OBJS} ${DIR}/build/${l}.cu.o"
    fi

    # Finally, link everything together into a .so
    g++ -std=c++11 -shared ${OBJS} -o ${DIR}/lib${l}.so ${LDFLAGS}
done