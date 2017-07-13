#!/bin/bash

# These commands will build libbinarize.so and libmultibit.so, assuming this
# repository has been mounted at `<tensorflow_root>/core/user_ops/BinOps`
for l in libbinarize.so libmultibit.so; do
    bazel build -c opt --config=cuda \
        //tensorflow/core/user_ops/BinOps:$l \
        --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0"
done