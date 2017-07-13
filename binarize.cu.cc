#ifdef GOOGLE_CUDA

#define EIGEN_USE_GPU
#define EIGEN_USE_THREADS

#include "binarize.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

using namespace tensorflow;

#define EIGEN_USE_GPU

// Define the CUDA kernel.
template <typename T>
__global__ void BinarizeCudaKernel(const int size, const T* in, T* out) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
       i += blockDim.x * gridDim.x) {
    if (ldg(in + i) <= 0) {
        out[i] = -1;
    }
    else {
        out[i] = 1;
    }
  }
}

// Define the GPU implementation that launches the CUDA kernel.
template <typename T>
struct BinarizeFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& d, int size, const T* in, T* out) {
    // Launch the cuda kernel.
    //
    // See core/util/cuda_kernel_helper.h for example of computing
    // block count and thread_per_block count.
    int block_count = 1024;
    int thread_per_block = 20;
    BinarizeCudaKernel<T>
        <<<block_count, thread_per_block, 0, d.stream()>>>(size, in, out);
  }
};

// Instantiate functors for the types of OpKernels registered.
typedef Eigen::GpuDevice GPUDevice;
template struct BinarizeFunctor<GPUDevice, float>;
template struct BinarizeFunctor<GPUDevice, double>;
template struct BinarizeFunctor<GPUDevice, int32>;
template struct BinarizeFunctor<GPUDevice, int64>;

#endif  // GOOGLE_CUDA
