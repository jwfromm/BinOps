#ifdef GOOGLE_CUDA

#define EIGEN_USE_GPU
#define EIGEN_USE_THREADS

#include "multibit2.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

using namespace tensorflow;

#define EIGEN_USE_GPU

// Define the CUDA kernel.
template <typename T>
__global__ void Multibit2CudaKernel(const int size, const int *b, const T* in, T* out) {
  int bits = *b;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
       i += blockDim.x * gridDim.x) {
    // perform clipping
    T val;
    if (ldg(in + i) < -1) {
        val = -1;
    }
    else if (ldg(in + i) > 1) {
        val = 1;
    }
    else {
        val = ldg(in + i);
    }
    // set space between 0 and 2
    val = val + 1;
    // multiply by power of 2 to get range
    val = val * powf(2.0, bits - 1);
    // round to get binary percision
    val = floorf(val);
    // check edge case and fix
    if (val == powf(2.0, bits)) {
        val = val - 1;
    }
    // divide by max value
    val = val / (powf(2.0, bits) - 1);
    // distribute about 0
    val = val - 0.5;
    // stretch to -1 to 1 range
    val = 2 * val;
    // assign output
    out[i] = val;
  }
}

// Define the GPU implementation that launches the CUDA kernel.
template <typename T>
struct Multibit2Functor<GPUDevice, T> {
  void operator()(const GPUDevice& d, int size, const int *bits, const T* in, T* out) {
    // Launch the cuda kernel.
    //
    // See core/util/cuda_kernel_helper.h for example of computing
    // block count and thread_per_block count.
    int block_count = 1024;
    int thread_per_block = 20;
    Multibit2CudaKernel<T>
        <<<block_count, thread_per_block, 0, d.stream()>>>(size, bits, in, out);
  }
};

// Instantiate functors for the types of OpKernels registered.
typedef Eigen::GpuDevice GPUDevice;
template struct Multibit2Functor<GPUDevice, float>;
//template struct Multibit2Functor<GPUDevice, int32>;

#endif  // GOOGLE_CUDA
