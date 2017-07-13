#ifdef GOOGLE_CUDA

#define EIGEN_USE_GPU
#define EIGEN_USE_THREADS

#include "multibit.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include <cuda_runtime.h>
//#include "cublas_v2.h"

using namespace tensorflow;

#define EIGEN_USE_GPU

// Explicitly template abs() so that we can dispatch to the correct abs/fabs/fabsf call.
// By default, we just call abs(), but for float and double, we call fabsf() and fabs().
template <typename T> T __device__ GPU_absKernel(T x) {
    return abs(x);
};
template<> float __device__ GPU_absKernel<float>(float x) {
    return fabsf(x);
}
template<> double __device__ GPU_absKernel<double>(double x) {
    return fabs(x);
}

// Sigh, CUDA doesn't have an atomicAdd() with int64 support, but it does have
// uint64 (?????) so we'll just cast it since the result is the same:
__device__ int64 atomicAdd(int64 * address, int64 val) {
    return (int64) atomicAdd((unsigned long long int *)address, (unsigned long long int)val);
}


template <typename T>
__global__ void bitUpdateKernel(const int size, const int bitindex, const int *bit_map, T* in, T* out, T* hot_sum, int* valid_count) {
    T bit_mean = *hot_sum / *valid_count;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        if (bit_map[i] > bitindex) {
            if (in[i] > 0) {
                out[i] += bit_mean;
                in[i] = in[i] - bit_mean;
            } else {
                out[i] -= bit_mean;
                in[i] = in[i] + bit_mean;
            }
        }
    }
}

template <typename T>
__global__ void bitMeanKernel(const int size, const int bitindex, const int *bit_map, const T* in, T* hot_sum, int* valid_count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        if (bit_map[i] > bitindex) {
            float val = GPU_absKernel<T>(in[i]);
            atomicAdd(hot_sum, val);
            atomicAdd(valid_count, 1);
        }
    }
} 

// Define the GPU implementation that launches the CUDA kernel.
template <typename T>
struct MultibitFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& d, int size, const int *max_bit, const int* bit_map, const T* in, T* out) {
    // Launch the cuda kernel.
    //
    // See core/util/cuda_kernel_helper.h for example of computing
    // block count and thread_per_block count.
    CudaLaunchConfig cudaconf = GetCudaLaunchConfig(size, d);
    int block_count = cudaconf.block_count;
    int thread_per_block = cudaconf.thread_per_block;

    // get the maximum seen bit value
    cudaError_t cudaStat;
    cudaStat = cudaMemset(out, 0, sizeof(T)*size);
    if (cudaStat != cudaSuccess) {
        printf("Output was not set to 0 properly\n");
    }
    // allocate some space
    T *carry_over;
    cudaStat = cudaMalloc((void **)&carry_over, sizeof(T)*size);
    if (cudaStat != cudaSuccess) {
        printf("Could not allocate carry over\n");
    }
    cudaStat = cudaMemcpy(carry_over, in, sizeof(T)*size, cudaMemcpyDeviceToDevice);
    if (cudaStat != cudaSuccess) {
        printf("Copy from input to carry over failed\n");
    }

    // now that we have things set up, lets start our algorithm
    // first figure out the maximum number of bits requested
    int bitmax;
    cudaStat = cudaMemcpy(&bitmax, max_bit, sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStat != cudaSuccess) {
        printf("Failed to pull max bit value to cpu\n");
    }
    // iterate through the number of bits we need to binarize
    int *hot_bits;
    int hot_bits_cpu;
    T *hot_sum;
    T hot_sum_cpu; 
    cudaMalloc((void **)&hot_bits, sizeof(int));
    cudaMalloc((void**)&hot_sum, sizeof(T));

    for (int i = 0; i < bitmax; i++) {
        // compute the mean of the current carry over
        cudaMemset(hot_bits, 0, sizeof(int));
        cudaMemset(hot_sum, 0, sizeof(T));
        bitMeanKernel<T> <<< block_count, thread_per_block, 0, d.stream() >>> (size, i, bit_map, carry_over, hot_sum, hot_bits);
        cudaMemcpy(&hot_bits_cpu, hot_bits, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&hot_sum_cpu, hot_sum, sizeof(T), cudaMemcpyDeviceToHost);
        // now add this bits contribution to the approximation and update carryover
        bitUpdateKernel<T> <<< block_count, thread_per_block, 0, d.stream() >>> (size, i, bit_map, carry_over, out, hot_sum, hot_bits);
    }
    cudaFree(carry_over);
    cudaFree(hot_sum);
    cudaFree(hot_bits);
  }
};

// Instantiate functors for the types of OpKernels registered.
typedef Eigen::GpuDevice GPUDevice;
template struct MultibitFunctor<GPUDevice, float>;
template struct MultibitFunctor<GPUDevice, double>;
template struct MultibitFunctor<GPUDevice, int32>;
template struct MultibitFunctor<GPUDevice, int64>;

#endif  // GOOGLE_CUDA
