// multibit.cc
#define EIGEN_USE_THREADS
#include "multibit.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <math.h>

using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice; using GPUDevice = Eigen::GpuDevice;

REGISTER_OP("Multibit")
        .Attr("T: {float, double, int32, int64}")
        .Input("input: T")
        .Input("bit_map: int32")
        .Input("max_bit: int32")
        .Output("output: T")
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            c->set_output(0, c->input(0));
            return Status::OK();
        })
        .Doc(R"doc(
Binarizes with straight through estimator gradient with the specified bit width
output: binarized tensor
)doc");

// Explicitly template abs() so that we can dispatch to the correct abs/fabs/fabsf call.
// By default, we just call abs(), but for float and double, we call fabsf() and fabs().
template <typename T> T absKernel(T x) {
    return abs(x);
};
template <> float absKernel<float>(float x) {
    return fabsf(x);
}
template <> double absKernel<double>(double x) {
    return fabs(x);
}

// CPU specialization of actual computation.
template <typename T>
struct MultibitFunctor<CPUDevice, T> {
  void operator()(const CPUDevice& d, int size, const int *max_bit, const int* bit_map, const T* in, T* out) {
    // allocate some space to store the carry over
    T* carry_over = (T*) malloc(size * sizeof(T));
    std::memcpy(carry_over, in, size * sizeof(T));
    // set all values of out to 0
    std::memset(out, 0, size * sizeof(T));
    // go ahead and pull out the max bit value
    int bitmax = *max_bit;
    // iterate through the number of bits needed to binarize
    for (int b = 0; b < bitmax; b++) {
        // compute the mean of the current carry over
        int hot_bits = 0;
        T hot_sum = 0;
        for (int i = 0; i < size; ++i) {
            if (bit_map[i] > b) {
                hot_sum += absKernel<T>(carry_over[i]);
                hot_bits += 1;
            }
        }
        T bit_mean = hot_sum / hot_bits;
        // now that mean is computed update approximation
        for (int i = 0; i < size; ++i) {
            if (bit_map[i] > b) {
                if (carry_over[i] > 0) {
                    out[i] += bit_mean;
                    carry_over[i] = carry_over[i] - bit_mean;
                } else {
                    out[i] -= bit_mean;
                    carry_over[i] = carry_over[i] + bit_mean;
                }
            }
        }
    }
    free(carry_over);
  }
};

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T>
class MultibitOp : public OpKernel {
 public:
  explicit MultibitOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    //printf("Shape: %lld\n", (*(input_tensor.shape().begin())).size);
    const Tensor& bit_map_tensor = context->input(1);
    const Tensor& max_bit_tensor = context->input(2);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(max_bit_tensor.shape()),
                errors::InvalidArgument("multibit expects a scalar for `max_bit`."));
    const auto max_bit = max_bit_tensor.scalar<int>();

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));

    // Do the computation.
    OP_REQUIRES(context, input_tensor.NumElements() <= tensorflow::kint32max,
                errors::InvalidArgument("Too many elements in tensor"));
    if (input_tensor.dims() > bit_map_tensor.dims()) {
        for(int i = 0; i < input_tensor.dim_size(0); i++) {
            auto input_slice = input_tensor.Slice(i, i+1);
            auto output_slice = output_tensor->Slice(i, i+1);
            OP_REQUIRES(context, input_slice.NumElements() == bit_map_tensor.NumElements(), errors::InvalidArgument("bit map must be same shape as input"));
            MultibitFunctor<Device, T>()(
                context->eigen_device<Device>(),
                static_cast<int>(input_slice.NumElements()),
                max_bit.data(),
                bit_map_tensor.flat<int>().data(),
                input_slice.flat<T>().data(),
                output_slice.flat<T>().data());
        }
    } else {
        OP_REQUIRES(context, input_tensor.NumElements() == bit_map_tensor.NumElements(), errors::InvalidArgument("bit map must be same shape as input"));
        MultibitFunctor<Device, T>()(
            context->eigen_device<Device>(),
            static_cast<int>(input_tensor.NumElements()),
            max_bit.data(),
            bit_map_tensor.flat<int>().data(),
            input_tensor.flat<T>().data(),
            output_tensor->flat<T>().data()); 
    }
  }
};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("Multibit").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      MultibitOp<CPUDevice, T>);
REGISTER_CPU(float);
REGISTER_CPU(double);
REGISTER_CPU(int32);
REGISTER_CPU(int64);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("Multibit").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      MultibitOp<GPUDevice, T>);
REGISTER_GPU(float);
REGISTER_GPU(double);
REGISTER_GPU(int32);
REGISTER_GPU(int64);
#endif  // GOOGLE_CUDA
