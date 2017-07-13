// multibit.cc
#define EIGEN_USE_THREADS
#include "multibit2.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <math.h>

using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

REGISTER_OP("Multibit2")
        .Attr("T: {float, int32}")
        .Input("input: T")
        .Input("bits: int32")
        .Output("output: T")
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            c->set_output(0, c->input(0));
            return Status::OK();
        })
        .Doc(R"doc(
Binarizes with straight through estimator gradient with the specified bit width
output: binarized tensor
)doc");

// CPU specialization of actual computation.
template <typename T>
struct Multibit2Functor<CPUDevice, T> {
  void operator()(const CPUDevice& d, int size, const int *b, const T* in, T* out) {
    int bits = *b;
    for (int i = 0; i < size; ++i) {
        // perform clipping of the input
        T val;
        if (in[i] < -1){
            val = -1;
        }
        else if (in[i] > 1){
            val = 1;
        }
        else {
            val = in[i];
        }
        
        // set space to between 0 and 2 
        val = val + 1; 
        // multiply by the appropriate power of 2
        val = val * pow(2.0, bits - 1);
        // round to get proper precision
        val = floor(val);
        // range is now 0 to 2^(n-1) 
        // check the edge case and reduce val to proper range
        if (val == pow(2.0, bits)){
            val = val - 1;
        }
        // divide by max value to crunch into proper range
        val = val / (pow(2.0, bits) - 1);
        // distribute about 0
        val = val - 0.5;
        // stretch to between -1 and 1
        val = 2*val;
        // assign output
        out[i] = val;
    }
  }
};

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T>
class Multibit2Op : public OpKernel {
 public:
  explicit Multibit2Op(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    const Tensor& bits_tensor = context->input(1);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(bits_tensor.shape()),
                errors::InvalidArgument("multibit expects a scalar for `bits`."));
    const auto bits = bits_tensor.scalar<int>();

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));

    // Do the computation.
    OP_REQUIRES(context, input_tensor.NumElements() <= tensorflow::kint32max,
                errors::InvalidArgument("Too many elements in tensor"));
    Multibit2Functor<Device, T>()(
        context->eigen_device<Device>(),
        static_cast<int>(input_tensor.NumElements()),
        bits.data(),
        input_tensor.flat<T>().data(),
        output_tensor->flat<T>().data());
  }
};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("Multibit2").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      Multibit2Op<CPUDevice, T>);
REGISTER_CPU(float);
//REGISTER_CPU(int32);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("Multibit2").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      Multibit2Op<GPUDevice, T>);
REGISTER_GPU(float);
//REGISTER_GPU(int32);
#endif  // GOOGLE_CUDA
