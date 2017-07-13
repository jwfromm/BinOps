// binarize.cc
#define EIGEN_USE_THREADS
#include "binarize.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

REGISTER_OP("Binarize")
        .Attr("T: {float, double, int32, int64}")
        .Input("input: T")
        .Output("output: T")
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            c->set_output(0, c->input(0));
            return Status::OK();
        })
        .Doc(R"doc(
Binarizes with straight through estimator gradient
output: binarized tensor
)doc");

// CPU specialization of actual computation.
template <typename T>
struct BinarizeFunctor<CPUDevice, T> {
  void operator()(const CPUDevice& d, int size, const T* in, T* out) {
    for (int i = 0; i < size; ++i) {
        if (in[i] <= 0){
            out[i] = -1;
        }
        else{
            out[i] = 1;
        }
    }
  }
};

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T>
class BinarizeOp : public OpKernel {
 public:
  explicit BinarizeOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));

    // Do the computation.
    OP_REQUIRES(context, input_tensor.NumElements() <= tensorflow::kint32max,
                errors::InvalidArgument("Too many elements in tensor"));
    BinarizeFunctor<Device, T>()(
        context->eigen_device<Device>(),
        static_cast<int>(input_tensor.NumElements()),
        input_tensor.flat<T>().data(),
        output_tensor->flat<T>().data());
  }
};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("Binarize").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      BinarizeOp<CPUDevice, T>);
REGISTER_CPU(float);
REGISTER_CPU(double);
REGISTER_CPU(int32);
REGISTER_CPU(int64);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("Binarize").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      BinarizeOp<GPUDevice, T>);
REGISTER_GPU(float);
REGISTER_GPU(double);
REGISTER_GPU(int32);
REGISTER_GPU(int64);
#endif  // GOOGLE_CUDA
