#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <turbojpeg.h>

using namespace tensorflow;
using shape_inference::ShapeHandle;
using shape_inference::InferenceContext;

REGISTER_OP("DecodeJpegBatch")
    .Input("raw: string")
    .Attr("H: int")
    .Attr("W: int")
    .Output("jpegs: uint8")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {

      ShapeHandle input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &input));
      c->set_output(0,
                c->MakeShape({c->Dim(input, 0),
                              InferenceContext::kUnknownDim,
                              InferenceContext::kUnknownDim, 3}));
      return Status::OK();
    });


class DecodeJpegBatchOp : public OpKernel {
 public:
  explicit DecodeJpegBatchOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("H", &H));
    OP_REQUIRES_OK(context, context->GetAttr("W", &W));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input_tensor = context->input(0);
    int nImages = input_tensor.dim_size(0);
    auto input_eigen=input_tensor.vec<string>();

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0,
                                                     TensorShape({nImages, H, W, 3}),
                                                     &output_tensor));
    uint8* output = output_tensor->flat<uint8>().data();

    tjhandle _jpegDecompressor = tjInitDecompress();
    for(int i=0; i < nImages; ++i){
        auto this_image = input_eigen(i);
        const uint8* raw_data = (const uint8*)this_image.data();
        int raw_data_size = this_image.size();

        // call libjpeg-turbo
        long unsigned int _jpegSize = raw_data_size; //!< _jpegSize from above
        unsigned char* _compressedImage = (unsigned char*)raw_data; //!< _compressedImage from above

        int jpegSubsamp, width, height;
        unsigned char* buffer = (unsigned char*)(output + i*H*W*3); //!< will contain the decompressed image

        tjDecompressHeader2(_jpegDecompressor, _compressedImage, _jpegSize, &width, &height, &jpegSubsamp);
        tjDecompress2(_jpegDecompressor, _compressedImage, _jpegSize, buffer, width, 0/*pitch*/, height,
                      TJPF_RGB,
                      TJFLAG_FASTDCT | TJFLAG_NOREALLOC);
    }
    tjDestroy(_jpegDecompressor);
  }

  private:
  int H, W;
};
REGISTER_KERNEL_BUILDER(Name("DecodeJpegBatch").Device(DEVICE_CPU), DecodeJpegBatchOp);
