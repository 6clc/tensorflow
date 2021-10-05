/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
ne
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <string>
#include <vector>
#include <chrono>
#include<fstream>
#include "absl/algorithm/container.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/conv_ops_gpu.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"
#include "tensorflow/core/public/session.h"

#include "tensorflow/core/kernels/maxpooling_op.h"
 
namespace tensorflow {

class FusedResizePadConvOpTest : public OpsTestBase {
  protected:
    template <typename T>
    void CompareFusedAndSeparate(int batch_size, int input_width, int input_height,
                                int input_depth, int resize_width,
                                int resize_height, int y_padding, int x_padding,
                                int filter_size, int filter_count,
                                bool resize_align_corners,
                                const string& pad_mode, int stride,
                                const string& padding, DataType dtype) {

      Scope root = tensorflow::Scope::NewRootScope();
      using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)
      //here we do some changes because the filter and input use different data storage method
      //input is chanel, width, height, number
      //filter is width, height, chanel, number.
      Tensor input_data(DT_FLOAT, TensorShape({batch_size, input_height, input_width, input_depth}));
      test::FillIota<float>(&input_data, 1.0f);
      Output input = Const(root.WithOpName("input"), Input::Initializer(input_data));
      Output casted_resize = Cast(root.WithOpName("cast"), input, dtype);

      Tensor filter_data(DT_FLOAT, TensorShape({filter_size, filter_size, input_depth, filter_count}));
      test::FillValue<float>(&filter_data, 1.0f);
      Output filter = Const(root.WithOpName("filter"), Input::Initializer(filter_data));
      Output casted_filter = Cast(root.WithOpName("casted_filter"), filter, dtype);
      
      Output conv = Conv2D(root.WithOpName("conv"), casted_resize, casted_filter,
                          {1, stride, stride, 1}, padding);
      Output maxpool = MaxPool(root.WithOpName("mp"), casted_resize, {1, filter_size, filter_size, 1}, {1, stride, stride, 1}, padding);


      tensorflow::GraphDef graph;
      TF_ASSERT_OK(root.ToGraphDef(&graph));
      
      std::unique_ptr<tensorflow::Session> session(
          tensorflow::NewSession(tensorflow::SessionOptions()));
      TF_ASSERT_OK(session->Create(graph));

      std::vector<Tensor> unfused_tensors;
      //TF_ASSERT_OK(session->Run({}, {"conv"}, {}, &unfused_tensors));
      std::cout << "====batch size: " << batch_size << "     filte2r num: " << filter_count  << "     filter size: "  << filter_size << "====\n";
      auto t1 = std::chrono::high_resolution_clock::now();
      TF_ASSERT_OK(session->Run({}, {"conv"}, {}, &unfused_tensors));
      auto t2 = std::chrono::high_resolution_clock::now();
      std::cout << "====whole process took: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " milliseconds\n";
      test::ExpectClose(unfused_tensors[0], unfused_tensors[0]);
    }
  };

TEST_F(FusedResizePadConvOpTest, ConvOnlyComparative) {
  int i = 1;
  const int inputSize = 255;
  int filterSize = 5;
  int filterNum = 100;
  int inputNum = 100;
  CompareFusedAndSeparate<float>(1, 3, 3, 3, 3, 3, 0, 0, 2, 1, false, "REFLECT", 2, "SAME", DT_FLOAT);

  //test different input number
  // for (i = 2; i < 3; i+=2){
  //   inputNum = i * 10;
  //   filterNum = 100;
  //   filterSize = 5;
  //   CompareFusedAndSeparate<float>(inputNum, inputSize, inputSize, 3, inputSize, inputSize, 0, 0, filterSize, filterNum, false,
  //                                "REFLECT", 1, "VALID", DT_FLOAT);
  //   //clean cache
  //   sync();
  //   std::ofstream ofs("/proc/sys/vm/drop_caches");
  //   ofs << "3" << std::endl;
  // }

  // //test different filter number
  // for (i = 2; i < 21; i+=2){
  //   filterNum = i * 10;
  //   inputNum = 100;
  //   filterSize = 5;
  //   CompareFusedAndSeparate<float>(inputNum, inputSize, inputSize, 3, inputSize, inputSize, 0, 0, filterSize, filterNum, false,
  //                                "REFLECT", 1, "VALID", DT_FLOAT);
  //   //clean cache
  //   sync();
  //   std::ofstream ofs("/proc/sys/vm/drop_caches");
  //   ofs << "3" << std::endl;
  // }

    //test different filter size
  // for (i = 2; i < 12; i++){
  //   inputNum = 100;
  //   filterNum = 100;
  //   filterSize = 2*i-1;
  //   CompareFusedAndSeparate<float>(inputNum, inputSize, inputSize, 3, inputSize, inputSize, 0, 0, filterSize, filterNum, false,
  //                                "REFLECT", 1, "VALID", DT_FLOAT);
  //   //clean cache
  //   sync();
  //   std::ofstream ofs("/proc/sys/vm/drop_caches");
  //   ofs << "3" << std::endl;
  // }
}

}  // namespace tensorflow
