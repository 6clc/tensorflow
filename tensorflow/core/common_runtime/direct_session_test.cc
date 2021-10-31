/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/common_runtime/direct_session.h"

#include <map>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/function_testlib.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/costmodel.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/util/device_name_utils.h"

using namespace tensorflow;

int main() {
  Graph graph(OpRegistry::Global()); // calculation y = a*x

  Node* a;
  Tensor a_tensor(DT_FLOAT, TensorShape({2, 2}));
  NodeBuilder(graph.NewName("const_a"), "Const")
              .Attr("dtype", a_tensor.dtype())
              .Attr("value", a_tensor)
              .Finalize(&graph, &a);
  a->set_assigned_device_name("/job:localhost/replica:0/task:0/cpu:0");

  Node* x;
  Tensor x_tensor(DT_FLOAT, TensorShape({2, 1}));
  NodeBuilder(graph.NewName("const_x"), "Const")
              .Attr("dtype", x_tensor.dtype())
              .Attr("value", x_tensor)
              .Finalize(&graph, &x);
  x->set_assigned_device_name("/job:localhost/replica:0/task:0/cpu:0");
  
  Node* y;
  NodeBuilder(graph.NewName("matmul"), "MatMul")
              .Input(a)
              .Input(x)
              .Attr("transpose_a", false)
              .Attr("transpose_b", false)
              .Finalize(&graph, &y);
  y->set_assigned_device_name("/job:localhost/replica:0/task:0/cpu:0");

  GraphDef graph_def;
  graph.ToGraphDef(&graph_def);

  // session
  SessionOptions options;
  (*options.config.mutable_device_count())["CPU"] = 1;
  options.config.set_intra_op_parallelism_threads(1);
  options.config.set_inter_op_parallelism_threads(1);
  options.config.add_session_inter_op_thread_pool()->set_num_threads(1);
  options.config.set_use_per_session_threads(false); 
  auto session = std::unique_ptr<Session>(NewSession(options));
  session->Create(graph_def);

  std::vector<std::pair<string, Tensor>> inputs;
  std::vector<string> output_names = {y->name(), y->name() + ":0"};
  std::vector<Tensor> outputs;

  std::vector<string> target_nodes = {y->name()};
  Status s = session->Run(inputs, output_names, target_nodes, &outputs);
  

  std::cout << outputs[0].DebugString() << " " << outputs[1].DebugString() << std::endl;

  // 遍历graph里的edge
  std::cout << "===========edges===================" << std::endl;
  for(auto itr=graph.edges().begin(); itr != graph.edges().end(); ++itr){
    std::cout << (*itr)->DebugString() <<std::endl;
  }

   std::cout << std::endl << "===========nodes===================" << std::endl;

    // 遍历graph里的node
  for(Node* node : graph.nodes()){
    std::cout << node->DebugString() <<std::endl;
  }

  return 0;
}