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

#include "tensorflow/core/graph/graph.h"

#include <set>
#include <unordered_map>
#include <vector>
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"

using namespace tensorflow;

int main() {
  Graph graph(OpRegistry::Global());
  Node* source = graph.source_node();
  Node* sink = graph.sink_node();
  std:: cout << graph.num_node_ids() <<std::endl;

  // 遍历graph里的edge
  for(auto itr=graph.edges().begin(); itr != graph.edges().end(); ++itr){
    std::cout << (*itr)->DebugString() <<std::endl;
  }
  return 0;
}