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

namespace tensorflow {
namespace {

// REGISTER_OP("OneInput").Input("x: float");

// REGISTER_OP("OneOutput").Output("y: float");

// REGISTER_OP("OneInputTwoOutputs")
//     .Input("x: float")
//     .Output("y: float")
//     .Output("z: float");

// REGISTER_OP("TwoInputsOneOutput")
//     .Input("x: float")
//     .Input("y: float")
//     .Output("z: float");

class GraphTest : public ::testing::Test {
 protected:
  GraphTest() : graph_(OpRegistry::Global()) {}
  ~GraphTest() override {}

  // static void VerifyNodes(Node* node, const std::vector<Node*>& expected_in,
  //                         const std::vector<Node*>& expected_out) {
  //   std::vector<Node*> in;
  //   for (const Edge* e : node->in_edges()) {
  //     in.push_back(e->src());
  //   }
  //   EXPECT_EQ(Stringify(expected_in), Stringify(in));

  //   std::vector<Node*> out;
  //   for (const Edge* e : node->out_edges()) {
  //     out.push_back(e->dst());
  //   }
  //   EXPECT_EQ(Stringify(expected_out), Stringify(out));
  // }

  // void VerifyGraphStats() {
  //   int nodes = 0;
  //   for (const Node* n : graph_.nodes()) {
  //     VLOG(1) << n->id();
  //     ++nodes;
  //   }
  //   EXPECT_EQ(nodes, graph_.num_nodes());
  //   int edges = 0;
  //   for (const Edge* e : graph_.edges()) {
  //     VLOG(1) << e->id();
  //     ++edges;
  //   }
  //   EXPECT_EQ(edges, graph_.num_edges());
  // }

  Graph graph_;

 private:
  // Convert a list of nodes to a sorted list of strings so failure messages
  // are readable.
  static std::vector<string> Stringify(const std::vector<Node*>& nodes) {
    std::vector<string> result;
    result.reserve(nodes.size());
    for (Node* n : nodes) {
      result.push_back(n->DebugString());
    }
    std::sort(result.begin(), result.end());
    return result;
  }
};

TEST_F(GraphTest, Constructor) {
  Node* source = graph_.source_node();
  EXPECT_NE(source, nullptr);

  Node* sink = graph_.sink_node();
  EXPECT_NE(sink, nullptr);
  
  // VerifyNodes(source, {}, {sink});
  // VerifyNodes(sink, {source}, {});
  EXPECT_EQ(2, graph_.num_node_ids());
  // VerifyGraphStats();
}

}  // namespace
}  // namespace tensorflow
