/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include <algorithm>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/executor.h"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/common_runtime/step_stats_collector.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/rendezvous.h"
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

class ExecutorTest : public ::testing::Test {
 protected:
  ExecutorTest()
      : device_(DeviceFactory::NewDevice("CPU", {},
                                         "/job:localhost/replica:0/task:0")),

        step_stats_collector_(&step_stats_) {
    SessionOptions options;
    thread_pool_ = ComputePool(options);
  }

  ~ExecutorTest() override {
    // There should always be exactly one Ref left on the Rendezvous
    // when the test completes.
    CHECK(rendez_->Unref());
    delete exec_;
  }

  // Resets executor_ with a new executor based on a graph 'gdef'.
  void Create(std::unique_ptr<const Graph> graph) {
    const int version = graph->versions().producer();
    LocalExecutorParams params;
    params.device = device_.get();
    params.create_kernel = [this, version](const NodeDef& ndef,
                                           OpKernel** kernel) {
      return CreateNonCachedKernel(device_.get(), nullptr, ndef, version,
                                   kernel);
    };
    params.delete_kernel = [](OpKernel* kernel) {
      DeleteNonCachedKernel(kernel);
    };
    delete exec_;
    TF_CHECK_OK(NewLocalExecutor(params, std::move(graph), &exec_));
    runner_ = [this](std::function<void()> fn) { thread_pool_->Schedule(fn); };
    rendez_ = NewLocalRendezvous();
  }

  Status Run(Rendezvous* rendez) {
    Executor::Args args;
    args.rendezvous = rendez;
    args.stats_collector = &step_stats_collector_;
    args.runner = runner_;
    return exec_->Run(args);
  }

  thread::ThreadPool* thread_pool_ = nullptr;
  std::unique_ptr<Device> device_;
  Executor* exec_ = nullptr;
  StepStatsCollector step_stats_collector_;
  StepStats step_stats_;
  Executor::Args::Runner runner_;
  Rendezvous* rendez_ = nullptr;
};

// A float val -> Tensor<float>
Tensor V(const float val) {
  Tensor tensor(DT_FLOAT, TensorShape({}));
  tensor.scalar<float>()() = val;
  return tensor;
}

// A int32 val -> Tensor<int32>
Tensor VI(const int32 val) {
  Tensor tensor(DT_INT32, TensorShape({}));
  tensor.scalar<int32>()() = val;
  return tensor;
}

// A bool val -> Tensor<bool>
Tensor VB(const bool val) {
  Tensor tensor(DT_BOOL, TensorShape({}));
  tensor.scalar<bool>()() = val;
  return tensor;
}

// A double val -> Tensor<double>
Tensor VD(const double val) {
  Tensor tensor(DT_DOUBLE, TensorShape({}));
  tensor.scalar<double>()() = val;
  return tensor;
}

// Tensor<float> -> a float val.
float V(const Tensor& tensor) {
  CHECK_EQ(tensor.dtype(), DT_FLOAT);
  CHECK(TensorShapeUtils::IsScalar(tensor.shape()));
  return tensor.scalar<float>()();
}

static uint64 kIncarnation = 1;  // Uses in following tests.

Rendezvous::ParsedKey Key(const string& sender, const uint64 incarnation,
                          const string& receiver, const string& name) {
  Rendezvous::ParsedKey result;
  CHECK(
      Rendezvous::ParseKey(Rendezvous::CreateKey(sender, incarnation, receiver,
                                                 name, FrameAndIter(0, 0)),
                           &result)
          .ok());
  return result;
}

#define ALICE "/job:j/replica:0/task:0/cpu:0"
#define BOB "/job:j/replica:0/task:0/device:GPU:0"

TEST_F(ExecutorTest, SimpleAdd) {
  // Create graph: c = a + b
  std::unique_ptr<Graph> g(new Graph(OpRegistry::Global()));
  auto in0 = test::graph::Recv(g.get(), "a", "float", ALICE, 1, BOB);
  auto in1 = test::graph::Recv(g.get(), "b", "float", ALICE, 1, BOB);
  auto tmp = test::graph::Add(g.get(), in0, in1);
  test::graph::Send(g.get(), tmp, "c", BOB, 1, ALICE);
  Create(std::move(g));
  // Creat Rendezvous
  Rendezvous::Args args;
  TF_ASSERT_OK(rendez_->Send(Key(ALICE, kIncarnation, BOB, "a"), args, V(1.0),
                             false));  // in0 = 1.0
  TF_ASSERT_OK(rendez_->Send(Key(ALICE, kIncarnation, BOB, "b"), args, V(1.0),
                             false));  // in1 = 1.0
  TF_ASSERT_OK(Run(rendez_));
  
  Tensor out = V(-1);
  bool is_dead = false;
  TF_ASSERT_OK(
      rendez_->Recv(Key(BOB, kIncarnation, ALICE, "c"), args, &out, &is_dead));
  EXPECT_EQ(2.0, V(out));  // out = 1.0 + 1.0 = 2.0
}

}  // namespace tensorflow
