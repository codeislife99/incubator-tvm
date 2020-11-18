/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 *
 * \file get_compute_heavy_graph.cc
 * \brief Pass to determine if a graph is compute heavy. Currently graphs which contain
 * convolutions(and their transpose), dense and batch matmul are categorized as compute_heavy
 */

#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op.h>
#include <tvm/relay/op_attr_types.h>

#include "../transforms/pattern_util.h"

namespace tvm {
namespace relay {

class ComputeHeavyGraphs : private ExprVisitor {
 public:
  ComputeHeavyGraphs() { is_compute_heavy_ = false; }
  static int64_t IsComputeHeavyGraph(const Expr& expr) {
    LOG(INFO) << "This pass only counts conv2d, conv2d_transpose, dense and batchmatmul as compute "
                 "heavy graphs";
    ComputeHeavyGraphs compute_heavy_graphs;
    compute_heavy_graphs(expr);
    return compute_heavy_graphs.is_compute_heavy_;
  }

 private:
  void VisitExpr_(const CallNode* call_node) final {
    std::set<std::string> heavy_ops{"nn.conv2d", "nn.conv2d_transpose",
                                    "nn.conv3d", "nn.conv3d_transpose",
                                    "nn.dense",  "nn.batch_matmul"};
    if (heavy_ops.count(call_node->op.as<OpNode>()->name)) {
      is_compute_heavy_ = true;
    }
    ExprVisitor::VisitExpr_(call_node);
  }
  bool is_compute_heavy_;
};

bool IsComputeHeavyGraph(const Expr& expr) { return ComputeHeavyGraphs::IsComputeHeavyGraph(expr); }

TVM_REGISTER_GLOBAL("relay.analysis.IsComputeHeavyGraph").set_body_typed(IsComputeHeavyGraph);

}  // namespace relay
}  // namespace tvm
