#include "torch/csrc/autograd/VariableTypeUtils.h"

// @generated from tools/autograd/templates/VariableType_4.cpp

// NOTE [Sharded File]: on this file's split-into-shards state
//
// Back in the good old days, VariableType.cpp was generated as one
// file with every function in it, and everything was great and
// simple.
//
// However, this file was also very large (over 36,000 lines), and
// compiling it was very slow, and in fact was a significant
// bottleneck for incremental rebuilds. To address this, we now
// generate the file split across multiple shards, named
// VariableType_0.cpp and so on, which can be compiled in parallel.
//
// For ease of inspection and debugging, so that it's not necessary to
// go rooting around in multiple files, we also generate all the
// functions together in VariableTypeEverything.cpp. This generated
// file is only for convenience; it's not actually used in the
// build. If the file you're looking at now is one of the shards, you
// may want to switch over to the Everything variant to make you
// grepping smoother.

using namespace at;
using namespace torch::autograd::generated;

namespace torch { namespace autograd {

Tensor & VariableType::__lshift___out(Tensor & result, const Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("__lshift___out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::__lshift__");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("__lshift___out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->__lshift___out(result_, self_, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::s___lshift___out(Tensor & result, const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("__lshift___out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& other_ = unpack(other, "other", 2);
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::__lshift__");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("__lshift___out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->s___lshift___out(result_, self_, other_);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::__or___out(Tensor & result, const Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("__or___out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::__or__");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("__or___out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->__or___out(result_, self_, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::s___or___out(Tensor & result, const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("__or___out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& other_ = unpack(other, "other", 2);
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::__or__");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("__or___out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->s___or___out(result_, self_, other_);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::__xor___out(Tensor & result, const Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("__xor___out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::__xor__");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("__xor___out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->__xor___out(result_, self_, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::s___xor___out(Tensor & result, const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("__xor___out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& other_ = unpack(other, "other", 2);
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::__xor__");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("__xor___out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->s___xor___out(result_, self_, other_);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_abs_out(Tensor & result, const Tensor & self) const {
  profiler::RecordFunction profiler("_abs_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_abs");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("_abs");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_abs");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("_abs_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_abs_out(result_, self_);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::s__addmv(const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) const {
  profiler::RecordFunction profiler("_addmv", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& mat_ = unpack(mat, "mat", 1);
  auto& vec_ = unpack(vec, "vec", 2);
  std::shared_ptr<AddmvBackward> grad_fn;
  if (compute_requires_grad( self, mat, vec )) {
    grad_fn = std::shared_ptr<AddmvBackward>(new AddmvBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, mat, vec ));
    grad_fn->vec_ = SavedVariable(vec, false);
    grad_fn->alpha = alpha;
    grad_fn->beta = beta;
    grad_fn->mat_ = SavedVariable(mat, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_addmv");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mat", mat);
    jit::tracer::addInputs(node, "vec", vec);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->s__addmv(self_, mat_, vec_, beta, alpha));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_addmv_(Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) const {
  profiler::RecordFunction profiler("_addmv_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& mat_ = unpack(mat, "mat", 1);
  auto& vec_ = unpack(vec, "vec", 2);
  check_inplace(self);
  std::shared_ptr<AddmvBackward> grad_fn;
  if (compute_requires_grad( self, mat, vec )) {
    grad_fn = std::shared_ptr<AddmvBackward>(new AddmvBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, mat, vec ));
    grad_fn->vec_ = SavedVariable(vec, false);
    grad_fn->alpha = alpha;
    grad_fn->beta = beta;
    grad_fn->mat_ = SavedVariable(mat, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_addmv");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mat", mat);
    jit::tracer::addInputs(node, "vec", vec);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("_addmv_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_addmv_(self_, mat_, vec_, beta, alpha);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::_atan_out(Tensor & result, const Tensor & self) const {
  profiler::RecordFunction profiler("_atan_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_atan");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("_atan");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_atan");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("_atan_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_atan_out(result_, self_);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_cast_Char(const Tensor & self, bool non_blocking) const {
  profiler::RecordFunction profiler("_cast_Char", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_cast_Char");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "non_blocking", non_blocking);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::_cast_Char(self, non_blocking);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_cast_Float(const Tensor & self, bool non_blocking) const {
  profiler::RecordFunction profiler("_cast_Float", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_cast_Float");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "non_blocking", non_blocking);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::_cast_Float(self, non_blocking);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_convolution_nogroup(const Tensor & input, const Tensor & weight, const Tensor & bias, IntList stride, IntList padding, IntList dilation, bool transposed, IntList output_padding) const {
  profiler::RecordFunction profiler("_convolution_nogroup", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_convolution_nogroup");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "transposed", transposed);
    jit::tracer::addInputs(node, "output_padding", output_padding);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::_convolution_nogroup(input, weight, bias, stride, padding, dilation, transposed, output_padding);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_cos_out(Tensor & result, const Tensor & self) const {
  profiler::RecordFunction profiler("_cos_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_cos");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("_cos");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_cos");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("_cos_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_cos_out(result_, self_);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_cosh(const Tensor & self) const {
  profiler::RecordFunction profiler("_cosh", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_cosh"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_cosh");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_cosh(self_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_ctc_loss_backward(const Tensor & grad, const Tensor & log_probs, const Tensor & targets, IntList input_lengths, IntList target_lengths, const Tensor & neg_log_likelihood, const Tensor & log_alpha, int64_t blank) const {
  profiler::RecordFunction profiler("_ctc_loss_backward", Function::peek_at_next_sequence_nr());
  auto& grad_ = unpack(grad, "grad", 0);
  auto& log_probs_ = unpack(log_probs, "log_probs", 1);
  auto& targets_ = unpack(targets, "targets", 2);
  auto& neg_log_likelihood_ = unpack(neg_log_likelihood, "neg_log_likelihood", 5);
  auto& log_alpha_ = unpack(log_alpha, "log_alpha", 6);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad, log_probs, targets, neg_log_likelihood, log_alpha )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_ctc_loss_backward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad, log_probs, targets, neg_log_likelihood, log_alpha ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_ctc_loss_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad", grad);
    jit::tracer::addInputs(node, "log_probs", log_probs);
    jit::tracer::addInputs(node, "targets", targets);
    jit::tracer::addInputs(node, "input_lengths", input_lengths);
    jit::tracer::addInputs(node, "target_lengths", target_lengths);
    jit::tracer::addInputs(node, "neg_log_likelihood", neg_log_likelihood);
    jit::tracer::addInputs(node, "log_alpha", log_alpha);
    jit::tracer::addInputs(node, "blank", blank);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_ctc_loss_backward(grad_, log_probs_, targets_, input_lengths, target_lengths, neg_log_likelihood_, log_alpha_, blank));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor,Tensor> VariableType::_cudnn_ctc_loss(const Tensor & log_probs, const Tensor & targets, IntList input_lengths, IntList target_lengths, int64_t blank, bool deterministic) const {
  profiler::RecordFunction profiler("_cudnn_ctc_loss", Function::peek_at_next_sequence_nr());
  auto& log_probs_ = unpack(log_probs, "log_probs", 0);
  auto& targets_ = unpack(targets, "targets", 1);
  check_no_requires_grad(targets, "targets");
  std::shared_ptr<CudnnCtcLossBackward> grad_fn;
  if (compute_requires_grad( log_probs )) {
    grad_fn = std::shared_ptr<CudnnCtcLossBackward>(new CudnnCtcLossBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( log_probs ));
  }
  Tensor result0;
  Tensor result1;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_cudnn_ctc_loss");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "log_probs", log_probs);
    jit::tracer::addInputs(node, "targets", targets);
    jit::tracer::addInputs(node, "input_lengths", input_lengths);
    jit::tracer::addInputs(node, "target_lengths", target_lengths);
    jit::tracer::addInputs(node, "blank", blank);
    jit::tracer::addInputs(node, "deterministic", deterministic);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(result0, result1) = as_variable(baseType->_cudnn_ctc_loss(log_probs_, targets_, input_lengths, target_lengths, blank, deterministic));
  set_history(flatten_tensor_args( result0, result1 ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  if (grad_fn) {
    grad_fn->result1_ = SavedVariable(result1, true);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
void VariableType::_cufft_set_plan_cache_max_size(int64_t max_size) const {
  profiler::RecordFunction profiler("_cufft_set_plan_cache_max_size", Function::peek_at_next_sequence_nr());
  TypeDefault::_cufft_set_plan_cache_max_size(max_size);
}
std::tuple<Tensor,Tensor> VariableType::_gesv_single(const Tensor & self, const Tensor & A) const {
  profiler::RecordFunction profiler("_gesv_single", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& A_ = unpack(A, "A", 1);
  std::shared_ptr<GesvSingleBackward> grad_fn;
  if (compute_requires_grad( self, A )) {
    grad_fn = std::shared_ptr<GesvSingleBackward>(new GesvSingleBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, A ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->A_ = SavedVariable(A, false);
  }
  Tensor solution;
  Tensor lu;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_gesv_single");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "A", A);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(solution, lu) = as_variable(baseType->_gesv_single(self_, A_));
  set_history(flatten_tensor_args( solution ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, solution);
    jit::tracer::addOutput(node, lu);
  }
  if (grad_fn) {
    grad_fn->solution_ = SavedVariable(solution, true);
  }
  return std::make_tuple(std::move(solution), std::move(lu));
}
Tensor VariableType::_getri(const Tensor & self) const {
  profiler::RecordFunction profiler("_getri", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_getri"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_getri");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto output = as_variable(baseType->_getri(self_));
  set_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor VariableType::_log10(const Tensor & self) const {
  profiler::RecordFunction profiler("_log10", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_log10"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_log10");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_log10(self_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_masked_scale(const Tensor & self, const Tensor & mask, double scale) const {
  profiler::RecordFunction profiler("_masked_scale", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& mask_ = unpack(mask, "mask", 1);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, mask )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_masked_scale"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, mask ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_masked_scale");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mask", mask);
    jit::tracer::addInputs(node, "scale", scale);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_masked_scale(self_, mask_, scale));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_pdist_backward(const Tensor & grad, const Tensor & self, double p, const Tensor & pdist) const {
  profiler::RecordFunction profiler("_pdist_backward", Function::peek_at_next_sequence_nr());
  auto& grad_ = unpack(grad, "grad", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& pdist_ = unpack(pdist, "pdist", 3);
  std::shared_ptr<PdistBackwardBackward> grad_fn;
  if (compute_requires_grad( grad, self, pdist )) {
    grad_fn = std::shared_ptr<PdistBackwardBackward>(new PdistBackwardBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad, self, pdist ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_pdist_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad", grad);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "pdist", pdist);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_pdist_backward(grad_, self_, p, pdist_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_range_out(Tensor & result, Scalar start, Scalar end, Scalar step) const {
  profiler::RecordFunction profiler("_range_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_range");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "start", start);
    jit::tracer::addInputs(node, "end", end);
    jit::tracer::addInputs(node, "step", step);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("_range_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_range_out(result_, start, end, step);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_s_where(const Tensor & condition, const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("_s_where", Function::peek_at_next_sequence_nr());
  auto& condition_ = unpack(condition, "condition", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& other_ = unpack(other, "other", 2);
  std::shared_ptr<SWhereBackward> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::shared_ptr<SWhereBackward>(new SWhereBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    grad_fn->condition_ = SavedVariable(condition, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_s_where");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "condition", condition);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_s_where(condition_, self_, other_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_sigmoid_backward(const Tensor & grad_output, const Tensor & output) const {
  profiler::RecordFunction profiler("_sigmoid_backward", Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& output_ = unpack(output, "output", 1);
  std::shared_ptr<SigmoidBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, output )) {
    grad_fn = std::shared_ptr<SigmoidBackwardBackward>(new SigmoidBackwardBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, output ));
    grad_fn->output_ = SavedVariable(output, false);
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_sigmoid_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "output", output);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto grad_input = as_variable(baseType->_sigmoid_backward(grad_output_, output_));
  set_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor & VariableType::_sigmoid_forward_out(Tensor & output, const Tensor & self) const {
  profiler::RecordFunction profiler("_sigmoid_forward_out", Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_sigmoid_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("_sigmoid_forward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_sigmoid_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("_sigmoid_forward_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_sigmoid_forward_out(output_, self_);
  increment_version(output);
  rebase_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor & VariableType::_sin_out(Tensor & result, const Tensor & self) const {
  profiler::RecordFunction profiler("_sin_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_sin");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("_sin");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_sin");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("_sin_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_sin_out(result_, self_);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_sinh(const Tensor & self) const {
  profiler::RecordFunction profiler("_sinh", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_sinh"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_sinh");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_sinh(self_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_sparse_coo_tensor_with_dims_and_tensors(int64_t sparse_dim, int64_t dense_dim, IntList size, const Tensor & indices, const Tensor & values, const TensorOptions & options) const {
  profiler::RecordFunction profiler("_sparse_coo_tensor_with_dims_and_tensors", Function::peek_at_next_sequence_nr());
  auto& indices_ = unpack(indices, "indices", 3);
  auto& values_ = unpack(values, "values", 4);
  auto options_ = TensorOptions(options).is_variable(false);
  check_no_requires_grad(indices, "indices");
  std::shared_ptr<SparseCooTensorWithDimsAndTensorsBackward> grad_fn;
  if (compute_requires_grad( values )) {
    grad_fn = std::shared_ptr<SparseCooTensorWithDimsAndTensorsBackward>(new SparseCooTensorWithDimsAndTensorsBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( values ));
    grad_fn->indices_ = SavedVariable(indices, false);
    grad_fn->values_sizes = values.sizes().vec();
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_sparse_coo_tensor_with_dims_and_tensors");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "sparse_dim", sparse_dim);
    jit::tracer::addInputs(node, "dense_dim", dense_dim);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "values", values);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_sparse_coo_tensor_with_dims_and_tensors(sparse_dim, dense_dim, size, indices_, values_, options_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_sparse_mul_zerodim_out(Tensor & result, const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("_sparse_mul_zerodim_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& other_ = unpack(other, "other", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, other )) {
    throw_error_out_requires_grad("_sparse_mul_zerodim");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("_sparse_mul_zerodim");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_sparse_mul_zerodim");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("_sparse_mul_zerodim_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_sparse_mul_zerodim_out(result_, self_, other_);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_standard_gamma(const Tensor & self, Generator * generator) const {
  profiler::RecordFunction profiler("_standard_gamma", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<StandardGammaBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<StandardGammaBackward>(new StandardGammaBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_standard_gamma");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_standard_gamma(self_, generator));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  return result;
}
Tensor VariableType::_tanh_forward(const Tensor & self) const {
  profiler::RecordFunction profiler("_tanh_forward", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_tanh_forward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_tanh_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto output = as_variable(baseType->_tanh_forward(self_));
  set_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor VariableType::s__th_addcdiv(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
  profiler::RecordFunction profiler("_th_addcdiv", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& tensor1_ = unpack(tensor1, "tensor1", 1);
  auto& tensor2_ = unpack(tensor2, "tensor2", 2);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, tensor1, tensor2 )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_addcdiv"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, tensor1, tensor2 ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_th_addcdiv");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "tensor1", tensor1);
    jit::tracer::addInputs(node, "tensor2", tensor2);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->s__th_addcdiv(self_, tensor1_, tensor2_, value));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::s__th_addcdiv_(Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
  profiler::RecordFunction profiler("_th_addcdiv_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& tensor1_ = unpack(tensor1, "tensor1", 1);
  auto& tensor2_ = unpack(tensor2, "tensor2", 2);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, tensor1, tensor2 )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_addcdiv_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, tensor1, tensor2 ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_th_addcdiv");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "tensor1", tensor1);
    jit::tracer::addInputs(node, "tensor2", tensor2);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("_th_addcdiv_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->s__th_addcdiv_(self_, tensor1_, tensor2_, value);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::s__th_addcmul_out(Tensor & result, const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
  profiler::RecordFunction profiler("_th_addcmul_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& tensor1_ = unpack(tensor1, "tensor1", 2);
  auto& tensor2_ = unpack(tensor2, "tensor2", 3);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, tensor1, tensor2 )) {
    throw_error_out_requires_grad("_th_addcmul");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("_th_addcmul");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_th_addcmul");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "tensor1", tensor1);
    jit::tracer::addInputs(node, "tensor2", tensor2);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("_th_addcmul_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->s__th_addcmul_out(result_, self_, tensor1_, tensor2_, value);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::s__th_atan2_out(Tensor & result, const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("_th_atan2_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& other_ = unpack(other, "other", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, other )) {
    throw_error_out_requires_grad("_th_atan2");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("_th_atan2");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_th_atan2");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("_th_atan2_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->s__th_atan2_out(result_, self_, other_);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor,Tensor,Tensor> VariableType::_th_btrifact_with_info(const Tensor & self, bool pivot) const {
  profiler::RecordFunction profiler("_th_btrifact_with_info", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_btrifact_with_info"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  Tensor result;
  Tensor pivots;
  Tensor info;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_th_btrifact_with_info");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "pivot", pivot);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(result, pivots, info) = as_variable(baseType->_th_btrifact_with_info(self_, pivot));
  set_history(flatten_tensor_args( result, pivots, info ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
    jit::tracer::addOutput(node, pivots);
    jit::tracer::addOutput(node, info);
  }
  return std::make_tuple(std::move(result), std::move(pivots), std::move(info));
}
Tensor VariableType::_th_clamp_min(const Tensor & self, Scalar min) const {
  profiler::RecordFunction profiler("_th_clamp_min", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_clamp_min"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_th_clamp_min");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "min", min);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_th_clamp_min(self_, min));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_th_clamp_out(Tensor & result, const Tensor & self, Scalar min, Scalar max) const {
  profiler::RecordFunction profiler("_th_clamp_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_th_clamp");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("_th_clamp");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_th_clamp");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "min", min);
    jit::tracer::addInputs(node, "max", max);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("_th_clamp_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_clamp_out(result_, self_, min, max);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_th_clone(const Tensor & self) const {
  profiler::RecordFunction profiler("_th_clone", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_clone"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_th_clone");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_th_clone(self_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_th_cross(const Tensor & self, const Tensor & other, int64_t dim) const {
  profiler::RecordFunction profiler("_th_cross", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_cross"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_th_cross");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_th_cross(self_, other_, dim));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_th_diag_out(Tensor & result, const Tensor & self, int64_t diagonal) const {
  profiler::RecordFunction profiler("_th_diag_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_th_diag");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("_th_diag");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_th_diag");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "diagonal", diagonal);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("_th_diag_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_diag_out(result_, self_, diagonal);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_th_digamma_out(Tensor & result, const Tensor & self) const {
  profiler::RecordFunction profiler("_th_digamma_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_th_digamma");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("_th_digamma");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_th_digamma");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("_th_digamma_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_digamma_out(result_, self_);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor,Tensor> VariableType::_th_eig(const Tensor & self, bool eigenvectors) const {
  profiler::RecordFunction profiler("_th_eig", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_eig"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  Tensor res1;
  Tensor res2;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_th_eig");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "eigenvectors", eigenvectors);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(res1, res2) = as_variable(baseType->_th_eig(self_, eigenvectors));
  set_history(flatten_tensor_args( res1, res2 ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, res1);
    jit::tracer::addOutput(node, res2);
  }
  return std::make_tuple(std::move(res1), std::move(res2));
}
Tensor VariableType::_th_eq(const Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("_th_eq", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_th_eq");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_th_eq(self_, other));
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::s__th_eq(const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("_th_eq", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_th_eq");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->s__th_eq(self_, other_));
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_th_eq_(Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("_th_eq_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_eq_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_th_eq");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("_th_eq_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_eq_(self_, other);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::s__th_eq_(Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("_th_eq_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_eq_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_th_eq");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("_th_eq_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->s__th_eq_(self_, other_);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::_th_erfinv_out(Tensor & result, const Tensor & self) const {
  profiler::RecordFunction profiler("_th_erfinv_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_th_erfinv");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("_th_erfinv");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_th_erfinv");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("_th_erfinv_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_erfinv_out(result_, self_);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_th_gather_out(Tensor & result, const Tensor & self, int64_t dim, const Tensor & index) const {
  profiler::RecordFunction profiler("_th_gather_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& index_ = unpack(index, "index", 3);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_th_gather");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("_th_gather");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_th_gather");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("_th_gather_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_gather_out(result_, self_, dim, index_);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_th_ge(const Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("_th_ge", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_th_ge");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_th_ge(self_, other));
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::s__th_ge(const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("_th_ge", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_th_ge");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->s__th_ge(self_, other_));
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_th_ge_(Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("_th_ge_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_ge_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_th_ge");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("_th_ge_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_ge_(self_, other);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::s__th_ge_(Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("_th_ge_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_ge_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_th_ge");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("_th_ge_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->s__th_ge_(self_, other_);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::_th_geometric_(Tensor & self, double p, Generator * generator) const {
  profiler::RecordFunction profiler("_th_geometric_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_geometric_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_th_geometric");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("_th_geometric_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_geometric_(self_, p, generator);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor VariableType::_th_gt(const Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("_th_gt", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_th_gt");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_th_gt(self_, other));
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::s__th_gt(const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("_th_gt", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_th_gt");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->s__th_gt(self_, other_));
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_th_gt_(Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("_th_gt_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_gt_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_th_gt");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("_th_gt_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_gt_(self_, other);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::s__th_gt_(Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("_th_gt_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_gt_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_th_gt");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("_th_gt_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->s__th_gt_(self_, other_);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor VariableType::_th_histc(const Tensor & self, int64_t bins, Scalar min, Scalar max) const {
  profiler::RecordFunction profiler("_th_histc", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_histc"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_th_histc");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "bins", bins);
    jit::tracer::addInputs(node, "min", min);
    jit::tracer::addInputs(node, "max", max);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_th_histc(self_, bins, min, max));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_th_index_fill_(Tensor & self, int64_t dim, const Tensor & index, Scalar value) const {
  profiler::RecordFunction profiler("_th_index_fill_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& index_ = unpack(index, "index", 2);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_index_fill_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_th_index_fill");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("_th_index_fill_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_index_fill_(self_, dim, index_, value);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::_th_index_fill_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & value) const {
  profiler::RecordFunction profiler("_th_index_fill_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& index_ = unpack(index, "index", 2);
  auto& value_ = unpack(value, "value", 3);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, value )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_index_fill_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, value ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_th_index_fill");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("_th_index_fill_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_index_fill_(self_, dim, index_, value_);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
bool VariableType::_th_is_contiguous(const Tensor & self) const {
  profiler::RecordFunction profiler("_th_is_contiguous", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto result = baseType->_th_is_contiguous(self_);
  return result;
}
bool VariableType::_th_is_set_to(const Tensor & self, const Tensor & tensor) const {
  profiler::RecordFunction profiler("_th_is_set_to", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& tensor_ = unpack(tensor, "tensor", 1);
  auto result = baseType->_th_is_set_to(self_, tensor_);
  return result;
}
Tensor VariableType::_th_le(const Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("_th_le", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_th_le");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_th_le(self_, other));
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::s__th_le(const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("_th_le", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_th_le");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->s__th_le(self_, other_));
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_th_le_(Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("_th_le_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_le_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_th_le");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("_th_le_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_le_(self_, other);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::s__th_le_(Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("_th_le_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_le_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_th_le");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("_th_le_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->s__th_le_(self_, other_);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::s__th_lerp_out(Tensor & result, const Tensor & self, const Tensor & end, Scalar weight) const {
  profiler::RecordFunction profiler("_th_lerp_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& end_ = unpack(end, "end", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, end )) {
    throw_error_out_requires_grad("_th_lerp");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("_th_lerp");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_th_lerp");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "end", end);
    jit::tracer::addInputs(node, "weight", weight);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("_th_lerp_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->s__th_lerp_out(result_, self_, end_, weight);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_th_lt(const Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("_th_lt", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_th_lt");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_th_lt(self_, other));
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::s__th_lt(const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("_th_lt", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_th_lt");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->s__th_lt(self_, other_));
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_th_lt_(Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("_th_lt_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_lt_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_th_lt");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("_th_lt_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_lt_(self_, other);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::s__th_lt_(Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("_th_lt_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_lt_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_th_lt");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("_th_lt_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->s__th_lt_(self_, other_);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor VariableType::s__th_masked_select(const Tensor & self, const Tensor & mask) const {
  profiler::RecordFunction profiler("_th_masked_select", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& mask_ = unpack(mask, "mask", 1);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_masked_select"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_th_masked_select");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mask", mask);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->s__th_masked_select(self_, mask_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::s__th_min(const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("_th_min", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_min"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_th_min");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->s__th_min(self_, other_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_th_min(const Tensor & self) const {
  profiler::RecordFunction profiler("_th_min", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_min"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_th_min");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_th_min(self_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor,Tensor> VariableType::_th_min(const Tensor & self, int64_t dim, bool keepdim) const {
  profiler::RecordFunction profiler("_th_min", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_min"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  Tensor min;
  Tensor min_indices;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_th_min");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(min, min_indices) = as_variable(baseType->_th_min(self_, dim, keepdim));
  set_history(flatten_tensor_args( min ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, min);
    jit::tracer::addOutput(node, min_indices);
  }
  return std::make_tuple(std::move(min), std::move(min_indices));
}
Tensor & VariableType::_th_multinomial_out(Tensor & result, const Tensor & self, int64_t num_samples, bool replacement, Generator * generator) const {
  profiler::RecordFunction profiler("_th_multinomial_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_th_multinomial");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "num_samples", num_samples);
    jit::tracer::addInputs(node, "replacement", replacement);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("_th_multinomial_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_multinomial_out(result_, self_, num_samples, replacement, generator);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_th_neg(const Tensor & self) const {
  profiler::RecordFunction profiler("_th_neg", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_neg"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_th_neg");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_th_neg(self_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_th_neg_(Tensor & self) const {
  profiler::RecordFunction profiler("_th_neg_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_neg_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_th_neg");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("_th_neg_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_neg_(self_);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor VariableType::_th_nonzero(const Tensor & self) const {
  profiler::RecordFunction profiler("_th_nonzero", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_th_nonzero");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_th_nonzero(self_));
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_th_norm(const Tensor & self, Scalar p) const {
  profiler::RecordFunction profiler("_th_norm", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_norm"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_th_norm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_th_norm(self_, p));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_th_norm(const Tensor & self, Scalar p, int64_t dim, bool keepdim) const {
  profiler::RecordFunction profiler("_th_norm", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_norm"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_th_norm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_th_norm(self_, p, dim, keepdim));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_th_normal(const Tensor & mean, double std, Generator * generator) const {
  profiler::RecordFunction profiler("_th_normal", Function::peek_at_next_sequence_nr());
  auto& mean_ = unpack(mean, "mean", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( mean )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_normal"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( mean ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_th_normal");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "std", std);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto output = as_variable(baseType->_th_normal(mean_, std, generator));
  set_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor VariableType::_th_normal(double mean, const Tensor & std, Generator * generator) const {
  profiler::RecordFunction profiler("_th_normal", Function::peek_at_next_sequence_nr());
  auto& std_ = unpack(std, "std", 1);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( std )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_normal"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( std ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_th_normal");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "std", std);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto output = as_variable(baseType->_th_normal(mean, std_, generator));
  set_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor VariableType::_th_normal(const Tensor & mean, const Tensor & std, Generator * generator) const {
  profiler::RecordFunction profiler("_th_normal", Function::peek_at_next_sequence_nr());
  auto& mean_ = unpack(mean, "mean", 0);
  auto& std_ = unpack(std, "std", 1);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( mean, std )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_normal"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( mean, std ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_th_normal");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "std", std);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto output = as_variable(baseType->_th_normal(mean_, std_, generator));
  set_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor & VariableType::_th_normal_(Tensor & self, double mean, double std, Generator * generator) const {
  profiler::RecordFunction profiler("_th_normal_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_normal_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_th_normal");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "std", std);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("_th_normal_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_normal_(self_, mean, std, generator);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::_th_orgqr_out(Tensor & result, const Tensor & self, const Tensor & input2) const {
  profiler::RecordFunction profiler("_th_orgqr_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& input2_ = unpack(input2, "input2", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, input2 )) {
    throw_error_out_requires_grad("_th_orgqr");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("_th_orgqr");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_th_orgqr");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "input2", input2);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("_th_orgqr_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_orgqr_out(result_, self_, input2_);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_th_potrf_out(Tensor & output, const Tensor & self, bool upper) const {
  profiler::RecordFunction profiler("_th_potrf_out", Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_th_potrf");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("_th_potrf");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_th_potrf");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "upper", upper);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("_th_potrf_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_potrf_out(output_, self_, upper);
  increment_version(output);
  rebase_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
std::tuple<Tensor,Tensor> VariableType::_th_pstrf(const Tensor & self, bool upper, Scalar tol) const {
  profiler::RecordFunction profiler("_th_pstrf", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_pstrf"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  Tensor res1;
  Tensor res2;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_th_pstrf");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "upper", upper);
    jit::tracer::addInputs(node, "tol", tol);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(res1, res2) = as_variable(baseType->_th_pstrf(self_, upper, tol));
  set_history(flatten_tensor_args( res1, res2 ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, res1);
    jit::tracer::addOutput(node, res2);
  }
  return std::make_tuple(std::move(res1), std::move(res2));
}
Tensor & VariableType::_th_reciprocal_out(Tensor & result, const Tensor & self) const {
  profiler::RecordFunction profiler("_th_reciprocal_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_th_reciprocal");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("_th_reciprocal");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_th_reciprocal");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("_th_reciprocal_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_reciprocal_out(result_, self_);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_th_renorm(const Tensor & self, Scalar p, int64_t dim, Scalar maxnorm) const {
  profiler::RecordFunction profiler("_th_renorm", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_renorm"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_th_renorm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "maxnorm", maxnorm);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_th_renorm(self_, p, dim, maxnorm));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_th_renorm_(Tensor & self, Scalar p, int64_t dim, Scalar maxnorm) const {
  profiler::RecordFunction profiler("_th_renorm_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_renorm_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_th_renorm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "maxnorm", maxnorm);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("_th_renorm_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_renorm_(self_, p, dim, maxnorm);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
int64_t VariableType::_th_storage_offset(const Tensor & self) const {
  profiler::RecordFunction profiler("_th_storage_offset", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto result = baseType->_th_storage_offset(self_);
  return result;
}
std::tuple<Tensor,Tensor> VariableType::_th_symeig(const Tensor & self, bool eigenvectors, bool upper) const {
  profiler::RecordFunction profiler("_th_symeig", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_symeig"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  Tensor res1;
  Tensor res2;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_th_symeig");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "eigenvectors", eigenvectors);
    jit::tracer::addInputs(node, "upper", upper);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(res1, res2) = as_variable(baseType->_th_symeig(self_, eigenvectors, upper));
  set_history(flatten_tensor_args( res1, res2 ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, res1);
    jit::tracer::addOutput(node, res2);
  }
  return std::make_tuple(std::move(res1), std::move(res2));
}
std::tuple<Tensor &,Tensor &> VariableType::_th_trtrs_out(Tensor & res1, Tensor & res2, const Tensor & self, const Tensor & A, bool upper, bool transpose, bool unitriangular) const {
  profiler::RecordFunction profiler("_th_trtrs_out", Function::peek_at_next_sequence_nr());
  auto& res1_ = unpack(res1, "res1", 0);
  auto& res2_ = unpack(res2, "res2", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& A_ = unpack(A, "A", 3);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, A )) {
    throw_error_out_requires_grad("_th_trtrs");
  }
  if (compute_requires_grad( res1, res2 )) {
    throw_error_out_requires_grad("_th_trtrs");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_th_trtrs");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "res2", res2);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "A", A);
    jit::tracer::addInputs(node, "upper", upper);
    jit::tracer::addInputs(node, "transpose", transpose);
    jit::tracer::addInputs(node, "unitriangular", unitriangular);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("_th_trtrs_out", res1);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->_th_trtrs_out(res1_, res2_, self_, A_, upper, transpose, unitriangular);
  increment_version(res1);
  increment_version(res2);
  rebase_history(flatten_tensor_args( res1, res2 ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, res1);
    jit::tracer::addOutput(node, res2);
  }
  return std::forward_as_tuple(res1, res2);
}
Tensor VariableType::_th_var(const Tensor & self, int64_t dim, bool unbiased, bool keepdim) const {
  profiler::RecordFunction profiler("_th_var", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_var"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_th_var");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "unbiased", unbiased);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_th_var(self_, dim, unbiased, keepdim));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_th_var(const Tensor & self, bool unbiased) const {
  profiler::RecordFunction profiler("_th_var", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_var"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_th_var");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "unbiased", unbiased);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->_th_var(self_, unbiased));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor,Tensor,Tensor> VariableType::_thnn_fused_lstm_cell(const Tensor & input_gates, const Tensor & hidden_gates, const Tensor & cx, const Tensor & input_bias, const Tensor & hidden_bias) const {
  profiler::RecordFunction profiler("_thnn_fused_lstm_cell", Function::peek_at_next_sequence_nr());
  auto& input_gates_ = unpack(input_gates, "input_gates", 0);
  auto& hidden_gates_ = unpack(hidden_gates, "hidden_gates", 1);
  auto& cx_ = unpack(cx, "cx", 2);
  auto input_bias_ = unpack_opt(input_bias, "input_bias", 3);
  auto hidden_bias_ = unpack_opt(hidden_bias, "hidden_bias", 4);
  std::shared_ptr<ThnnFusedLstmCellBackward> grad_fn;
  if (compute_requires_grad( input_gates, hidden_gates, cx, input_bias, hidden_bias )) {
    grad_fn = std::shared_ptr<ThnnFusedLstmCellBackward>(new ThnnFusedLstmCellBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( input_gates, hidden_gates, cx, input_bias, hidden_bias ));
    grad_fn->cx_ = SavedVariable(cx, false);
    grad_fn->input_bias_ = SavedVariable(input_bias, false);
  }
  Tensor result0;
  Tensor result1;
  Tensor result2;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::_thnn_fused_lstm_cell");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input_gates", input_gates);
    jit::tracer::addInputs(node, "hidden_gates", hidden_gates);
    jit::tracer::addInputs(node, "cx", cx);
    jit::tracer::addInputs(node, "input_bias", input_bias);
    jit::tracer::addInputs(node, "hidden_bias", hidden_bias);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(result0, result1, result2) = as_variable(baseType->_thnn_fused_lstm_cell(input_gates_, hidden_gates_, cx_, input_bias_, hidden_bias_));
  set_history(flatten_tensor_args( result0, result1 ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
  }
  if (grad_fn) {
    grad_fn->result1_ = SavedVariable(result1, true);
    grad_fn->result2_ = SavedVariable(result2, true);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
Tensor & VariableType::abs_out(Tensor & result, const Tensor & self) const {
  profiler::RecordFunction profiler("abs_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("abs");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("abs");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::abs");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("abs_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->abs_out(result_, self_);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::adaptive_avg_pool3d_forward(const Tensor & self, IntList output_size) const {
  profiler::RecordFunction profiler("adaptive_avg_pool3d_forward", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<AdaptiveAvgPool3DBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<AdaptiveAvgPool3DBackward>(new AdaptiveAvgPool3DBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::adaptive_avg_pool3d_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto output = as_variable(baseType->adaptive_avg_pool3d_forward(self_, output_size));
  set_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
std::tuple<Tensor,Tensor> VariableType::adaptive_max_pool2d(const Tensor & self, IntList output_size) const {
  profiler::RecordFunction profiler("adaptive_max_pool2d", Function::peek_at_next_sequence_nr());
  Tensor output;
  Tensor indices;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::adaptive_max_pool2d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(output, indices) = TypeDefault::adaptive_max_pool2d(self, output_size);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
    jit::tracer::addOutput(node, indices);
  }
  return std::make_tuple(std::move(output), std::move(indices));
}
Tensor & VariableType::adaptive_max_pool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & indices) const {
  profiler::RecordFunction profiler("adaptive_max_pool2d_backward_out", Function::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& indices_ = unpack(indices, "indices", 3);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    throw_error_out_requires_grad("adaptive_max_pool2d_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("adaptive_max_pool2d_backward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::adaptive_max_pool2d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "indices", indices);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("adaptive_max_pool2d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->adaptive_max_pool2d_backward_out(grad_input_, grad_output_, self_, indices_);
  increment_version(grad_input);
  rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
std::tuple<Tensor &,Tensor &> VariableType::adaptive_max_pool3d_out(Tensor & output, Tensor & indices, const Tensor & self, IntList output_size) const {
  profiler::RecordFunction profiler("adaptive_max_pool3d_out", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::adaptive_max_pool3d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("adaptive_max_pool3d_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::adaptive_max_pool3d_out(output, indices, self, output_size);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
    jit::tracer::addOutput(node, indices);
  }
  return std::forward_as_tuple(output, indices);
}
Tensor VariableType::addcdiv(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
  profiler::RecordFunction profiler("addcdiv", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& tensor1_ = unpack(tensor1, "tensor1", 1);
  auto& tensor2_ = unpack(tensor2, "tensor2", 2);
  std::shared_ptr<AddcdivBackward> grad_fn;
  if (compute_requires_grad( self, tensor1, tensor2 )) {
    grad_fn = std::shared_ptr<AddcdivBackward>(new AddcdivBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, tensor1, tensor2 ));
    grad_fn->tensor2_ = SavedVariable(tensor2, false);
    grad_fn->value = value;
    grad_fn->tensor1_ = SavedVariable(tensor1, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::addcdiv");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "tensor1", tensor1);
    jit::tracer::addInputs(node, "tensor2", tensor2);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->addcdiv(self_, tensor1_, tensor2_, value));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::addcdiv_(Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
  profiler::RecordFunction profiler("addcdiv_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& tensor1_ = unpack(tensor1, "tensor1", 1);
  auto& tensor2_ = unpack(tensor2, "tensor2", 2);
  check_inplace(self);
  std::shared_ptr<AddcdivBackward> grad_fn;
  if (compute_requires_grad( self, tensor1, tensor2 )) {
    grad_fn = std::shared_ptr<AddcdivBackward>(new AddcdivBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, tensor1, tensor2 ));
    grad_fn->tensor2_ = SavedVariable(tensor2, false);
    grad_fn->value = value;
    grad_fn->tensor1_ = SavedVariable(tensor1, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::addcdiv");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "tensor1", tensor1);
    jit::tracer::addInputs(node, "tensor2", tensor2);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("addcdiv_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->addcdiv_(self_, tensor1_, tensor2_, value);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::addcmul_out(Tensor & result, const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
  profiler::RecordFunction profiler("addcmul_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& tensor1_ = unpack(tensor1, "tensor1", 2);
  auto& tensor2_ = unpack(tensor2, "tensor2", 3);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, tensor1, tensor2 )) {
    throw_error_out_requires_grad("addcmul");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("addcmul");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::addcmul");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "tensor1", tensor1);
    jit::tracer::addInputs(node, "tensor2", tensor2);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("addcmul_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->addcmul_out(result_, self_, tensor1_, tensor2_, value);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::addmm_out(Tensor & result, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
  profiler::RecordFunction profiler("addmm_out", Function::peek_at_next_sequence_nr());
  TypeDefault::addmm_out(result, self, mat1, mat2, beta, alpha);
  return result;
}
Tensor VariableType::addmv(const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) const {
  profiler::RecordFunction profiler("addmv", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::addmv");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mat", mat);
    jit::tracer::addInputs(node, "vec", vec);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::addmv(self, mat, vec, beta, alpha);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::addmv_(Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) const {
  profiler::RecordFunction profiler("addmv_", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::addmv");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mat", mat);
    jit::tracer::addInputs(node, "vec", vec);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("addmv_", self);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::addmv_(self, mat, vec, beta, alpha);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor VariableType::alpha_dropout(const Tensor & input, double p, bool train) const {
  profiler::RecordFunction profiler("alpha_dropout", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::alpha_dropout");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "train", train);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::alpha_dropout(input, p, train);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::alpha_dropout_(Tensor & self, double p, bool train) const {
  profiler::RecordFunction profiler("alpha_dropout_", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::alpha_dropout");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "train", train);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("alpha_dropout_", self);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::alpha_dropout_(self, p, train);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::atan2_out(Tensor & result, const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("atan2_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& other_ = unpack(other, "other", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, other )) {
    throw_error_out_requires_grad("atan2");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("atan2");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::atan2");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("atan2_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->atan2_out(result_, self_, other_);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::atan_out(Tensor & result, const Tensor & self) const {
  profiler::RecordFunction profiler("atan_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("atan");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("atan");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::atan");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("atan_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->atan_out(result_, self_);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::avg_pool1d(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
  profiler::RecordFunction profiler("avg_pool1d", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::avg_pool1d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "ceil_mode", ceil_mode);
    jit::tracer::addInputs(node, "count_include_pad", count_include_pad);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::avg_pool1d(self, kernel_size, stride, padding, ceil_mode, count_include_pad);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::avg_pool2d_out(Tensor & output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
  profiler::RecordFunction profiler("avg_pool2d_out", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::avg_pool2d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "ceil_mode", ceil_mode);
    jit::tracer::addInputs(node, "count_include_pad", count_include_pad);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("avg_pool2d_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::avg_pool2d_out(output, self, kernel_size, stride, padding, ceil_mode, count_include_pad);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor VariableType::binary_cross_entropy(const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction) const {
  profiler::RecordFunction profiler("binary_cross_entropy", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::binary_cross_entropy");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "reduction", reduction);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto output = TypeDefault::binary_cross_entropy(self, target, weight, reduction);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor & VariableType::binary_cross_entropy_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction) const {
  profiler::RecordFunction profiler("binary_cross_entropy_backward_out", Function::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& target_ = unpack(target, "target", 3);
  auto weight_ = unpack_opt(weight, "weight", 4);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self, target, weight )) {
    throw_error_out_requires_grad("binary_cross_entropy_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("binary_cross_entropy_backward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::binary_cross_entropy_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "reduction", reduction);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("binary_cross_entropy_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->binary_cross_entropy_backward_out(grad_input_, grad_output_, self_, target_, weight_, reduction);
  increment_version(grad_input);
  rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
std::vector<Tensor> VariableType::broadcast_tensors(TensorList tensors) const {
  profiler::RecordFunction profiler("broadcast_tensors", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::broadcast_tensors");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "tensors", tensors);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::broadcast_tensors(tensors);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor,Tensor,Tensor> VariableType::btrifact_with_info(const Tensor & self, bool pivot) const {
  profiler::RecordFunction profiler("btrifact_with_info", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<BtrifactWithInfoBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<BtrifactWithInfoBackward>(new BtrifactWithInfoBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  Tensor result0;
  Tensor result1;
  Tensor result2;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::btrifact_with_info");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "pivot", pivot);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(result0, result1, result2) = as_variable(baseType->btrifact_with_info(self_, pivot));
  set_history(flatten_tensor_args( result0, result1, result2 ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
Tensor VariableType::clamp_min(const Tensor & self, Scalar min) const {
  profiler::RecordFunction profiler("clamp_min", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<ClampMinBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<ClampMinBackward>(new ClampMinBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->min = min;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::clamp_min");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "min", min);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->clamp_min(self_, min));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::clamp_min_(Tensor & self, Scalar min) const {
  profiler::RecordFunction profiler("clamp_min_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<ClampMinBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<ClampMinBackward>(new ClampMinBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self.clone(), false);
    grad_fn->min = min;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::clamp_min");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "min", min);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("clamp_min_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->clamp_min_(self_, min);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::clamp_out(Tensor & result, const Tensor & self, c10::optional<Scalar> min, c10::optional<Scalar> max) const {
  profiler::RecordFunction profiler("clamp_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("clamp");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("clamp");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::clamp");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "min", min);
    jit::tracer::addInputs(node, "max", max);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("clamp_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->clamp_out(result_, self_, min, max);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::clone(const Tensor & self) const {
  profiler::RecordFunction profiler("clone", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<CloneBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<CloneBackward>(new CloneBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::clone");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->clone(self_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::contiguous(const Tensor & self) const {
  profiler::RecordFunction profiler("contiguous", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::contiguous");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::contiguous(self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::conv3d(const Tensor & input, const Tensor & weight, const Tensor & bias, IntList stride, IntList padding, IntList dilation, int64_t groups) const {
  profiler::RecordFunction profiler("conv3d", Function::peek_at_next_sequence_nr());
  auto result = TypeDefault::conv3d(input, weight, bias, stride, padding, dilation, groups);
  return result;
}
Tensor VariableType::conv_transpose2d(const Tensor & input, const Tensor & weight, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, int64_t groups, IntList dilation) const {
  profiler::RecordFunction profiler("conv_transpose2d", Function::peek_at_next_sequence_nr());
  auto result = TypeDefault::conv_transpose2d(input, weight, bias, stride, padding, output_padding, groups, dilation);
  return result;
}
Tensor & VariableType::cos_out(Tensor & result, const Tensor & self) const {
  profiler::RecordFunction profiler("cos_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("cos");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("cos");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::cos");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("cos_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->cos_out(result_, self_);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::cosh(const Tensor & self) const {
  profiler::RecordFunction profiler("cosh", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<CoshBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<CoshBackward>(new CoshBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::cosh");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->cosh(self_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::cosh_(Tensor & self) const {
  profiler::RecordFunction profiler("cosh_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<CoshBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<CoshBackward>(new CoshBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self.clone(), false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::cosh");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("cosh_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->cosh_(self_);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor VariableType::cross(const Tensor & self, const Tensor & other, int64_t dim) const {
  profiler::RecordFunction profiler("cross", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  std::shared_ptr<CrossBackward> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::shared_ptr<CrossBackward>(new CrossBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->dim = dim;
    grad_fn->other_ = SavedVariable(other, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::cross");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->cross(self_, other_, dim));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor,Tensor,Tensor> VariableType::cudnn_batch_norm(const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double exponential_average_factor, double epsilon) const {
  profiler::RecordFunction profiler("cudnn_batch_norm", Function::peek_at_next_sequence_nr());
  auto& input_ = unpack(input, "input", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  auto bias_ = unpack_opt(bias, "bias", 2);
  auto running_mean_ = unpack_opt(running_mean, "running_mean", 3);
  auto running_var_ = unpack_opt(running_var, "running_var", 4);
  check_no_requires_grad(running_mean, "running_mean");
  check_no_requires_grad(running_var, "running_var");
  std::shared_ptr<CudnnBatchNormBackward> grad_fn;
  if (compute_requires_grad( input, weight, bias )) {
    grad_fn = std::shared_ptr<CudnnBatchNormBackward>(new CudnnBatchNormBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( input, weight, bias ));
    grad_fn->input_ = SavedVariable(input, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->running_mean_ = SavedVariable(running_mean, false);
    grad_fn->running_var_ = SavedVariable(running_var, false);
    grad_fn->training = training;
    grad_fn->epsilon = epsilon;
  }
  Tensor result0;
  Tensor result1;
  Tensor result2;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::cudnn_batch_norm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "running_mean", running_mean);
    jit::tracer::addInputs(node, "running_var", running_var);
    jit::tracer::addInputs(node, "training", training);
    jit::tracer::addInputs(node, "exponential_average_factor", exponential_average_factor);
    jit::tracer::addInputs(node, "epsilon", epsilon);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(result0, result1, result2) = as_variable(baseType->cudnn_batch_norm(input_, weight_, bias_, running_mean_, running_var_, training, exponential_average_factor, epsilon));
  set_history(flatten_tensor_args( result0 ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
  }
  if (grad_fn) {
    grad_fn->result1_ = SavedVariable(result1, true);
    grad_fn->result2_ = SavedVariable(result2, true);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
std::tuple<Tensor,Tensor,Tensor> VariableType::cudnn_convolution_transpose_backward(const Tensor & self, const Tensor & grad_output, const Tensor & weight, IntList padding, IntList output_padding, IntList stride, IntList dilation, int64_t groups, bool benchmark, bool deterministic, std::array<bool,3> output_mask) const {
  profiler::RecordFunction profiler("cudnn_convolution_transpose_backward", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& weight_ = unpack(weight, "weight", 2);
  std::shared_ptr<CudnnConvolutionTransposeBackwardBackward> grad_fn;
  if (compute_requires_grad( self, grad_output, weight )) {
    grad_fn = std::shared_ptr<CudnnConvolutionTransposeBackwardBackward>(new CudnnConvolutionTransposeBackwardBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, grad_output, weight ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->padding = padding.vec();
    grad_fn->output_padding = output_padding.vec();
    grad_fn->stride = stride.vec();
    grad_fn->dilation = dilation.vec();
    grad_fn->groups = groups;
    grad_fn->benchmark = benchmark;
    grad_fn->deterministic = deterministic;
  }
  Tensor result0;
  Tensor result1;
  Tensor result2;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::cudnn_convolution_transpose_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "output_padding", output_padding);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "groups", groups);
    jit::tracer::addInputs(node, "benchmark", benchmark);
    jit::tracer::addInputs(node, "deterministic", deterministic);
    jit::tracer::addInputs(node, "output_mask", output_mask);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(result0, result1, result2) = as_variable(baseType->cudnn_convolution_transpose_backward(self_, grad_output_, weight_, padding, output_padding, stride, dilation, groups, benchmark, deterministic, output_mask));
  set_history(flatten_tensor_args( result0, result1, result2 ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
Tensor VariableType::cudnn_convolution_transpose_backward_bias(const Tensor & grad_output) const {
  profiler::RecordFunction profiler("cudnn_convolution_transpose_backward_bias", Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad_output )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("cudnn_convolution_transpose_backward_bias"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::cudnn_convolution_transpose_backward_bias");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->cudnn_convolution_transpose_backward_bias(grad_output_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::cudnn_convolution_transpose_backward_input(const Tensor & grad_output, const Tensor & weight, IntList padding, IntList stride, IntList dilation, int64_t groups, bool benchmark, bool deterministic) const {
  profiler::RecordFunction profiler("cudnn_convolution_transpose_backward_input", Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad_output, weight )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("cudnn_convolution_transpose_backward_input"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, weight ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::cudnn_convolution_transpose_backward_input");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "groups", groups);
    jit::tracer::addInputs(node, "benchmark", benchmark);
    jit::tracer::addInputs(node, "deterministic", deterministic);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->cudnn_convolution_transpose_backward_input(grad_output_, weight_, padding, stride, dilation, groups, benchmark, deterministic));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::cudnn_grid_sampler(const Tensor & self, const Tensor & grid) const {
  profiler::RecordFunction profiler("cudnn_grid_sampler", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& grid_ = unpack(grid, "grid", 1);
  std::shared_ptr<CudnnGridSamplerBackward> grad_fn;
  if (compute_requires_grad( self, grid )) {
    grad_fn = std::shared_ptr<CudnnGridSamplerBackward>(new CudnnGridSamplerBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, grid ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->grid_ = SavedVariable(grid, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::cudnn_grid_sampler");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "grid", grid);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto output = as_variable(baseType->cudnn_grid_sampler(self_, grid_));
  set_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
bool VariableType::cudnn_is_acceptable(const Tensor & self) const {
  profiler::RecordFunction profiler("cudnn_is_acceptable", Function::peek_at_next_sequence_nr());
  auto result = TypeDefault::cudnn_is_acceptable(self);
  return result;
}
Tensor & VariableType::diag_out(Tensor & result, const Tensor & self, int64_t diagonal) const {
  profiler::RecordFunction profiler("diag_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("diag");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("diag");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::diag");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "diagonal", diagonal);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("diag_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->diag_out(result_, self_, diagonal);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::digamma_out(Tensor & result, const Tensor & self) const {
  profiler::RecordFunction profiler("digamma_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("digamma");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("digamma");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::digamma");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("digamma_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->digamma_out(result_, self_);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor,Tensor> VariableType::eig(const Tensor & self, bool eigenvectors) const {
  profiler::RecordFunction profiler("eig", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<EigBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<EigBackward>(new EigBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  Tensor result0;
  Tensor result1;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::eig");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "eigenvectors", eigenvectors);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(result0, result1) = as_variable(baseType->eig(self_, eigenvectors));
  set_history(flatten_tensor_args( result0, result1 ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
Tensor VariableType::empty(IntList size, const TensorOptions & options) const {
  profiler::RecordFunction profiler("empty", Function::peek_at_next_sequence_nr());
  auto options_ = TensorOptions(options).is_variable(false);
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::empty");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->empty(size, options_));
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::eq(const Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("eq", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::eq");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::eq(self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::eq(const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("eq", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::eq");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::eq(self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::eq_(Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("eq_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<EqBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<EqBackward0>(new EqBackward0(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_info = self;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::eq");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("eq_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->eq_(self_, other);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::eq_(Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("eq_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  check_inplace(self);
  std::shared_ptr<EqBackward1> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::shared_ptr<EqBackward1>(new EqBackward1(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    grad_fn->other_info = other;
    grad_fn->self_info = self;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::eq");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("eq_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->eq_(self_, other_);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::erfinv_out(Tensor & result, const Tensor & self) const {
  profiler::RecordFunction profiler("erfinv_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("erfinv");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("erfinv");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::erfinv");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("erfinv_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->erfinv_out(result_, self_);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::feature_dropout(const Tensor & input, double p, bool train) const {
  profiler::RecordFunction profiler("feature_dropout", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::feature_dropout");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "train", train);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::feature_dropout(input, p, train);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::feature_dropout_(Tensor & self, double p, bool train) const {
  profiler::RecordFunction profiler("feature_dropout_", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::feature_dropout");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "train", train);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("feature_dropout_", self);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::feature_dropout_(self, p, train);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
std::tuple<Tensor,Tensor> VariableType::fractional_max_pool2d_forward(const Tensor & self, IntList kernel_size, IntList output_size, const Tensor & random_samples) const {
  profiler::RecordFunction profiler("fractional_max_pool2d_forward", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& random_samples_ = unpack(random_samples, "random_samples", 3);
  check_no_requires_grad(random_samples, "random_samples");
  std::shared_ptr<FractionalMaxPool2DBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<FractionalMaxPool2DBackward>(new FractionalMaxPool2DBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->kernel_size = kernel_size.vec();
    grad_fn->output_size = output_size.vec();
  }
  Tensor output;
  Tensor indices;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::fractional_max_pool2d_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "random_samples", random_samples);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(output, indices) = as_variable(baseType->fractional_max_pool2d_forward(self_, kernel_size, output_size, random_samples_));
  set_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
    jit::tracer::addOutput(node, indices);
  }
  if (grad_fn) {
    grad_fn->indices_ = SavedVariable(indices, true);
  }
  return std::make_tuple(std::move(output), std::move(indices));
}
Tensor & VariableType::full_out(Tensor & result, IntList size, Scalar fill_value) const {
  profiler::RecordFunction profiler("full_out", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::full");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "fill_value", fill_value);
    jit::tracer::addInputs(node, "result", result.options());
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("full_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::full_out(result, size, fill_value);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::gather_out(Tensor & result, const Tensor & self, int64_t dim, const Tensor & index) const {
  profiler::RecordFunction profiler("gather_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& index_ = unpack(index, "index", 3);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, index )) {
    throw_error_out_requires_grad("gather");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("gather");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::gather");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("gather_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->gather_out(result_, self_, dim, index_);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::ge(const Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("ge", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::ge");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::ge(self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::ge(const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("ge", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::ge");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::ge(self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::ge_(Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("ge_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<GeBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<GeBackward0>(new GeBackward0(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_info = self;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::ge");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("ge_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->ge_(self_, other);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::ge_(Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("ge_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  check_inplace(self);
  std::shared_ptr<GeBackward1> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::shared_ptr<GeBackward1>(new GeBackward1(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    grad_fn->other_info = other;
    grad_fn->self_info = self;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::ge");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("ge_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->ge_(self_, other_);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::geometric_(Tensor & self, double p, Generator * generator) const {
  profiler::RecordFunction profiler("geometric_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<GeometricBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<GeometricBackward>(new GeometricBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::geometric");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("geometric_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->geometric_(self_, p, generator);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor VariableType::glu_backward(const Tensor & grad_output, const Tensor & self, int64_t dim) const {
  profiler::RecordFunction profiler("glu_backward", Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<GluBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::shared_ptr<GluBackwardBackward>(new GluBackwardBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->dim = dim;
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::glu_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto grad_input = as_variable(baseType->glu_backward(grad_output_, self_, dim));
  set_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor & VariableType::glu_forward_out(Tensor & output, const Tensor & self, int64_t dim) const {
  profiler::RecordFunction profiler("glu_forward_out", Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("glu_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("glu_forward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::glu_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("glu_forward_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->glu_forward_out(output_, self_, dim);
  increment_version(output);
  rebase_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
std::tuple<Tensor,Tensor> VariableType::grid_sampler_2d_backward(const Tensor & grad_output, const Tensor & input, const Tensor & grid, int64_t interpolation_mode, int64_t padding_mode) const {
  profiler::RecordFunction profiler("grid_sampler_2d_backward", Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& input_ = unpack(input, "input", 1);
  auto& grid_ = unpack(grid, "grid", 2);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad_output, input, grid )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("grid_sampler_2d_backward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, input, grid ));
  }
  Tensor result0;
  Tensor result1;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::grid_sampler_2d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "grid", grid);
    jit::tracer::addInputs(node, "interpolation_mode", interpolation_mode);
    jit::tracer::addInputs(node, "padding_mode", padding_mode);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(result0, result1) = as_variable(baseType->grid_sampler_2d_backward(grad_output_, input_, grid_, interpolation_mode, padding_mode));
  set_history(flatten_tensor_args( result0, result1 ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
Tensor VariableType::grid_sampler_3d(const Tensor & input, const Tensor & grid, int64_t interpolation_mode, int64_t padding_mode) const {
  profiler::RecordFunction profiler("grid_sampler_3d", Function::peek_at_next_sequence_nr());
  auto& input_ = unpack(input, "input", 0);
  auto& grid_ = unpack(grid, "grid", 1);
  std::shared_ptr<GridSampler3DBackward> grad_fn;
  if (compute_requires_grad( input, grid )) {
    grad_fn = std::shared_ptr<GridSampler3DBackward>(new GridSampler3DBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( input, grid ));
    grad_fn->input_ = SavedVariable(input, false);
    grad_fn->grid_ = SavedVariable(grid, false);
    grad_fn->interpolation_mode = interpolation_mode;
    grad_fn->padding_mode = padding_mode;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::grid_sampler_3d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "grid", grid);
    jit::tracer::addInputs(node, "interpolation_mode", interpolation_mode);
    jit::tracer::addInputs(node, "padding_mode", padding_mode);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->grid_sampler_3d(input_, grid_, interpolation_mode, padding_mode));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor,Tensor> VariableType::gru(const Tensor & input, const Tensor & hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first) const {
  profiler::RecordFunction profiler("gru", Function::peek_at_next_sequence_nr());
  Tensor result0;
  Tensor result1;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::gru");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "hx", hx);
    jit::tracer::addInputs(node, "params", params);
    jit::tracer::addInputs(node, "has_biases", has_biases);
    jit::tracer::addInputs(node, "num_layers", num_layers);
    jit::tracer::addInputs(node, "dropout", dropout);
    jit::tracer::addInputs(node, "train", train);
    jit::tracer::addInputs(node, "bidirectional", bidirectional);
    jit::tracer::addInputs(node, "batch_first", batch_first);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(result0, result1) = TypeDefault::gru(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
std::tuple<Tensor,Tensor> VariableType::gru(const Tensor & data, const Tensor & batch_sizes, const Tensor & hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional) const {
  profiler::RecordFunction profiler("gru", Function::peek_at_next_sequence_nr());
  Tensor result0;
  Tensor result1;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::gru");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "data", data);
    jit::tracer::addInputs(node, "batch_sizes", batch_sizes);
    jit::tracer::addInputs(node, "hx", hx);
    jit::tracer::addInputs(node, "params", params);
    jit::tracer::addInputs(node, "has_biases", has_biases);
    jit::tracer::addInputs(node, "num_layers", num_layers);
    jit::tracer::addInputs(node, "dropout", dropout);
    jit::tracer::addInputs(node, "train", train);
    jit::tracer::addInputs(node, "bidirectional", bidirectional);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(result0, result1) = TypeDefault::gru(data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
Tensor VariableType::gt(const Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("gt", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::gt");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::gt(self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::gt(const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("gt", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::gt");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::gt(self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::gt_(Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("gt_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<GtBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<GtBackward0>(new GtBackward0(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_info = self;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::gt");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("gt_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->gt_(self_, other);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::gt_(Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("gt_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  check_inplace(self);
  std::shared_ptr<GtBackward1> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::shared_ptr<GtBackward1>(new GtBackward1(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    grad_fn->other_info = other;
    grad_fn->self_info = self;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::gt");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("gt_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->gt_(self_, other_);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor VariableType::hardtanh_forward(const Tensor & self, Scalar min_val, Scalar max_val) const {
  profiler::RecordFunction profiler("hardtanh_forward", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<HardtanhBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<HardtanhBackward0>(new HardtanhBackward0(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->min_val = min_val;
    grad_fn->max_val = max_val;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::hardtanh_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "min_val", min_val);
    jit::tracer::addInputs(node, "max_val", max_val);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto output = as_variable(baseType->hardtanh_forward(self_, min_val, max_val));
  set_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor & VariableType::hardtanh_forward_(Tensor & self, Scalar min_val, Scalar max_val) const {
  profiler::RecordFunction profiler("hardtanh_forward_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<HardtanhBackward1> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<HardtanhBackward1>(new HardtanhBackward1(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->min_val = min_val;
    grad_fn->max_val = max_val;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::hardtanh_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "min_val", min_val);
    jit::tracer::addInputs(node, "max_val", max_val);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("hardtanh_forward_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->hardtanh_forward_(self_, min_val, max_val);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  if (grad_fn) {
    grad_fn->output_ = SavedVariable(self, true);
  }
  return self;
}
Tensor VariableType::hinge_embedding_loss(const Tensor & self, const Tensor & target, double margin, int64_t reduction) const {
  profiler::RecordFunction profiler("hinge_embedding_loss", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::hinge_embedding_loss");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "margin", margin);
    jit::tracer::addInputs(node, "reduction", reduction);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::hinge_embedding_loss(self, target, margin, reduction);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::histc(const Tensor & self, int64_t bins, Scalar min, Scalar max) const {
  profiler::RecordFunction profiler("histc", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<HistcBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<HistcBackward>(new HistcBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::histc");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "bins", bins);
    jit::tracer::addInputs(node, "min", min);
    jit::tracer::addInputs(node, "max", max);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->histc(self_, bins, min, max));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::hspmm(const Tensor & mat1, const Tensor & mat2) const {
  profiler::RecordFunction profiler("hspmm", Function::peek_at_next_sequence_nr());
  auto& mat1_ = unpack(mat1, "mat1", 0);
  auto& mat2_ = unpack(mat2, "mat2", 1);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( mat1, mat2 )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("hspmm"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( mat1, mat2 ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::hspmm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "mat1", mat1);
    jit::tracer::addInputs(node, "mat2", mat2);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->hspmm(mat1_, mat2_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::index_copy_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) const {
  profiler::RecordFunction profiler("index_copy_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& index_ = unpack(index, "index", 2);
  auto& source_ = unpack(source, "source", 3);
  check_inplace(self);
  std::shared_ptr<IndexCopyBackward> grad_fn;
  if (compute_requires_grad( self, source )) {
    grad_fn = std::shared_ptr<IndexCopyBackward>(new IndexCopyBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, source ));
    grad_fn->dim = dim;
    grad_fn->index_ = SavedVariable(index, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::index_copy");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "source", source);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("index_copy_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->index_copy_(self_, dim, index_, source_);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::index_fill_(Tensor & self, int64_t dim, const Tensor & index, Scalar value) const {
  profiler::RecordFunction profiler("index_fill_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& index_ = unpack(index, "index", 2);
  check_inplace(self);
  check_no_requires_grad(index, "index");
  std::shared_ptr<IndexFillBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<IndexFillBackward0>(new IndexFillBackward0(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->dim = dim;
    grad_fn->index_ = SavedVariable(index, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::index_fill");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("index_fill_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->index_fill_(self_, dim, index_, value);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::index_fill_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & value) const {
  profiler::RecordFunction profiler("index_fill_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& index_ = unpack(index, "index", 2);
  auto& value_ = unpack(value, "value", 3);
  check_inplace(self);
  check_no_requires_grad(index, "index");
  std::shared_ptr<IndexFillBackward1> grad_fn;
  if (compute_requires_grad( self, value )) {
    grad_fn = std::shared_ptr<IndexFillBackward1>(new IndexFillBackward1(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, value ));
    grad_fn->dim = dim;
    grad_fn->index_ = SavedVariable(index, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::index_fill");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("index_fill_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->index_fill_(self_, dim, index_, value_);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor VariableType::inverse(const Tensor & self) const {
  profiler::RecordFunction profiler("inverse", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<InverseBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<InverseBackward>(new InverseBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::inverse");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->inverse(self_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  return result;
}
Tensor VariableType::irfft(const Tensor & self, int64_t signal_ndim, bool normalized, bool onesided, IntList signal_sizes) const {
  profiler::RecordFunction profiler("irfft", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::irfft");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "signal_ndim", signal_ndim);
    jit::tracer::addInputs(node, "normalized", normalized);
    jit::tracer::addInputs(node, "onesided", onesided);
    jit::tracer::addInputs(node, "signal_sizes", signal_sizes);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::irfft(self, signal_ndim, normalized, onesided, signal_sizes);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
bool VariableType::is_contiguous(const Tensor & self) const {
  auto result = TypeDefault::is_contiguous(self);
  return result;
}
bool VariableType::is_nonzero(const Tensor & self) const {
  profiler::RecordFunction profiler("is_nonzero", Function::peek_at_next_sequence_nr());
  auto result = TypeDefault::is_nonzero(self);
  return result;
}
bool VariableType::is_set_to(const Tensor & self, const Tensor & tensor) const {
  auto result = TypeDefault::is_set_to(self, tensor);
  return result;
}
bool VariableType::is_signed(const Tensor & self) const {
  auto result = TypeDefault::is_signed(self);
  return result;
}
Tensor VariableType::isclose(const Tensor & self, const Tensor & other, double rtol, double atol, bool equal_nan) const {
  profiler::RecordFunction profiler("isclose", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::isclose");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "rtol", rtol);
    jit::tracer::addInputs(node, "atol", atol);
    jit::tracer::addInputs(node, "equal_nan", equal_nan);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::isclose(self, other, rtol, atol, equal_nan);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::kl_div_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction) const {
  profiler::RecordFunction profiler("kl_div_backward", Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& target_ = unpack(target, "target", 2);
  std::shared_ptr<KlDivBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self, target )) {
    grad_fn = std::shared_ptr<KlDivBackwardBackward>(new KlDivBackwardBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self, target ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->target_ = SavedVariable(target, false);
    grad_fn->reduction = reduction;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::kl_div_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "reduction", reduction);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->kl_div_backward(grad_output_, self_, target_, reduction));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::le(const Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("le", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::le");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::le(self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::le(const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("le", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::le");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::le(self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::le_(Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("le_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<LeBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<LeBackward0>(new LeBackward0(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_info = self;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::le");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("le_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->le_(self_, other);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::le_(Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("le_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  check_inplace(self);
  std::shared_ptr<LeBackward1> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::shared_ptr<LeBackward1>(new LeBackward1(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    grad_fn->other_info = other;
    grad_fn->self_info = self;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::le");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("le_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->le_(self_, other_);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor VariableType::leaky_relu(const Tensor & self, Scalar negative_slope) const {
  profiler::RecordFunction profiler("leaky_relu", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::leaky_relu");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "negative_slope", negative_slope);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto output = TypeDefault::leaky_relu(self, negative_slope);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor & VariableType::leaky_relu_(Tensor & self, Scalar negative_slope) const {
  profiler::RecordFunction profiler("leaky_relu_", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::leaky_relu");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "negative_slope", negative_slope);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("leaky_relu_", self);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::leaky_relu_(self, negative_slope);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::leaky_relu_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar negative_slope) const {
  profiler::RecordFunction profiler("leaky_relu_backward_out", Function::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    throw_error_out_requires_grad("leaky_relu_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("leaky_relu_backward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::leaky_relu_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "negative_slope", negative_slope);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("leaky_relu_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->leaky_relu_backward_out(grad_input_, grad_output_, self_, negative_slope);
  increment_version(grad_input);
  rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor & VariableType::lerp_out(Tensor & result, const Tensor & self, const Tensor & end, Scalar weight) const {
  profiler::RecordFunction profiler("lerp_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& end_ = unpack(end, "end", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, end )) {
    throw_error_out_requires_grad("lerp");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("lerp");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::lerp");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "end", end);
    jit::tracer::addInputs(node, "weight", weight);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("lerp_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->lerp_out(result_, self_, end_, weight);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::log10(const Tensor & self) const {
  profiler::RecordFunction profiler("log10", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<Log10Backward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<Log10Backward>(new Log10Backward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::log10");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->log10(self_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::log10_(Tensor & self) const {
  profiler::RecordFunction profiler("log10_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<Log10Backward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<Log10Backward>(new Log10Backward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self.clone(), false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::log10");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("log10_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->log10_(self_);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::log_sigmoid_out(Tensor & output, const Tensor & self) const {
  profiler::RecordFunction profiler("log_sigmoid_out", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::log_sigmoid");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("log_sigmoid_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::log_sigmoid_out(output, self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor VariableType::logdet(const Tensor & self) const {
  profiler::RecordFunction profiler("logdet", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<LogdetBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<LogdetBackward>(new LogdetBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::logdet");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->logdet(self_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  return result;
}
std::tuple<Tensor,Tensor> VariableType::lstm_cell(const Tensor & input, TensorList hx, const Tensor & w_ih, const Tensor & w_hh, const Tensor & b_ih, const Tensor & b_hh) const {
  profiler::RecordFunction profiler("lstm_cell", Function::peek_at_next_sequence_nr());
  Tensor result0;
  Tensor result1;
  std::tie(result0, result1) = TypeDefault::lstm_cell(input, hx, w_ih, w_hh, b_ih, b_hh);
  return std::make_tuple(std::move(result0), std::move(result1));
}
Tensor VariableType::lt(const Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("lt", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::lt");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::lt(self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::lt(const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("lt", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::lt");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::lt(self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::lt_(Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("lt_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<LtBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<LtBackward0>(new LtBackward0(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_info = self;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::lt");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("lt_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->lt_(self_, other);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::lt_(Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("lt_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  check_inplace(self);
  std::shared_ptr<LtBackward1> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::shared_ptr<LtBackward1>(new LtBackward1(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    grad_fn->other_info = other;
    grad_fn->self_info = self;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::lt");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("lt_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->lt_(self_, other_);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor VariableType::masked_select(const Tensor & self, const Tensor & mask) const {
  profiler::RecordFunction profiler("masked_select", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& mask_ = unpack(mask, "mask", 1);
  check_no_requires_grad(mask, "mask");
  std::shared_ptr<MaskedSelectBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<MaskedSelectBackward>(new MaskedSelectBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes().vec();
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->mask_sizes = mask.sizes().vec();
    grad_fn->mask_ = SavedVariable(mask, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::masked_select");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mask", mask);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->masked_select(self_, mask_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::matrix_rank(const Tensor & self, double tol, bool symmetric) const {
  profiler::RecordFunction profiler("matrix_rank", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::matrix_rank");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "tol", tol);
    jit::tracer::addInputs(node, "symmetric", symmetric);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::matrix_rank(self, tol, symmetric);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::matrix_rank(const Tensor & self, bool symmetric) const {
  profiler::RecordFunction profiler("matrix_rank", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::matrix_rank");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "symmetric", symmetric);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::matrix_rank(self, symmetric);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor,Tensor> VariableType::max_pool2d_with_indices_forward(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) const {
  profiler::RecordFunction profiler("max_pool2d_with_indices_forward", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<MaxPool2DWithIndicesBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<MaxPool2DWithIndicesBackward>(new MaxPool2DWithIndicesBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->kernel_size = kernel_size.vec();
    grad_fn->stride = stride.vec();
    grad_fn->padding = padding.vec();
    grad_fn->dilation = dilation.vec();
    grad_fn->ceil_mode = ceil_mode;
  }
  Tensor output;
  Tensor indices;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::max_pool2d_with_indices_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "ceil_mode", ceil_mode);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(output, indices) = as_variable(baseType->max_pool2d_with_indices_forward(self_, kernel_size, stride, padding, dilation, ceil_mode));
  set_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
    jit::tracer::addOutput(node, indices);
  }
  if (grad_fn) {
    grad_fn->indices_ = SavedVariable(indices, true);
  }
  return std::make_tuple(std::move(output), std::move(indices));
}
Tensor VariableType::max_pool3d(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) const {
  profiler::RecordFunction profiler("max_pool3d", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::max_pool3d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "ceil_mode", ceil_mode);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::max_pool3d(self, kernel_size, stride, padding, dilation, ceil_mode);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::max_pool3d_with_indices_backward(const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode, const Tensor & indices) const {
  profiler::RecordFunction profiler("max_pool3d_with_indices_backward", Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& indices_ = unpack(indices, "indices", 7);
  std::shared_ptr<MaxPool3DWithIndicesBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::shared_ptr<MaxPool3DWithIndicesBackwardBackward>(new MaxPool3DWithIndicesBackwardBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->indices_ = SavedVariable(indices, false);
    grad_fn->self_info = self;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::max_pool3d_with_indices_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "ceil_mode", ceil_mode);
    jit::tracer::addInputs(node, "indices", indices);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto grad_input = as_variable(baseType->max_pool3d_with_indices_backward(grad_output_, self_, kernel_size, stride, padding, dilation, ceil_mode, indices_));
  set_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
std::tuple<Tensor &,Tensor &> VariableType::max_pool3d_with_indices_forward_out(Tensor & output, Tensor & indices, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) const {
  profiler::RecordFunction profiler("max_pool3d_with_indices_forward_out", Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& indices_ = unpack(indices, "indices", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("max_pool3d_with_indices_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("max_pool3d_with_indices_forward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::max_pool3d_with_indices_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "ceil_mode", ceil_mode);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("max_pool3d_with_indices_forward_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->max_pool3d_with_indices_forward_out(output_, indices_, self_, kernel_size, stride, padding, dilation, ceil_mode);
  increment_version(output);
  rebase_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
    jit::tracer::addOutput(node, indices);
  }
  return std::forward_as_tuple(output, indices);
}
Tensor & VariableType::max_unpool2d_out(Tensor & output, const Tensor & self, const Tensor & indices, IntList output_size) const {
  profiler::RecordFunction profiler("max_unpool2d_out", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::max_unpool2d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "output_size", output_size);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("max_unpool2d_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::max_unpool2d_out(output, self, indices, output_size);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
std::tuple<Tensor,Tensor> VariableType::min(const Tensor & self, int64_t dim, bool keepdim) const {
  profiler::RecordFunction profiler("min", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<MinBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<MinBackward0>(new MinBackward0(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes().vec();
    grad_fn->dim = dim;
    grad_fn->keepdim = keepdim;
  }
  Tensor result0;
  Tensor result1;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::min");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(result0, result1) = as_variable(baseType->min(self_, dim, keepdim));
  set_history(flatten_tensor_args( result0 ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  if (grad_fn) {
    grad_fn->result1_ = SavedVariable(result1, true);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
Tensor VariableType::min(const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("min", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  std::shared_ptr<MinBackward2> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::shared_ptr<MinBackward2>(new MinBackward2(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->other_ = SavedVariable(other, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::min");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->min(self_, other_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::min(const Tensor & self) const {
  profiler::RecordFunction profiler("min", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<MinBackward1> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<MinBackward1>(new MinBackward1(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::min");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->min(self_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  return result;
}
Tensor VariableType::miopen_convolution(const Tensor & self, const Tensor & weight, const Tensor & bias, IntList padding, IntList stride, IntList dilation, int64_t groups, bool benchmark, bool deterministic) const {
  profiler::RecordFunction profiler("miopen_convolution", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  auto bias_ = unpack_opt(bias, "bias", 2);
  std::shared_ptr<MiopenConvolutionBackward> grad_fn;
  if (compute_requires_grad( self, weight, bias )) {
    grad_fn = std::shared_ptr<MiopenConvolutionBackward>(new MiopenConvolutionBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, weight, bias ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->padding = padding.vec();
    grad_fn->stride = stride.vec();
    grad_fn->dilation = dilation.vec();
    grad_fn->groups = groups;
    grad_fn->benchmark = benchmark;
    grad_fn->deterministic = deterministic;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::miopen_convolution");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "groups", groups);
    jit::tracer::addInputs(node, "benchmark", benchmark);
    jit::tracer::addInputs(node, "deterministic", deterministic);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->miopen_convolution(self_, weight_, bias_, padding, stride, dilation, groups, benchmark, deterministic));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::miopen_convolution_transpose_backward_weight(IntList weight_size, const Tensor & grad_output, const Tensor & self, IntList padding, IntList stride, IntList dilation, int64_t groups, bool benchmark, bool deterministic) const {
  profiler::RecordFunction profiler("miopen_convolution_transpose_backward_weight", Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("miopen_convolution_transpose_backward_weight"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::miopen_convolution_transpose_backward_weight");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "weight_size", weight_size);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "groups", groups);
    jit::tracer::addInputs(node, "benchmark", benchmark);
    jit::tracer::addInputs(node, "deterministic", deterministic);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->miopen_convolution_transpose_backward_weight(weight_size, grad_output_, self_, padding, stride, dilation, groups, benchmark, deterministic));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor,Tensor> VariableType::mkldnn_convolution_backward_weights(IntList weight_size, const Tensor & grad_output, const Tensor & self, IntList padding, IntList stride, IntList dilation, int64_t groups, bool bias_defined) const {
  profiler::RecordFunction profiler("mkldnn_convolution_backward_weights", Function::peek_at_next_sequence_nr());
  Tensor result0;
  Tensor result1;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::mkldnn_convolution_backward_weights");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "weight_size", weight_size);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "groups", groups);
    jit::tracer::addInputs(node, "bias_defined", bias_defined);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(result0, result1) = TypeDefault::mkldnn_convolution_backward_weights(weight_size, grad_output, self, padding, stride, dilation, groups, bias_defined);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
Tensor VariableType::mse_loss(const Tensor & self, const Tensor & target, int64_t reduction) const {
  profiler::RecordFunction profiler("mse_loss", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::mse_loss");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "reduction", reduction);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto output = TypeDefault::mse_loss(self, target, reduction);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor & VariableType::mse_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction) const {
  profiler::RecordFunction profiler("mse_loss_backward_out", Function::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& target_ = unpack(target, "target", 3);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self, target )) {
    throw_error_out_requires_grad("mse_loss_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("mse_loss_backward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::mse_loss_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "reduction", reduction);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("mse_loss_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->mse_loss_backward_out(grad_input_, grad_output_, self_, target_, reduction);
  increment_version(grad_input);
  rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor VariableType::mul(const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("mul", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  std::shared_ptr<MulBackward0> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::shared_ptr<MulBackward0>(new MulBackward0(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->other_ = SavedVariable(other, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::mul");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->mul(self_, other_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::mul(const Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("mul", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<MulBackward1> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<MulBackward1>(new MulBackward1(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->other = other;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::mul");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->mul(self_, other));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::mul_(Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("mul_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  check_inplace(self);
  std::shared_ptr<MulBackward0> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::shared_ptr<MulBackward0>(new MulBackward0(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    grad_fn->self_ = SavedVariable(self.clone(), false);
    grad_fn->other_ = SavedVariable(other, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::mul");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("mul_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->mul_(self_, other_);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::mul_(Tensor & self, Scalar other) const {
  profiler::RecordFunction profiler("mul_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<MulBackward1> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<MulBackward1>(new MulBackward1(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->other = other;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::mul");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("mul_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->mul_(self_, other);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor VariableType::multi_margin_loss_forward(const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, int64_t reduction) const {
  profiler::RecordFunction profiler("multi_margin_loss_forward", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& target_ = unpack(target, "target", 1);
  auto weight_ = unpack_opt(weight, "weight", 4);
  check_no_requires_grad(weight, "weight");
  std::shared_ptr<MultiMarginLossBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<MultiMarginLossBackward>(new MultiMarginLossBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->target_ = SavedVariable(target, false);
    grad_fn->p = p;
    grad_fn->margin = margin;
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->reduction = reduction;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::multi_margin_loss_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "margin", margin);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "reduction", reduction);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto output = as_variable(baseType->multi_margin_loss_forward(self_, target_, p, margin, weight_, reduction));
  set_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor VariableType::multilabel_margin_loss(const Tensor & self, const Tensor & target, int64_t reduction) const {
  profiler::RecordFunction profiler("multilabel_margin_loss", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::multilabel_margin_loss");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "reduction", reduction);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto output = TypeDefault::multilabel_margin_loss(self, target, reduction);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor & VariableType::multilabel_margin_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction, const Tensor & is_target) const {
  profiler::RecordFunction profiler("multilabel_margin_loss_backward_out", Function::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& target_ = unpack(target, "target", 3);
  auto& is_target_ = unpack(is_target, "is_target", 5);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self, is_target )) {
    throw_error_out_requires_grad("multilabel_margin_loss_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("multilabel_margin_loss_backward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::multilabel_margin_loss_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "reduction", reduction);
    jit::tracer::addInputs(node, "is_target", is_target);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("multilabel_margin_loss_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->multilabel_margin_loss_backward_out(grad_input_, grad_output_, self_, target_, reduction, is_target_);
  increment_version(grad_input);
  rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor & VariableType::multinomial_out(Tensor & result, const Tensor & self, int64_t num_samples, bool replacement, Generator * generator) const {
  profiler::RecordFunction profiler("multinomial_out", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::multinomial");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "num_samples", num_samples);
    jit::tracer::addInputs(node, "replacement", replacement);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("multinomial_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::multinomial_out(result, self, num_samples, replacement, generator);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::native_pow(const Tensor & self, Scalar exponent) const {
  profiler::RecordFunction profiler("native_pow", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("native_pow"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::native_pow");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "exponent", exponent);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->native_pow(self_, exponent));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::neg(const Tensor & self) const {
  profiler::RecordFunction profiler("neg", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NegBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NegBackward>(new NegBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::neg");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->neg(self_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::neg_(Tensor & self) const {
  profiler::RecordFunction profiler("neg_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<NegBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NegBackward>(new NegBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::neg");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("neg_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->neg_(self_);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::nll_loss2d_out(Tensor & output, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index) const {
  profiler::RecordFunction profiler("nll_loss2d_out", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::nll_loss2d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "reduction", reduction);
    jit::tracer::addInputs(node, "ignore_index", ignore_index);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("nll_loss2d_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::nll_loss2d_out(output, self, target, weight, reduction, ignore_index);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor & VariableType::nll_loss_out(Tensor & output, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index) const {
  profiler::RecordFunction profiler("nll_loss_out", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::nll_loss");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "reduction", reduction);
    jit::tracer::addInputs(node, "ignore_index", ignore_index);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("nll_loss_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::nll_loss_out(output, self, target, weight, reduction, ignore_index);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor VariableType::nonzero(const Tensor & self) const {
  profiler::RecordFunction profiler("nonzero", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::nonzero");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::nonzero(self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::norm(const Tensor & self, Scalar p) const {
  profiler::RecordFunction profiler("norm", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NormBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NormBackward0>(new NormBackward0(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->p = p;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::norm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->norm(self_, p));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  return result;
}
Tensor VariableType::norm(const Tensor & self, Scalar p, int64_t dim, bool keepdim) const {
  profiler::RecordFunction profiler("norm", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NormBackward1> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NormBackward1>(new NormBackward1(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->p = p;
    grad_fn->dim = dim;
    grad_fn->keepdim = keepdim;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::norm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->norm(self_, p, dim, keepdim));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  return result;
}
Tensor VariableType::normal(const Tensor & mean, double std, Generator * generator) const {
  profiler::RecordFunction profiler("normal", Function::peek_at_next_sequence_nr());
  auto& mean_ = unpack(mean, "mean", 0);
  std::shared_ptr<NormalBackward1> grad_fn;
  if (compute_requires_grad( mean )) {
    grad_fn = std::shared_ptr<NormalBackward1>(new NormalBackward1(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( mean ));
    grad_fn->mean_sizes = mean.sizes().vec();
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::normal");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "std", std);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->normal(mean_, std, generator));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::normal(double mean, const Tensor & std, Generator * generator) const {
  profiler::RecordFunction profiler("normal", Function::peek_at_next_sequence_nr());
  auto& std_ = unpack(std, "std", 1);
  std::shared_ptr<NormalBackward2> grad_fn;
  if (compute_requires_grad( std )) {
    grad_fn = std::shared_ptr<NormalBackward2>(new NormalBackward2(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( std ));
    grad_fn->std_sizes = std.sizes().vec();
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::normal");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "std", std);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->normal(mean, std_, generator));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::normal(const Tensor & mean, const Tensor & std, Generator * generator) const {
  profiler::RecordFunction profiler("normal", Function::peek_at_next_sequence_nr());
  auto& mean_ = unpack(mean, "mean", 0);
  auto& std_ = unpack(std, "std", 1);
  std::shared_ptr<NormalBackward3> grad_fn;
  if (compute_requires_grad( mean, std )) {
    grad_fn = std::shared_ptr<NormalBackward3>(new NormalBackward3(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( mean, std ));
    grad_fn->mean_sizes = mean.sizes().vec();
    grad_fn->std_sizes = std.sizes().vec();
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::normal");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "std", std);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->normal(mean_, std_, generator));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::normal_(Tensor & self, double mean, double std, Generator * generator) const {
  profiler::RecordFunction profiler("normal_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<NormalBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NormalBackward0>(new NormalBackward0(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::normal");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "std", std);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("normal_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->normal_(self_, mean, std, generator);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::nuclear_norm_out(Tensor & result, const Tensor & self, bool keepdim) const {
  profiler::RecordFunction profiler("nuclear_norm_out", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::nuclear_norm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("nuclear_norm_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::nuclear_norm_out(result, self, keepdim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::orgqr_out(Tensor & result, const Tensor & self, const Tensor & input2) const {
  profiler::RecordFunction profiler("orgqr_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& input2_ = unpack(input2, "input2", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, input2 )) {
    throw_error_out_requires_grad("orgqr");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("orgqr");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::orgqr");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "input2", input2);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("orgqr_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->orgqr_out(result_, self_, input2_);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::poisson(const Tensor & self, Generator * generator) const {
  profiler::RecordFunction profiler("poisson", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<PoissonBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<PoissonBackward>(new PoissonBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_info = self;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::poisson");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->poisson(self_, generator));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::potrf_out(Tensor & result, const Tensor & self, bool upper) const {
  profiler::RecordFunction profiler("potrf_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("potrf");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("potrf");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::potrf");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "upper", upper);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("potrf_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->potrf_out(result_, self_, upper);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor,Tensor> VariableType::pstrf(const Tensor & self, bool upper, Scalar tol) const {
  profiler::RecordFunction profiler("pstrf", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<PstrfBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<PstrfBackward>(new PstrfBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  Tensor result0;
  Tensor result1;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::pstrf");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "upper", upper);
    jit::tracer::addInputs(node, "tol", tol);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(result0, result1) = as_variable(baseType->pstrf(self_, upper, tol));
  set_history(flatten_tensor_args( result0, result1 ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
Tensor & VariableType::range_out(Tensor & result, Scalar start, Scalar end) const {
  profiler::RecordFunction profiler("range_out", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::range");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "start", start);
    jit::tracer::addInputs(node, "end", end);
    jit::tracer::addInputs(node, "result", result.options());
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("range_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::range_out(result, start, end);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::range_out(Tensor & result, Scalar start, Scalar end, Scalar step) const {
  profiler::RecordFunction profiler("range_out", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::range");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "start", start);
    jit::tracer::addInputs(node, "end", end);
    jit::tracer::addInputs(node, "step", step);
    jit::tracer::addInputs(node, "result", result.options());
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("range_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::range_out(result, start, end, step);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::reciprocal_out(Tensor & result, const Tensor & self) const {
  profiler::RecordFunction profiler("reciprocal_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("reciprocal");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("reciprocal");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::reciprocal");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("reciprocal_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->reciprocal_out(result_, self_);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::reflection_pad1d_out(Tensor & output, const Tensor & self, IntList padding) const {
  profiler::RecordFunction profiler("reflection_pad1d_out", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::reflection_pad1d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "padding", padding);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("reflection_pad1d_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::reflection_pad1d_out(output, self, padding);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor VariableType::renorm(const Tensor & self, Scalar p, int64_t dim, Scalar maxnorm) const {
  profiler::RecordFunction profiler("renorm", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<RenormBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<RenormBackward>(new RenormBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->p = p;
    grad_fn->dim = dim;
    grad_fn->maxnorm = maxnorm;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::renorm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "maxnorm", maxnorm);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->renorm(self_, p, dim, maxnorm));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::renorm_(Tensor & self, Scalar p, int64_t dim, Scalar maxnorm) const {
  profiler::RecordFunction profiler("renorm_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<RenormBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<RenormBackward>(new RenormBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self.clone(), false);
    grad_fn->p = p;
    grad_fn->dim = dim;
    grad_fn->maxnorm = maxnorm;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::renorm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "maxnorm", maxnorm);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("renorm_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->renorm_(self_, p, dim, maxnorm);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor VariableType::replication_pad2d_forward(const Tensor & self, IntList padding) const {
  profiler::RecordFunction profiler("replication_pad2d_forward", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<ReplicationPad2DBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<ReplicationPad2DBackward>(new ReplicationPad2DBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->padding = padding.vec();
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::replication_pad2d_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "padding", padding);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto output = as_variable(baseType->replication_pad2d_forward(self_, padding));
  set_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor VariableType::replication_pad3d_backward(const Tensor & grad_output, const Tensor & self, IntList padding) const {
  profiler::RecordFunction profiler("replication_pad3d_backward", Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<ReplicationPad3DBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::shared_ptr<ReplicationPad3DBackwardBackward>(new ReplicationPad3DBackwardBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->padding = padding.vec();
    grad_fn->self_info = self;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::replication_pad3d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "padding", padding);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto grad_input = as_variable(baseType->replication_pad3d_backward(grad_output_, self_, padding));
  set_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor & VariableType::replication_pad3d_forward_out(Tensor & output, const Tensor & self, IntList padding) const {
  profiler::RecordFunction profiler("replication_pad3d_forward_out", Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("replication_pad3d_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("replication_pad3d_forward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::replication_pad3d_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "padding", padding);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("replication_pad3d_forward_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->replication_pad3d_forward_out(output_, self_, padding);
  increment_version(output);
  rebase_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor VariableType::reshape(const Tensor & self, IntList shape) const {
  profiler::RecordFunction profiler("reshape", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::reshape");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "shape", shape);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::reshape(self, shape);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::rfft(const Tensor & self, int64_t signal_ndim, bool normalized, bool onesided) const {
  profiler::RecordFunction profiler("rfft", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::rfft");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "signal_ndim", signal_ndim);
    jit::tracer::addInputs(node, "normalized", normalized);
    jit::tracer::addInputs(node, "onesided", onesided);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::rfft(self, signal_ndim, normalized, onesided);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor,Tensor> VariableType::rnn_relu(const Tensor & input, const Tensor & hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first) const {
  profiler::RecordFunction profiler("rnn_relu", Function::peek_at_next_sequence_nr());
  Tensor result0;
  Tensor result1;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::rnn_relu");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "hx", hx);
    jit::tracer::addInputs(node, "params", params);
    jit::tracer::addInputs(node, "has_biases", has_biases);
    jit::tracer::addInputs(node, "num_layers", num_layers);
    jit::tracer::addInputs(node, "dropout", dropout);
    jit::tracer::addInputs(node, "train", train);
    jit::tracer::addInputs(node, "bidirectional", bidirectional);
    jit::tracer::addInputs(node, "batch_first", batch_first);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(result0, result1) = TypeDefault::rnn_relu(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
std::tuple<Tensor,Tensor> VariableType::rnn_relu(const Tensor & data, const Tensor & batch_sizes, const Tensor & hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional) const {
  profiler::RecordFunction profiler("rnn_relu", Function::peek_at_next_sequence_nr());
  Tensor result0;
  Tensor result1;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::rnn_relu");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "data", data);
    jit::tracer::addInputs(node, "batch_sizes", batch_sizes);
    jit::tracer::addInputs(node, "hx", hx);
    jit::tracer::addInputs(node, "params", params);
    jit::tracer::addInputs(node, "has_biases", has_biases);
    jit::tracer::addInputs(node, "num_layers", num_layers);
    jit::tracer::addInputs(node, "dropout", dropout);
    jit::tracer::addInputs(node, "train", train);
    jit::tracer::addInputs(node, "bidirectional", bidirectional);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(result0, result1) = TypeDefault::rnn_relu(data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
Tensor VariableType::rrelu(const Tensor & self, Scalar lower, Scalar upper, bool training, Generator * generator) const {
  profiler::RecordFunction profiler("rrelu", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::rrelu");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "lower", lower);
    jit::tracer::addInputs(node, "upper", upper);
    jit::tracer::addInputs(node, "training", training);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::rrelu(self, lower, upper, training, generator);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::rrelu_(Tensor & self, Scalar lower, Scalar upper, bool training, Generator * generator) const {
  profiler::RecordFunction profiler("rrelu_", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::rrelu");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "lower", lower);
    jit::tracer::addInputs(node, "upper", upper);
    jit::tracer::addInputs(node, "training", training);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("rrelu_", self);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::rrelu_(self, lower, upper, training, generator);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::rrelu_with_noise_out(Tensor & output, const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, Generator * generator) const {
  profiler::RecordFunction profiler("rrelu_with_noise_out", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::rrelu_with_noise");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "noise", noise);
    jit::tracer::addInputs(node, "lower", lower);
    jit::tracer::addInputs(node, "upper", upper);
    jit::tracer::addInputs(node, "training", training);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("rrelu_with_noise_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::rrelu_with_noise_out(output, self, noise, lower, upper, training, generator);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor & VariableType::sin_out(Tensor & result, const Tensor & self) const {
  profiler::RecordFunction profiler("sin_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("sin");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("sin");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::sin");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("sin_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->sin_out(result_, self_);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::sinh(const Tensor & self) const {
  profiler::RecordFunction profiler("sinh", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<SinhBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<SinhBackward>(new SinhBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::sinh");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->sinh(self_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::sinh_(Tensor & self) const {
  profiler::RecordFunction profiler("sinh_", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<SinhBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<SinhBackward>(new SinhBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self.clone(), false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::sinh");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("sinh_", self);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->sinh_(self_);
  increment_version(self);
  rebase_history(flatten_tensor_args( self ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
std::tuple<Tensor,Tensor> VariableType::slogdet(const Tensor & self) const {
  profiler::RecordFunction profiler("slogdet", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<SlogdetBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<SlogdetBackward>(new SlogdetBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  Tensor result0;
  Tensor result1;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::slogdet");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(result0, result1) = as_variable(baseType->slogdet(self_));
  set_history(flatten_tensor_args( result0, result1 ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  if (grad_fn) {
    grad_fn->result0_ = SavedVariable(result0, true);
    grad_fn->result1_ = SavedVariable(result1, true);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
Tensor VariableType::smooth_l1_loss_forward(const Tensor & self, const Tensor & target, int64_t reduction) const {
  profiler::RecordFunction profiler("smooth_l1_loss_forward", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& target_ = unpack(target, "target", 1);
  check_no_requires_grad(target, "target");
  std::shared_ptr<SmoothL1LossBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<SmoothL1LossBackward>(new SmoothL1LossBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->target_ = SavedVariable(target, false);
    grad_fn->reduction = reduction;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::smooth_l1_loss_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "reduction", reduction);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto output = as_variable(baseType->smooth_l1_loss_forward(self_, target_, reduction));
  set_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor VariableType::softshrink(const Tensor & self, Scalar lambd) const {
  profiler::RecordFunction profiler("softshrink", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::softshrink");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "lambd", lambd);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto output = TypeDefault::softshrink(self, lambd);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor & VariableType::softshrink_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar lambd) const {
  profiler::RecordFunction profiler("softshrink_backward_out", Function::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    throw_error_out_requires_grad("softshrink_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("softshrink_backward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::softshrink_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "lambd", lambd);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("softshrink_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->softshrink_backward_out(grad_input_, grad_output_, self_, lambd);
  increment_version(grad_input);
  rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor VariableType::stack(TensorList tensors, int64_t dim) const {
  profiler::RecordFunction profiler("stack", Function::peek_at_next_sequence_nr());
  auto tensors_ = unpack(tensors, "tensors", 0);
  std::shared_ptr<StackBackward> grad_fn;
  if (compute_requires_grad( tensors )) {
    grad_fn = std::shared_ptr<StackBackward>(new StackBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( tensors ));
    grad_fn->dim = dim;
    grad_fn->tensors_size_ = tensors.size();
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::stack");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "tensors", tensors);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->stack(tensors_, dim));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::stft(const Tensor & self, int64_t n_fft, int64_t hop_length, int64_t win_length, const Tensor & window, bool normalized, bool onesided) const {
  profiler::RecordFunction profiler("stft", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::stft");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "n_fft", n_fft);
    jit::tracer::addInputs(node, "hop_length", hop_length);
    jit::tracer::addInputs(node, "win_length", win_length);
    jit::tracer::addInputs(node, "window", window);
    jit::tracer::addInputs(node, "normalized", normalized);
    jit::tracer::addInputs(node, "onesided", onesided);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::stft(self, n_fft, hop_length, win_length, window, normalized, onesided);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
int64_t VariableType::storage_offset(const Tensor & self) const {
  auto result = TypeDefault::storage_offset(self);
  return result;
}
Tensor & VariableType::sub_out(Tensor & result, const Tensor & self, const Tensor & other, Scalar alpha) const {
  profiler::RecordFunction profiler("sub_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& other_ = unpack(other, "other", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, other )) {
    throw_error_out_requires_grad("sub");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("sub");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::sub");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("sub_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->sub_out(result_, self_, other_, alpha);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor,Tensor> VariableType::symeig(const Tensor & self, bool eigenvectors, bool upper) const {
  profiler::RecordFunction profiler("symeig", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<SymeigBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<SymeigBackward>(new SymeigBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->eigenvectors = eigenvectors;
    grad_fn->upper = upper;
  }
  Tensor result0;
  Tensor result1;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::symeig");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "eigenvectors", eigenvectors);
    jit::tracer::addInputs(node, "upper", upper);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(result0, result1) = as_variable(baseType->symeig(self_, eigenvectors, upper));
  set_history(flatten_tensor_args( result0, result1 ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  if (grad_fn) {
    grad_fn->result0_ = SavedVariable(result0, true);
    grad_fn->result1_ = SavedVariable(result1, true);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
Tensor VariableType::tensordot(const Tensor & self, const Tensor & other, IntList dims_self, IntList dims_other) const {
  profiler::RecordFunction profiler("tensordot", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::tensordot");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "dims_self", dims_self);
    jit::tracer::addInputs(node, "dims_other", dims_other);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::tensordot(self, other, dims_self, dims_other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::s_th_addmm_out(Tensor & result, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
  profiler::RecordFunction profiler("th_addmm_out", Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& mat1_ = unpack(mat1, "mat1", 2);
  auto& mat2_ = unpack(mat2, "mat2", 3);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, mat1, mat2 )) {
    throw_error_out_requires_grad("th_addmm");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("th_addmm");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::addmm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mat1", mat1);
    jit::tracer::addInputs(node, "mat2", mat2);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("th_addmm_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->s_th_addmm_out(result_, self_, mat1_, mat2_, beta, alpha);
  increment_version(result);
  rebase_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor,Tensor,Tensor> VariableType::thnn_conv2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, const Tensor & finput, const Tensor & fgrad_input, std::array<bool,3> output_mask) const {
  profiler::RecordFunction profiler("thnn_conv2d_backward", Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& weight_ = unpack(weight, "weight", 2);
  auto& finput_ = unpack(finput, "finput", 6);
  auto& fgrad_input_ = unpack(fgrad_input, "fgrad_input", 7);
  check_no_requires_grad(finput, "finput");
  check_no_requires_grad(fgrad_input, "fgrad_input");
  std::shared_ptr<ThnnConv2DBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self, weight )) {
    grad_fn = std::shared_ptr<ThnnConv2DBackwardBackward>(new ThnnConv2DBackwardBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self, weight ));
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->stride = stride.vec();
    grad_fn->padding = padding.vec();
  }
  Tensor grad_input;
  Tensor grad_weight;
  Tensor grad_bias;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::thnn_conv2d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "finput", finput);
    jit::tracer::addInputs(node, "fgrad_input", fgrad_input);
    jit::tracer::addInputs(node, "output_mask", output_mask);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(grad_input, grad_weight, grad_bias) = as_variable(baseType->thnn_conv2d_backward(grad_output_, self_, weight_, kernel_size, stride, padding, finput_, fgrad_input_, output_mask));
  set_history(flatten_tensor_args( grad_input, grad_weight, grad_bias ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
    jit::tracer::addOutput(node, grad_weight);
    jit::tracer::addOutput(node, grad_bias);
  }
  return std::make_tuple(std::move(grad_input), std::move(grad_weight), std::move(grad_bias));
}
std::tuple<Tensor &,Tensor &,Tensor &> VariableType::thnn_conv2d_forward_out(Tensor & output, Tensor & finput, Tensor & fgrad_input, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding) const {
  profiler::RecordFunction profiler("thnn_conv2d_forward_out", Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& finput_ = unpack(finput, "finput", 1);
  auto& fgrad_input_ = unpack(fgrad_input, "fgrad_input", 2);
  auto& self_ = unpack(self, "self", 3);
  auto& weight_ = unpack(weight, "weight", 4);
  auto bias_ = unpack_opt(bias, "bias", 6);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, weight, bias )) {
    throw_error_out_requires_grad("thnn_conv2d_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("thnn_conv2d_forward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::thnn_conv2d_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "finput", finput);
    jit::tracer::addInputs(node, "fgrad_input", fgrad_input);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("thnn_conv2d_forward_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->thnn_conv2d_forward_out(output_, finput_, fgrad_input_, self_, weight_, kernel_size, bias_, stride, padding);
  increment_version(output);
  rebase_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
    jit::tracer::addOutput(node, finput);
    jit::tracer::addOutput(node, fgrad_input);
  }
  return std::forward_as_tuple(output, finput, fgrad_input);
}
Tensor VariableType::thnn_conv3d(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding) const {
  profiler::RecordFunction profiler("thnn_conv3d", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::thnn_conv3d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto output = TypeDefault::thnn_conv3d(self, weight, kernel_size, bias, stride, padding);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
std::tuple<Tensor &,Tensor &,Tensor &> VariableType::thnn_conv3d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, const Tensor & finput, const Tensor & fgrad_input) const {
  profiler::RecordFunction profiler("thnn_conv3d_backward_out", Function::peek_at_next_sequence_nr());
  auto grad_input_ = unpack_opt(grad_input, "grad_input", 0);
  auto grad_weight_ = unpack_opt(grad_weight, "grad_weight", 1);
  auto grad_bias_ = unpack_opt(grad_bias, "grad_bias", 2);
  auto& grad_output_ = unpack(grad_output, "grad_output", 3);
  auto& self_ = unpack(self, "self", 4);
  auto& weight_ = unpack(weight, "weight", 5);
  auto& finput_ = unpack(finput, "finput", 9);
  auto& fgrad_input_ = unpack(fgrad_input, "fgrad_input", 10);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self, weight, finput, fgrad_input )) {
    throw_error_out_requires_grad("thnn_conv3d_backward");
  }
  if (compute_requires_grad( grad_input, grad_weight, grad_bias )) {
    throw_error_out_requires_grad("thnn_conv3d_backward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::thnn_conv3d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_weight", grad_weight);
    jit::tracer::addInputs(node, "grad_bias", grad_bias);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "finput", finput);
    jit::tracer::addInputs(node, "fgrad_input", fgrad_input);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("thnn_conv3d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->thnn_conv3d_backward_out(grad_input_, grad_weight_, grad_bias_, grad_output_, self_, weight_, kernel_size, stride, padding, finput_, fgrad_input_);
  increment_version(grad_input);
  increment_version(grad_weight);
  increment_version(grad_bias);
  rebase_history(flatten_tensor_args( grad_input, grad_weight, grad_bias ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
    jit::tracer::addOutput(node, grad_weight);
    jit::tracer::addOutput(node, grad_bias);
  }
  return std::forward_as_tuple(grad_input, grad_weight, grad_bias);
}
Tensor & VariableType::thnn_conv_dilated2d_out(Tensor & output, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) const {
  profiler::RecordFunction profiler("thnn_conv_dilated2d_out", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::thnn_conv_dilated2d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("thnn_conv_dilated2d_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::thnn_conv_dilated2d_out(output, self, weight, kernel_size, bias, stride, padding, dilation);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor VariableType::thnn_conv_transpose2d(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation) const {
  profiler::RecordFunction profiler("thnn_conv_transpose2d", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::thnn_conv_transpose2d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "output_padding", output_padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto output = TypeDefault::thnn_conv_transpose2d(self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
std::tuple<Tensor &,Tensor &,Tensor &> VariableType::thnn_conv_transpose2d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList output_padding, IntList dilation, const Tensor & columns, const Tensor & ones) const {
  profiler::RecordFunction profiler("thnn_conv_transpose2d_backward_out", Function::peek_at_next_sequence_nr());
  auto grad_input_ = unpack_opt(grad_input, "grad_input", 0);
  auto grad_weight_ = unpack_opt(grad_weight, "grad_weight", 1);
  auto grad_bias_ = unpack_opt(grad_bias, "grad_bias", 2);
  auto& grad_output_ = unpack(grad_output, "grad_output", 3);
  auto& self_ = unpack(self, "self", 4);
  auto& weight_ = unpack(weight, "weight", 5);
  auto& columns_ = unpack(columns, "columns", 11);
  auto& ones_ = unpack(ones, "ones", 12);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self, weight, columns, ones )) {
    throw_error_out_requires_grad("thnn_conv_transpose2d_backward");
  }
  if (compute_requires_grad( grad_input, grad_weight, grad_bias )) {
    throw_error_out_requires_grad("thnn_conv_transpose2d_backward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::thnn_conv_transpose2d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_weight", grad_weight);
    jit::tracer::addInputs(node, "grad_bias", grad_bias);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "output_padding", output_padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "columns", columns);
    jit::tracer::addInputs(node, "ones", ones);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("thnn_conv_transpose2d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->thnn_conv_transpose2d_backward_out(grad_input_, grad_weight_, grad_bias_, grad_output_, self_, weight_, kernel_size, stride, padding, output_padding, dilation, columns_, ones_);
  increment_version(grad_input);
  increment_version(grad_weight);
  increment_version(grad_bias);
  rebase_history(flatten_tensor_args( grad_input, grad_weight, grad_bias ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
    jit::tracer::addOutput(node, grad_weight);
    jit::tracer::addOutput(node, grad_bias);
  }
  return std::forward_as_tuple(grad_input, grad_weight, grad_bias);
}
Tensor & VariableType::thnn_conv_transpose3d_out(Tensor & output, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation) const {
  profiler::RecordFunction profiler("thnn_conv_transpose3d_out", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::thnn_conv_transpose3d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "output_padding", output_padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("thnn_conv_transpose3d_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::thnn_conv_transpose3d_out(output, self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor VariableType::threshold_backward(const Tensor & grad_output, const Tensor & self, Scalar threshold, Scalar value) const {
  profiler::RecordFunction profiler("threshold_backward", Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<ThresholdBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::shared_ptr<ThresholdBackwardBackward>(new ThresholdBackwardBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->threshold = threshold;
    grad_fn->value = value;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::threshold_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "threshold", threshold);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto grad_input = as_variable(baseType->threshold_backward(grad_output_, self_, threshold, value));
  set_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor & VariableType::threshold_forward_out(Tensor & output, const Tensor & self, Scalar threshold, Scalar value) const {
  profiler::RecordFunction profiler("threshold_forward_out", Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("threshold_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("threshold_forward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::threshold_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "threshold", threshold);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("threshold_forward_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->threshold_forward_out(output_, self_, threshold, value);
  increment_version(output);
  rebase_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor VariableType::to_dense(const Tensor & self) const {
  profiler::RecordFunction profiler("to_dense", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("to_dense"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::to_dense");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->to_dense(self_));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::triplet_margin_loss(const Tensor & anchor, const Tensor & positive, const Tensor & negative, double margin, double p, double eps, bool swap, int64_t reduction) const {
  profiler::RecordFunction profiler("triplet_margin_loss", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::triplet_margin_loss");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "anchor", anchor);
    jit::tracer::addInputs(node, "positive", positive);
    jit::tracer::addInputs(node, "negative", negative);
    jit::tracer::addInputs(node, "margin", margin);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "eps", eps);
    jit::tracer::addInputs(node, "swap", swap);
    jit::tracer::addInputs(node, "reduction", reduction);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::triplet_margin_loss(anchor, positive, negative, margin, p, eps, swap, reduction);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor &,Tensor &> VariableType::trtrs_out(Tensor & X, Tensor & M, const Tensor & self, const Tensor & A, bool upper, bool transpose, bool unitriangular) const {
  profiler::RecordFunction profiler("trtrs_out", Function::peek_at_next_sequence_nr());
  auto& X_ = unpack(X, "X", 0);
  auto& M_ = unpack(M, "M", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& A_ = unpack(A, "A", 3);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, A )) {
    throw_error_out_requires_grad("trtrs");
  }
  if (compute_requires_grad( X, M )) {
    throw_error_out_requires_grad("trtrs");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::trtrs");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "M", M);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "A", A);
    jit::tracer::addInputs(node, "upper", upper);
    jit::tracer::addInputs(node, "transpose", transpose);
    jit::tracer::addInputs(node, "unitriangular", unitriangular);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("trtrs_out", X);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->trtrs_out(X_, M_, self_, A_, upper, transpose, unitriangular);
  increment_version(X);
  increment_version(M);
  rebase_history(flatten_tensor_args( X, M ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, X);
    jit::tracer::addOutput(node, M);
  }
  return std::forward_as_tuple(X, M);
}
Tensor VariableType::upsample_bilinear2d(const Tensor & self, IntList output_size, bool align_corners) const {
  profiler::RecordFunction profiler("upsample_bilinear2d", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::upsample_bilinear2d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "align_corners", align_corners);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto output = TypeDefault::upsample_bilinear2d(self, output_size, align_corners);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor & VariableType::upsample_bilinear2d_backward_out(Tensor & grad_input, const Tensor & grad_output, IntList output_size, IntList input_size, bool align_corners) const {
  profiler::RecordFunction profiler("upsample_bilinear2d_backward_out", Function::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output )) {
    throw_error_out_requires_grad("upsample_bilinear2d_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("upsample_bilinear2d_backward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::upsample_bilinear2d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "input_size", input_size);
    jit::tracer::addInputs(node, "align_corners", align_corners);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("upsample_bilinear2d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->upsample_bilinear2d_backward_out(grad_input_, grad_output_, output_size, input_size, align_corners);
  increment_version(grad_input);
  rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor & VariableType::upsample_linear1d_out(Tensor & output, const Tensor & self, IntList output_size, bool align_corners) const {
  profiler::RecordFunction profiler("upsample_linear1d_out", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::upsample_linear1d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "align_corners", align_corners);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("upsample_linear1d_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::upsample_linear1d_out(output, self, output_size, align_corners);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor VariableType::upsample_nearest1d(const Tensor & self, IntList output_size) const {
  profiler::RecordFunction profiler("upsample_nearest1d", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::upsample_nearest1d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto output = TypeDefault::upsample_nearest1d(self, output_size);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor & VariableType::upsample_nearest1d_backward_out(Tensor & grad_input, const Tensor & grad_output, IntList output_size, IntList input_size) const {
  profiler::RecordFunction profiler("upsample_nearest1d_backward_out", Function::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output )) {
    throw_error_out_requires_grad("upsample_nearest1d_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("upsample_nearest1d_backward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::upsample_nearest1d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "input_size", input_size);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("upsample_nearest1d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  baseType->upsample_nearest1d_backward_out(grad_input_, grad_output_, output_size, input_size);
  increment_version(grad_input);
  rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor & VariableType::upsample_nearest2d_out(Tensor & output, const Tensor & self, IntList output_size) const {
  profiler::RecordFunction profiler("upsample_nearest2d_out", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::upsample_nearest2d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);
    tracer_state->graph->appendNode(node);
    jit::tracer::ensureUnique("upsample_nearest2d_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::upsample_nearest2d_out(output, self, output_size);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor VariableType::upsample_trilinear3d_forward(const Tensor & self, IntList output_size, bool align_corners) const {
  profiler::RecordFunction profiler("upsample_trilinear3d_forward", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<UpsampleTrilinear3DBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<UpsampleTrilinear3DBackward>(new UpsampleTrilinear3DBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes().vec();
    grad_fn->output_size = output_size.vec();
    grad_fn->align_corners = align_corners;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::upsample_trilinear3d_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "align_corners", align_corners);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto output = as_variable(baseType->upsample_trilinear3d_forward(self_, output_size, align_corners));
  set_history(flatten_tensor_args( output ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor VariableType::var(const Tensor & self, bool unbiased) const {
  profiler::RecordFunction profiler("var", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<VarBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<VarBackward0>(new VarBackward0(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->unbiased = unbiased;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::var");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "unbiased", unbiased);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->var(self_, unbiased));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::var(const Tensor & self, int64_t dim, bool unbiased, bool keepdim) const {
  profiler::RecordFunction profiler("var", Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<VarBackward1> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<VarBackward1>(new VarBackward1(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->dim = dim;
    grad_fn->unbiased = unbiased;
    grad_fn->keepdim = keepdim;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::var");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "unbiased", unbiased);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = as_variable(baseType->var(self_, dim, unbiased, keepdim));
  set_history(flatten_tensor_args( result ), grad_fn);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::where(const Tensor & condition, const Tensor & self, const Tensor & other) const {
  profiler::RecordFunction profiler("where", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::where");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "condition", condition);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::where(condition, self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::zeros_like(const Tensor & self) const {
  profiler::RecordFunction profiler("zeros_like", Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    const static auto op_name = jit::Symbol::fromQualString("aten::zeros_like");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->appendNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::zeros_like(self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}

}} // namespace torch::autograd
