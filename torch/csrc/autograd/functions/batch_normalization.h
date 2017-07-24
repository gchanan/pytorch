#pragma once

#include <Python.h>
#include <memory>
#include <THPP/THPP.h>

#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/variable.h"

namespace torch { namespace autograd {

struct BatchNormParams {
  std::shared_ptr<thpp::Tensor> running_mean;
  std::shared_ptr<thpp::Tensor> running_var;
  bool training;
  double momentum;
  double eps;
  bool cudnn_enabled;
};

struct BatchNormForward : public Function, public BatchNormParams {
  BatchNormForward(BatchNormParams params)
    : BatchNormParams(std::move(params)) {}

  virtual variable_list apply(const variable_list& inputs) override;
};

struct BatchNormBackward : public Function, public BatchNormParams {
  BatchNormBackward(
      FunctionFlags flags,
      BatchNormParams params,
      std::unique_ptr<thpp::Tensor> save_mean,
      std::unique_ptr<thpp::Tensor> save_std,
      SavedVariable input,
      SavedVariable weight,
      SavedVariable bias)
    : Function(std::move(flags))
    , BatchNormParams(std::move(params)) {
      if (is_executable) {
        this->save_mean_ = std::move(save_mean);
        this->save_std_ = std::move(save_std);
        this->input_ = std::move(input);
        this->weight_ = std::move(weight);
        this->bias_ = std::move(bias);
      }
    }

  virtual variable_list apply(const variable_list& gradOutputs) override;

  virtual void releaseVariables() override;

  std::unique_ptr<thpp::Tensor> save_mean_;
  std::unique_ptr<thpp::Tensor> save_std_;
  SavedVariable input_;
  SavedVariable weight_;
  SavedVariable bias_;
};

struct BatchNormBackwardBackward : public Function, public BatchNormParams {
  BatchNormBackwardBackward(
      FunctionFlags flags,
      BatchNormParams params,
      std::unique_ptr<thpp::Tensor> save_mean,
      std::unique_ptr<thpp::Tensor> save_std,
      SavedVariable input,
      SavedVariable weight,
      SavedVariable bias,
      SavedVariable grad_output)
    : Function(std::move(flags))
    , BatchNormParams(std::move(params)) {
      if (is_executable) {
        this->save_mean_ = std::move(save_mean);
        this->save_std_ = std::move(save_std);
        this->input_ = std::move(input);
        this->weight_ = std::move(weight);
        this->bias_ = std::move(bias);
        this->grad_output_ = std::move(grad_output);
      }
    }

  virtual variable_list apply(const variable_list& grad_grad_inputs) override;

  virtual void releaseVariables() override;

  std::unique_ptr<thpp::Tensor> save_mean_;
  std::unique_ptr<thpp::Tensor> save_std_;
  SavedVariable input_;
  SavedVariable weight_;
  SavedVariable bias_;
  SavedVariable grad_output_;
};

}}
