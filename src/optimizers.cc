// Copyright 2021 CBrainX
// Project URL: https://github.com/mansoormemon/cbrainx
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Copyright (c) 2021 Mansoor Ahmed Memon <mansoorahmed.one@gmail.com>

#include "cbrainx/optimizers.hh"

#include <fmt/core.h>

namespace cbx {

// /////////////////////////////////////////////////////////////
// AbstractOptimizer
// /////////////////////////////////////////////////////////////

// /////////////////////////////////////////////
// Query Functions
// /////////////////////////////////////////////

auto AbstractOptimizer::iterations() const -> u32 { return iterations_; }

// /////////////////////////////////////////////
// Informative
// /////////////////////////////////////////////

auto AbstractOptimizer::meta_info() const -> std::string {
  return fmt::format("{{ iterations = {} }}", iterations_);
}

// /////////////////////////////////////////////
// Interface
// /////////////////////////////////////////////

auto AbstractOptimizer::operator++() -> AbstractOptimizer & {
  ++iterations_;
  return *this;
};

auto AbstractOptimizer::reset() -> AbstractOptimizer & {
  iterations_ = {};
  return *this;
}

// /////////////////////////////////////////////////////////////
// Optimizers
// /////////////////////////////////////////////////////////////

// /////////////////////////////////////////////
// Constructors (and Destructors)
// /////////////////////////////////////////////

GradientDescent::GradientDescent(f32 learning_rate, f32 decay_rate)
    : learning_rate_{learning_rate}, alpha_{learning_rate}, decay_rate_{decay_rate} {}

// /////////////////////////////////////////////
// Query Functions
// /////////////////////////////////////////////

auto GradientDescent::type() const -> Optimizer { return Optimizer::GradientDescent; }

// /////////////////////////////////////////////
// Informative
// /////////////////////////////////////////////

auto GradientDescent::property() const -> std::string {
  return fmt::format("Initial={}, Alpha={}, Decay={}", learning_rate_, alpha_, decay_rate_);
}

auto GradientDescent::to_string() const -> std::string { return "Gradient Descent"; }

auto GradientDescent::type_name() const -> std::string { return "GradientDescent"; }

auto GradientDescent::meta_info() const -> std::string {
  return fmt::format("{{ iterations={}, learning_rate={}, alpha={}, decay_rate={} }}", iterations_,
                     learning_rate_, alpha_, decay_rate_);
}

// /////////////////////////////////////////////
// Interface
// /////////////////////////////////////////////

auto GradientDescent::operator++() -> AbstractOptimizer & {
  // With each iteration, the learning rate decays using the following rule.
  //
  // Rule: ⍺ = ȹ . 1 / (1 + Ɣ . ὶ)
  //
  // where:
  //  ⍺ - New learning rate
  //  ȹ - Initial learning rate
  //  Ɣ - Decay rate
  //  ὶ - Iterations

  AbstractOptimizer::operator++();
  alpha_ = learning_rate_ * (1 / (1 + decay_rate_ * iterations_));
  return *this;
}

auto GradientDescent::reset() -> AbstractOptimizer & {
  AbstractOptimizer::reset();
  alpha_ = learning_rate_;
  return *this;
}

auto GradientDescent::update_params(tensor_type &params, const tensor_type &gradient) -> void {
  // The parameters are updated using the following rule.
  //
  // Formula: Ŵ = Ŵ - ⍺ . ẟ / ẟŴ Ĺ
  //
  // where:
  //  Ŵ        - Trainable parameters
  //  ⍺        - Learning rate
  //  ẟ / ẟŴ Ĺ - Gradient of the loss function w.r.t. Ŵ

  params -= alpha_ * gradient;
}

// /////////////////////////////////////////////////////////////
// OptimizerWrapper
// /////////////////////////////////////////////////////////////

// /////////////////////////////////////////////
// Query Functions
// /////////////////////////////////////////////

auto OptimizerWrapper::iterations() const -> u32 { return optimizer_->iterations(); }

auto OptimizerWrapper::type() const -> Optimizer { return optimizer_->type(); }

auto OptimizerWrapper::is_null() const -> bool { return optimizer_ == nullptr; }

// /////////////////////////////////////////////
// Informative
// /////////////////////////////////////////////

auto OptimizerWrapper::property() const -> std::string { return optimizer_->property(); }

auto OptimizerWrapper::to_string() const -> std::string { return optimizer_->to_string(); }

auto OptimizerWrapper::type_name() const -> std::string { return optimizer_->type_name(); }

auto OptimizerWrapper::meta_info() const -> std::string { return optimizer_->meta_info(); }

// /////////////////////////////////////////////
// Wrapper Interface
// /////////////////////////////////////////////

auto OptimizerWrapper::operator++() -> OptimizerWrapper & {
  optimizer_->operator++();
  return *this;
}

auto OptimizerWrapper::reset() -> OptimizerWrapper & {
  optimizer_->reset();
  return *this;
}

auto OptimizerWrapper::update_params(tensor_type &params, const tensor_type &gradient) -> void {
  optimizer_->update_params(params, gradient);
}

}
