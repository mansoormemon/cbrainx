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

// /////////////////////////////////////////////
// Query Functions
// /////////////////////////////////////////////

auto Optimizer::iterations() const -> u32 { return iterations_; }

// /////////////////////////////////////////////
// Informative
// /////////////////////////////////////////////

auto Optimizer::meta_info() -> std::string { return fmt::format("{{ iterations = {} }}", iterations_); }

// /////////////////////////////////////////////
// Interface
// /////////////////////////////////////////////

auto Optimizer::operator++() -> Optimizer & {
  ++iterations_;
  return *this;
};

auto Optimizer::reset() -> Optimizer & {
  iterations_ = {};
  return *this;
}

// /////////////////////////////////////////////
// Constructors (and Destructors)
// /////////////////////////////////////////////

GradientDescent::GradientDescent(f32 learning_rate, f32 decay_rate)
    : learning_rate_{learning_rate}, alpha_{learning_rate}, decay_rate_{decay_rate} {}

// /////////////////////////////////////////////
// Informative
// /////////////////////////////////////////////

auto GradientDescent::meta_info() -> std::string {
  return fmt::format("{{ iterations={}, learning_rate={}, alpha={}, decay_rate={} }}", iterations_,
                     learning_rate_, alpha_, decay_rate_);
}

// /////////////////////////////////////////////
// Interface
// /////////////////////////////////////////////

auto GradientDescent::operator++() -> Optimizer & {
  // With each iteration, the learning rate decays using the following rule.
  //
  // Rule: ⍺ = ȹ . 1 / (1 + Ɣ . ὶ)
  //
  // where:
  //  ⍺ - New learning rate
  //  ȹ - Initial learning rate
  //  Ɣ - Decay rate
  //  ὶ - Iterations

  Optimizer::operator++();
  alpha_ = learning_rate_ * (1 / (1 + decay_rate_ * iterations_));
  return *this;
}

auto GradientDescent::reset() -> Optimizer & {
  Optimizer::reset();
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

}
