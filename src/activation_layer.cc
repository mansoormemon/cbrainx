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

#include "cbrainx/activation_layer.hh"

#include <utility>

namespace cbx {

// /////////////////////////////////////////////
// Constructors and Destructors
// /////////////////////////////////////////////

ActivationLayer::ActivationLayer(size_type inputs, Activation activation)
    : AbstractLayer{"ACTL"}, neurons_{inputs}, act_func_{activation} {}

ActivationLayer::ActivationLayer(ActivationLayer &&other) noexcept
    : neurons_{std::exchange(other.neurons_, {})}, act_func_{std::move(other.act_func_)} {}

// /////////////////////////////////////////////
// Assignment Operators
// /////////////////////////////////////////////

auto ActivationLayer::operator=(ActivationLayer &&other) noexcept -> ActivationLayer & {
  neurons_ = std::exchange(other.neurons_, {});
  act_func_ = std::move(other.act_func_);
  return *this;
}

// /////////////////////////////////////////////
// Query Functions
// /////////////////////////////////////////////

auto ActivationLayer::neurons() const -> size_type { return neurons_; }

auto ActivationLayer::parameters() const -> size_type { return {}; }

auto ActivationLayer::type() const -> LayerType { return LayerType::Activation; }

// /////////////////////////////////////////////
// Informative
// /////////////////////////////////////////////

auto ActivationLayer::property() const -> std::string { return act_func_.to_string(); }

// /////////////////////////////////////////////
// Core Functionality
// /////////////////////////////////////////////

auto ActivationLayer::forward_pass(const container &input) -> container {
  // Formula: Ô = ζ(Î)
  //
  // where:
  //  ζ - Activation function
  //  Î - Input (Matrix)  : Shape => (m, n)
  //  Ô - Output (Matrix) : Shape => (m, n)

  // Apply the activation function to transform the input.
  return input | act_func_;
}

}
