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

#include "cbrainx/activationLayer.hh"

#include <utility>

namespace cbx {

// /////////////////////////////////////////////
// Constructors (and Destructors)
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

auto ActivationLayer::property() const -> std::string {
  return fmt::format("Function: {}", act_func_.to_string());
}

auto ActivationLayer::type_name() const -> std::string { return "Activation"; }

// /////////////////////////////////////////////
// Core Functionality
// /////////////////////////////////////////////

auto ActivationLayer::forward_pass(const container &input) const -> container {
  // Formula: Ô = ζ(Î)
  //
  // where:
  //  ζ - Activation function
  //  Î - Input (Matrix)  => Shape = (m, n)
  //  Ô - Output (Matrix) => Shape = (m, n)
  //
  // Note: Cached input and output will be used during back-propagation.

  input_ = input;

  // Apply the activation function as a transformation.
  output_ = input | act_func_;
  return output_;
}

auto ActivationLayer::backward_pass(const container &upstream_gradient, OptimizerWrapper) -> container {
  // Formula: ΔḒ = ζ'(Î) . ΔÛ
  //
  // where:
  //  ζ  - Activation function
  //  ζ' - Derivative of the activation function
  //  Î  - Input (Matrix)                => Shape = (m, n)
  //  Ô  - Output (Matrix)               => Shape = (m, n)
  //  ΔḒ - Downstream gradient (Matrix)  => Shape = (m, n)
  //  ΔÛ - Upstream gradient (Matrix)    => Shape = (m, n)

  // Calculate the local gradient.
  // Formula: ζ'(Î)
  auto local_gradient = input_ | act_func_.derivative();

  // Return the downstream gradient.
  // Formula: ΔḒ = ζ'(Î) . ΔÛ
  return local_gradient * upstream_gradient;
}

}
