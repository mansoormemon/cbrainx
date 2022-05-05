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

#include "cbrainx/denseLayer.hh"

#include <limits>

#include <fmt/core.h>

namespace cbx {

// /////////////////////////////////////////////
// Constructors (and Destructors)
// /////////////////////////////////////////////

DenseLayer::DenseLayer(size_type inputs, size_type neurons) : AbstractLayer{"DNSL"} {
  weights_ = container::random({inputs, neurons}, {}, -1, 1);
  biases_ = container{{neurons}, std::numeric_limits<value_type>::epsilon()};
}

DenseLayer::DenseLayer(DenseLayer &&other) noexcept
    : weights_{std::move(other.weights_)}, biases_{std::move(other.biases_)} {}

// /////////////////////////////////////////////
// Assignment Operators
// /////////////////////////////////////////////

auto DenseLayer::operator=(DenseLayer &&other) noexcept -> DenseLayer & {
  weights_ = std::move(other.weights_);
  biases_ = std::move(other.biases_);
  return *this;
}

// /////////////////////////////////////////////
// Query Functions
// /////////////////////////////////////////////

auto DenseLayer::neurons() const -> size_type { return biases_.total(); }

auto DenseLayer::parameters() const -> size_type { return weights_.total() + biases_.total(); }

auto DenseLayer::type() const -> LayerType { return LayerType::Dense; }

// /////////////////////////////////////////////
// Informative
// /////////////////////////////////////////////

auto DenseLayer::property() const -> std::string {
  return fmt::format("Shape: W={}, B={}", weights_.shape().to_string(), biases_.shape().to_string());
}

auto DenseLayer::type_name() const -> std::string { return "Dense"; }

// /////////////////////////////////////////////
// Core Functionality
// /////////////////////////////////////////////

auto DenseLayer::forward_pass(const container &input) const -> container {
  // Formula: Ô = Î ⊙ Ŵ + Ƀ
  //
  // where:
  //  Î - Input (Matrix)   : Shape => (m, n)
  //  Ŵ - Weights (Matrix) : Shape => (n, o)
  //  Ƀ - Biases (Vector)  : Shape => (o)
  //  Ô - Output (Matrix)  : Shape => (m, o)
  //
  // and, the symbol `⊙` denotes dot product (typically matrix multiplication).

  // Applying forward pass and caching the input and output layers.
  input_ = input;
  output_ = input.matmul(weights_) + biases_;
  return output_;
}

}
