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
  // Formula: Ô = Î ⎊ Ŵ + Ƀ
  //
  // where:
  //  Î - Input (Matrix)   => Shape = (m, n)
  //  Ŵ - Weights (Matrix) => Shape = (n, o)
  //  Ƀ - Biases (Vector)  => Shape = (o)
  //  Ô - Output (Matrix)  => Shape = (m, o)
  //
  // and, the symbol `⎊` denotes dot product (typically matrix multiplication).
  //
  // Note: The cached input and output will be used during back-propagation.

  input_ = input;
  output_ = input.matmul(weights_) + biases_;
  return output_;
}

auto DenseLayer::backward_pass(const container &upstream_gradient, OptimizerWrapper optimizer) -> container {
  // Formula: ΔŴ = Î.T ⎊ ΔÛ         :> Ʊ(Ŵ, ΔŴ)
  //          ΔɃ = sum(ΔÛ, axis=y)  :> Ʊ(Ƀ, ΔɃ)
  //          ΔḒ = ΔÛ ⎊ Ŵ.T
  //
  // where:
  //  Î   - Input (Matrix)                 => Shape = (m, n)
  //  Î.T - Transpose of input (Matrix)    => Shape = (n, m)
  //  Ŵ   - Weights (Matrix)               => Shape = (n, o)
  //  Ŵ.T - Transpose of weights (Matrix)  => Shape = (o, n)
  //  ΔŴ  - Weights gradient (Matrix)      => Shape = (n, o)
  //  Ƀ   - Biases (Vector)                => Shape = (o)
  //  ΔɃ  - Biases gradient (Vector)       => Shape = (o)
  //  ΔḒ  - Downstream gradient (Matrix)   => Shape = (m, n)
  //  ΔÛ  - Upstream gradient (Matrix)     => Shape = (m, o)
  //  Ʊ   - Optimizer
  //
  // and, the symbol `⎊` denotes dot product (typically matrix multiplication).

  // Returns the transpose of the given matrix.
  auto transpose = [](const container &matrix) -> container {
    auto [rows, cols] = matrix.shape().unwrap<2>();
    auto result = container{{cols, rows}};
    for (size_type r = {}; r < rows; ++r) {
      for (size_type c = {}; c < cols; ++c) {
        result(c, r) = matrix(r, c);
      }
    }
    return result;
  };

  // Reduces the given matrix by summing it along the y-axis.
  auto sum_y = [](const container &matrix) -> container {
    auto [rows, cols] = matrix.shape().unwrap<2>();
    auto result = container{{cols}};
    for (size_type c = {}; c < cols; ++c) {
      for (size_type r = {}; r < rows; ++r) {
        result[c] += matrix(r, c);
      }
    }
    return result;
  };

  // Compute the gradient of weights and update the parameters.
  // Formula: ΔŴ = Î.T ⎊ ΔÛ  :> Ʊ(Ŵ, ΔŴ)
  auto weights_gradient = transpose(input_).matmul(upstream_gradient);
  optimizer.update_params(weights_, weights_gradient);

  // Compute the gradient of biases and update the parameters.
  // Formula: ΔɃ = sum(ΔÛ, axis=y)  :> Ʊ(Ƀ, ΔɃ)
  auto biases_gradient = sum_y(upstream_gradient);
  optimizer.update_params(biases_, biases_gradient);

  // Return the downstream gradient.
  // Formula: ΔḒ = ΔÛ ⎊ Ŵ.T
  return upstream_gradient.matmul(transpose(weights_));
}

}
