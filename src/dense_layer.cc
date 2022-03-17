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

// Copyright (c) 2021 Mansoor Ahmed <mansoorahmed.one@gmail.com>

#include "cbrainx/dense_layer.hh"

#include <limits>

#include <fmt/format.h>

#include "cbrainx/matrix.hh"

namespace cbx {

DenseLayer::DenseLayer(shape_value_t inputs, shape_value_t neurons) : AbstractLayer{"DNSL"} {
  weights_ = Tensor<f32>::random({inputs, neurons}, {}, -1, 1);
  biases_ = Tensor<f32>::fill({neurons}, std::numeric_limits<f32>::epsilon());
}

DenseLayer::DenseLayer(DenseLayer &&other) noexcept
    : weights_{std::move(other.weights_)}, biases_{std::move(other.biases_)} {}

// /////////////////////////////////////////////////////////////

auto DenseLayer::operator=(DenseLayer &&other) noexcept -> DenseLayer & {
  weights_ = std::move(other.weights_);
  biases_ = std::move(other.biases_);
  return *this;
}

// /////////////////////////////////////////////////////////////

auto DenseLayer::neurons() const -> size_type { return biases_.total(); }

auto DenseLayer::parameters() const -> size_type { return weights_.total() + biases_.total(); }

auto DenseLayer::property() const -> std::string {
  return fmt::format("Shape: W={}, B={}", weights_.shape().to_string(), biases_.shape().to_string());
}

auto DenseLayer::type() const -> LayerType { return LayerType::Dense; }

auto DenseLayer::output() const -> const Tensor<f32> & { return output_; }

// /////////////////////////////////////////////////////////////

auto DenseLayer::forward_pass(const Tensor<f32> &input) -> AbstractLayer & {
  output_ = Matrix::multiply(input, weights_);
  auto [samples] = input.shape().unwrap<1>();
  for (shape_value_t i = {}; i < samples; ++i) {
    Matrix::add_row_wise(output_, biases_);
  }
  return *this;
}

}
