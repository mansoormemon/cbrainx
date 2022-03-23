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

// /////////////////////////////////////////////////////////////

auto DenseLayer::forward_pass(container_const_reference input) -> AbstractLayer & {
  input_ = input;
  output_ = Matrix::multiply(input, weights_);
  auto [samples] = input.shape().unwrap<1>();
  for (shape_value_t i = {}; i < samples; ++i) {
    Matrix::add_row_wise(output_, biases_);
  }
  return *this;
}

auto DenseLayer::backward_pass(container_const_reference dinput, std::shared_ptr<Optimizer> optimizer)
    -> container {
  auto transpose = [](const Tensor<f32> &input) -> Tensor<f32> {
    auto [row, col] = input.shape().unwrap<2>();
    auto transposed = Tensor<f32>::zeros({col, row});
    for (auto r = 0U; r < row; ++r) {
      for (auto c = 0U; c < col; ++c) {
        transposed(c, r) = input(r, c);
      }
    }
    return transposed;
  };

  auto t_dinput = transpose(dinput);
  auto d_weights = Matrix::multiply(t_dinput, input_);

  optimizer->update_params(weights_.begin(), weights_.end(), d_weights.begin());

  auto [rows, cols] = dinput.shape().unwrap<2>();
  auto d_biases = Tensor<f32>::zeros(biases_.shape());

  // Summing all the rows.
  for (auto c = 0U; c < cols; ++c) {
    for (auto r = 0U; r < rows; ++r) {
      d_biases[c] += dinput(r, c);
    }
  }
  optimizer->update_params(biases_.begin(), biases_.end(), d_biases.begin());

  auto t_weights = transpose(weights_);
  return Matrix::multiply(dinput, t_weights);
}

}
