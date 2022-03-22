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

#include "cbrainx/soft_max.hh"

#include <algorithm>
#include <numeric>
#include <utility>

namespace cbx {

SoftMax::SoftMax(shape_value_t inputs) : AbstractLayer{"SFML"}, neurons_{inputs} {}

SoftMax::SoftMax(SoftMax &&other) noexcept : neurons_{std::exchange(other.neurons_, {})} {}

// /////////////////////////////////////////////////////////////

auto SoftMax::operator=(SoftMax &&other) noexcept -> SoftMax & {
  neurons_ = std::exchange(other.neurons_, {});
  return *this;
}

// /////////////////////////////////////////////////////////////

auto SoftMax::neurons() const -> size_type { return neurons_; }

auto SoftMax::parameters() const -> size_type { return {}; }

auto SoftMax::property() const -> std::string { return "-"; }

auto SoftMax::type() const -> LayerType { return LayerType::SoftMax; }

// /////////////////////////////////////////////////////////////

auto SoftMax::forward_pass(container_const_reference input) -> AbstractLayer & {
  input_ = input;
  output_ = Tensor<f32>::zeros(input.shape());
  auto [_, cols] = input.shape().template unwrap<2>();
  auto total = input.total();
  for (Shape::value_type i = {}; i < total; i += cols) {
    auto in_begin = input.begin() + i;
    auto in_end = in_begin + cols;
    auto out_begin = output_.begin() + i;
    auto acc = std::accumulate(in_begin, in_end, 0.0F, [](const auto &acc, const auto &x) {
      return acc + std::exp(x);
    });
    std::transform(in_begin, in_end, out_begin, [acc](const auto &x) {
      return std::exp(x) / acc;
    });
  }
  return *this;
}

}
