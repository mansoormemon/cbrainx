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

#include "cbrainx/softmax.hh"

#include <algorithm>
#include <numeric>
#include <utility>

namespace cbx {

// /////////////////////////////////////////////
// Constructors and Destructors
// /////////////////////////////////////////////

SoftMax::SoftMax(size_type inputs) : AbstractLayer{"SFML"}, neurons_{inputs} {}

SoftMax::SoftMax(SoftMax &&other) noexcept : neurons_{std::exchange(other.neurons_, {})} {}

// /////////////////////////////////////////////
// Assignment Operators
// /////////////////////////////////////////////

auto SoftMax::operator=(SoftMax &&other) noexcept -> SoftMax & {
  neurons_ = std::exchange(other.neurons_, {});
  return *this;
}

// /////////////////////////////////////////////
// Query Functions
// /////////////////////////////////////////////

auto SoftMax::neurons() const -> size_type { return neurons_; }

auto SoftMax::parameters() const -> size_type { return {}; }

auto SoftMax::property() const -> std::string { return "-"; }

auto SoftMax::type() const -> LayerType { return LayerType::Softmax; }

// /////////////////////////////////////////////
// Informative
// /////////////////////////////////////////////

auto SoftMax::forward_pass(const container &input) -> container {
  // The forward pass of this layer performs the subsequent operation.
  //
  // Formula: Ō = σ(Ƶ)ὶ [ὶ = 1, ƙ] = ęᶼ / ⅀ [ʝ = 1, ƙ] ęᶽ
  ///
  // where:
  //  σ - Softmax function
  //  Ƶ - Input (Vector) : Shape => (k)
  //  ʐ - ὶᵗʰ element in the vector
  //  ʑ - ʝᵗʰ element in the vector
  //  ƙ = Number of classes in the multi-class classifier
  //  Ō = Output (Vector) : Shape => (k)
  ///
  // It should be noted that the formula above only pertains to one sample (along the x-axis).
  auto output = input.zeros_like();
  auto [_, cols] = input.shape().unwrap<2, isize>();
  auto total = input.total();
  // Iterate along the y-axis.
  for (size_type i = {}; i < total; i += cols) {
    // Determine the boundaries of each sample.
    auto in_begin = input.begin() + i;
    auto in_end = in_begin + cols;
    auto out_begin = output.begin() + i;
    // Accumulate inputs along the x-axis.
    // Formula: ⅀ [ʝ = 1, ƙ] ęᶽ
    auto acc = std::accumulate(in_begin, in_end, 0.0F, [](auto acc, auto x) {
      return acc + std::exp(x);
    });
    // Calculate probability distributions.
    // Formula: ęᶼ / ⅀ [ʝ = 1, ƙ] ęᶽ
    std::transform(in_begin, in_end, out_begin, [acc](auto x) {
      return std::exp(x) / acc;
    });
  }
  return output;
}

}
