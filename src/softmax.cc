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

Softmax::Softmax(size_type inputs) : AbstractLayer{"SFML"}, neurons_{inputs} {}

Softmax::Softmax(Softmax &&other) noexcept : neurons_{std::exchange(other.neurons_, {})} {}

// /////////////////////////////////////////////
// Assignment Operators
// /////////////////////////////////////////////

auto Softmax::operator=(Softmax &&other) noexcept -> Softmax & {
  neurons_ = std::exchange(other.neurons_, {});
  return *this;
}

// /////////////////////////////////////////////
// Query Functions
// /////////////////////////////////////////////

auto Softmax::neurons() const -> size_type { return neurons_; }

auto Softmax::parameters() const -> size_type { return {}; }

auto Softmax::type() const -> LayerType { return LayerType::Softmax; }

// /////////////////////////////////////////////
// Informative
// /////////////////////////////////////////////

auto Softmax::property() const -> std::string { return "-"; }

auto Softmax::type_name() const -> std::string { return "Softmax"; }

// /////////////////////////////////////////////
// Core Functionality
// /////////////////////////////////////////////

auto Softmax::forward_pass(const container &input) const -> container {
  // Formula: Ō = σ(Ƶ)i [i = 1, n] = ęᶼ / ⅀ [j = 1, n] ęᶽ
  //
  // where:
  //  σ - Softmax function
  //  Ƶ - Input (Vector) => Shape = (k)
  //  ʐ - iᵗʰ element in the vector
  //  ʑ - jᵗʰ element in the vector
  //  n = Number of classes in the multi-class classifier
  //  Ō = Output (Vector) => Shape = (k)
  //
  // Note: The formula above only pertains to one sample (along the x-axis). Also, the cached input and output
  // will be used during back-propagation.

  input_ = input;
  output_ = input.zeros_like();
  auto total = input.total();
  // Iterate along the y-axis.
  for (size_type i = {}; i < total; i += neurons_) {
    // Determine the boundaries of each sample.
    auto in_begin = input.begin() + i;
    auto in_end = in_begin + neurons_;
    auto out_begin = output_.begin() + i;
    // Accumulate inputs along the x-axis.
    // Formula: ⅀ [j = 1, n] ęᶽ
    auto acc = std::accumulate(in_begin, in_end, 0.0F, [](auto acc, auto x) {
      return acc + std::exp(x);
    });
    // Calculate probability distributions.
    // Formula: ęᶼ / ⅀ [j = 1, n] ęᶽ
    std::transform(in_begin, in_end, out_begin, [acc](auto x) {
      return std::exp(x) / acc;
    });
  }
  return output_;
}

auto Softmax::backward_pass(const container &upstream_gradient, OptimizerWrapper) -> container {
  // Formula: ΔḒ = ΔÛ ⎊ Ĵ
  //
  // where:
  //  Ĵ   - Local gradient (Jacobian Matrix)  => Shape = (n, n)
  //  ΔḒ  - Downstream gradient (Matrix)      => Shape = (1, n)
  //  ΔÛ  - Upstream gradient (Matrix)        => Shape = (1, n)
  //
  // and, the symbol `⎊` denotes dot product (typically matrix multiplication).
  //
  // Computing the Jacobian matrix
  //
  // The Jacobian matrix is computed as follows:
  //
  //  Ĵ = [[ ẟ / ẟ𝓍₁  𝑦₁    ẟ / ẟ𝓍₂  𝑦₁    ...       ẟ / ẟ𝓍ⱼ  𝑦₁ ],
  //       [ ẟ / ẟ𝓍₁  𝑦₂    ẟ / ẟ𝓍₂  𝑦₂    ...       ẟ / ẟ𝓍ⱼ  𝑦₂ ],
  //        ⋮              ⋮               ⋱       ⋮
  //       [ ẟ / ẟ𝓍₁  𝑦ᵢ    ẟ / ẟ𝓍₂  𝑦ᵢ    ...       ẟ / ẟ𝓍ⱼ  𝑦ᵢ ]
  //
  // where:
  //  𝓍           - Input (Vector)                => Shape = (n)
  //  𝑦           - Output (Vector)               => Shape = (n)
  //  Ĵ           -  Jacobian Matrix              => Shape = (n, n)
  //  ẟ / ẟ𝓍ⱼ  𝑦ᵢ - Derivative of  𝑦ᵢ w.r.t. 𝓍ⱼ  => Formula: yᵢ . (ƍ - yⱼ)
  //  ƍ           - Kronecker delta               => Formula: ƍᵢⱼ = [i = j]
  //
  // It should be noted that the formulas above only pertain to one sample (along the x-axis). The actual
  // implementation iterates along the y-axis and applies the above formulas to each sample individually.

  using iterator = container::iterator;

  // Computes the jacobian matrix of the given sample.
  auto compute_jacobian = [this](iterator output) {
    // Set dimensions of the Jacobian matrix.
    auto jacobian = container{{neurons_, neurons_}};

    // Formula: yᵢ . (ƍ - yⱼ)
    // ƍᵢⱼ = 1 if i == j else 0
    auto kronecker_delta = [](auto i, auto j) {
      return i == j;
    };

    // `i` represents the y-axis whereas `j` represents the x-axis.
    for (size_type i = {}; i < neurons_; ++i) {
      for (size_type j = {}; j < neurons_; ++j) {
        jacobian(i, j) = output[i] * (kronecker_delta(i, j) - output[j]);
      }
    }
    return jacobian;
  };

  // Downstream gradient will have the same dimensions as the input.
  // Shape = (m, n)
  auto downstream_gradient = input_.zeros_like();
  auto total = output_.total();
  // Iterate along the y-axis.
  for (size_type i = {}; i < total; i += neurons_) {
    // Determine the boundaries of each sample.
    auto out_begin = output_.begin() + i;
    auto up_grad_begin = upstream_gradient.begin() + i;
    auto down_grad_begin = downstream_gradient.begin() + i;
    // Compute the Jacobian matrix.
    auto jacobian = compute_jacobian(out_begin);
    auto sample_up_grad = container{{neurons_}, up_grad_begin}.reshape(container::MATRIX_RANK, true);
    // Compute the downstream gradient of the sample.
    // Formula: ΔḒ = ΔÛ ⎊ Ĵ
    auto sample_local_grad = sample_up_grad.matmul(jacobian);
    // Copy the sample's downstream gradient to the final tensor.
    std::copy(sample_local_grad.begin(), sample_local_grad.end(), down_grad_begin);
  }
  // Return the downstream gradient.
  return downstream_gradient;
}

}
