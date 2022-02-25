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

#include "cbrainx/activation_functions.hh"

#include <cmath>

namespace cbx {

auto ArcTan::operator()(value_type x) const -> value_type { return std::atan(x); }

auto ArcTan::derivative(value_type x) const -> value_type {
  value_type denominator = 1 + (x * x);
  return 1 / denominator;
}

auto ArcTan::to_string() -> str { return "Function: ArcTan"; }

// /////////////////////////////////////////////////////////////

auto BinaryStep::operator()(value_type x) const -> value_type { return static_cast<value_type>(x >= 0); }

auto BinaryStep::derivative(value_type) const -> value_type { return 0; }

auto BinaryStep::to_string() -> str { return "Function: Binary Step"; }

// /////////////////////////////////////////////////////////////

auto ELU::operator()(value_type x) const -> value_type { return x >= 0 ? x : ALPHA * std::expm1(x); }

auto ELU::derivative(value_type x) const -> value_type { return x >= 0 ? 1 : ALPHA * std::exp(x); }

auto ELU::to_string() -> str { return "Function: ELU"; }

// /////////////////////////////////////////////////////////////

auto Gaussian::operator()(value_type x) const -> value_type { return std::exp(-x * x); }

auto Gaussian::derivative(value_type x) const -> value_type { return -2 * x * this->operator()(x); }

auto Gaussian::to_string() -> str { return "Function: Gaussian"; }

// /////////////////////////////////////////////////////////////

auto GELU::operator()(value_type x) const -> value_type {
  auto sigmoid = Sigmoid{};
  return x * sigmoid(C * x);
}

auto GELU::derivative(value_type x) const -> value_type {
  auto sigmoid = Sigmoid{};
  auto swish = Swish{};
  return sigmoid(C * x) + swish(C * x) * (1 - sigmoid(C * x));
}

auto GELU::to_string() -> str { return "Function: GELU"; }

// /////////////////////////////////////////////////////////////

auto LeakyReLU::operator()(value_type x) const -> value_type { return std::max(M * x, x); }

auto LeakyReLU::derivative(value_type x) const -> value_type { return x >= 0 ? 1 : M; }

auto LeakyReLU::to_string() -> str { return "Function: Leaky ReLU"; }

// /////////////////////////////////////////////////////////////

auto Linear::operator()(value_type x) const -> value_type { return x; }

auto Linear::derivative(value_type) const -> value_type { return 1; }

auto Linear::to_string() -> str { return "Function: Linear"; }

// /////////////////////////////////////////////////////////////

auto ReLU::operator()(value_type x) const -> value_type { return std::max(0.0F, x); }

auto ReLU::derivative(value_type x) const -> value_type { return static_cast<value_type>(x >= 0); }

auto ReLU::to_string() -> str { return "Function: ReLU"; }

// /////////////////////////////////////////////////////////////

auto Sigmoid::operator()(value_type x) const -> value_type {
  value_type denominator = 1 + std::exp(-x);
  return 1 / denominator;
}

auto Sigmoid::derivative(value_type x) const -> value_type {
  return this->operator()(x) * (1 - this->operator()(x));
}

auto Sigmoid::to_string() -> str { return "Function: Sigmoid"; }

// /////////////////////////////////////////////////////////////

auto SoftPlus::operator()(value_type x) const -> value_type { return std::log1p(std::exp(x)); }

auto SoftPlus::derivative(value_type x) const -> value_type {
  auto sigmoid = Sigmoid{};
  return sigmoid(x);
}

auto SoftPlus::to_string() -> str { return "Function: Soft Plus"; }

// /////////////////////////////////////////////////////////////

auto Swish::operator()(value_type x) const -> value_type {
  auto sigmoid = Sigmoid{};
  return x * sigmoid(x);
}

auto Swish::derivative(value_type x) const -> value_type {
  auto sigmoid = Sigmoid{};
  return this->operator()(x) + (sigmoid(x) * (1 - this->operator()(x)));
}

auto Swish::to_string() -> str { return "Function: Swish"; }

// /////////////////////////////////////////////////////////////

auto TanH::operator()(value_type x) const -> value_type { return std::tanh(x); }

auto TanH::derivative(value_type x) const -> value_type {
  return 1 - (this->operator()(x) * this->operator()(x));
}

auto TanH::to_string() -> str { return "Function: TanH"; }

}
