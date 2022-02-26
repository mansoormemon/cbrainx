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

auto ArcTan::type() const -> Activation { return Activation::ArcTan; }

auto ArcTan::to_string() const -> std::string { return "Function: ArcTan"; }

auto ArcTan::type_name() const -> std::string { return "ArcTan"; }

// /////////////////////////////////////////////////////////////

auto BinaryStep::operator()(value_type x) const -> value_type { return static_cast<value_type>(x >= 0); }

auto BinaryStep::derivative(value_type) const -> value_type { return 0; }

auto BinaryStep::type() const -> Activation { return Activation::BinaryStep; }

auto BinaryStep::to_string() const -> std::string { return "Function: Binary Step"; }

auto BinaryStep::type_name() const -> std::string { return "BinaryStep"; }

// /////////////////////////////////////////////////////////////

auto ELU::operator()(value_type x) const -> value_type { return x >= 0 ? x : ALPHA * std::expm1(x); }

auto ELU::derivative(value_type x) const -> value_type { return x >= 0 ? 1 : ALPHA * std::exp(x); }

auto ELU::type() const -> Activation { return Activation::ELU; }

auto ELU::to_string() const -> std::string { return "Function: ELU"; }

auto ELU::type_name() const -> std::string { return "ELU"; }

// /////////////////////////////////////////////////////////////

auto Gaussian::operator()(value_type x) const -> value_type { return std::exp(-x * x); }

auto Gaussian::derivative(value_type x) const -> value_type { return -2 * x * this->operator()(x); }

auto Gaussian::type() const -> Activation { return Activation::Gaussian; }

auto Gaussian::to_string() const -> std::string { return "Function: Gaussian"; }

auto Gaussian::type_name() const -> std::string { return "Gaussian"; }

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

auto GELU::type() const -> Activation { return Activation::GELU; }

auto GELU::to_string() const -> std::string { return "Function: GELU"; }

auto GELU::type_name() const -> std::string { return "GELU"; }

// /////////////////////////////////////////////////////////////

auto LeakyReLU::operator()(value_type x) const -> value_type { return std::max(M * x, x); }

auto LeakyReLU::derivative(value_type x) const -> value_type { return x >= 0 ? 1 : M; }

auto LeakyReLU::type() const -> Activation { return Activation::LeakyReLU; }

auto LeakyReLU::to_string() const -> std::string { return "Function: Leaky ReLU"; }

auto LeakyReLU::type_name() const -> std::string { return "LeakyReLU"; }

// /////////////////////////////////////////////////////////////

auto Linear::operator()(value_type x) const -> value_type { return x; }

auto Linear::derivative(value_type) const -> value_type { return 1; }

auto Linear::type() const -> Activation { return Activation::Linear; }

auto Linear::to_string() const -> std::string { return "Function: Linear"; }

auto Linear::type_name() const -> std::string { return "Linear"; }

// /////////////////////////////////////////////////////////////

auto ReLU::operator()(value_type x) const -> value_type { return std::max(0.0F, x); }

auto ReLU::derivative(value_type x) const -> value_type { return static_cast<value_type>(x >= 0); }

auto ReLU::type() const -> Activation { return Activation::ReLU; }

auto ReLU::to_string() const -> std::string { return "Function: ReLU"; }

auto ReLU::type_name() const -> std::string { return "ReLU"; }

// /////////////////////////////////////////////////////////////

auto Sigmoid::operator()(value_type x) const -> value_type {
  value_type denominator = 1 + std::exp(-x);
  return 1 / denominator;
}

auto Sigmoid::derivative(value_type x) const -> value_type {
  return this->operator()(x) * (1 - this->operator()(x));
}

auto Sigmoid::type() const -> Activation { return Activation::Sigmoid; }

auto Sigmoid::to_string() const -> std::string { return "Function: Sigmoid"; }

auto Sigmoid::type_name() const -> std::string { return "Sigmoid"; }

// /////////////////////////////////////////////////////////////

auto SoftPlus::operator()(value_type x) const -> value_type { return std::log1p(std::exp(x)); }

auto SoftPlus::derivative(value_type x) const -> value_type {
  auto sigmoid = Sigmoid{};
  return sigmoid(x);
}

auto SoftPlus::type() const -> Activation { return Activation::SoftPlus; }

auto SoftPlus::to_string() const -> std::string { return "Function: Soft Plus"; }

auto SoftPlus::type_name() const -> std::string { return "SoftPlus"; }

// /////////////////////////////////////////////////////////////

auto Swish::operator()(value_type x) const -> value_type {
  auto sigmoid = Sigmoid{};
  return x * sigmoid(x);
}

auto Swish::derivative(value_type x) const -> value_type {
  auto sigmoid = Sigmoid{};
  return this->operator()(x) + (sigmoid(x) * (1 - this->operator()(x)));
}

auto Swish::type() const -> Activation { return Activation::Swish; }

auto Swish::to_string() const -> std::string { return "Function: Swish"; }

auto Swish::type_name() const -> std::string { return "Swish"; }

// /////////////////////////////////////////////////////////////

auto TanH::operator()(value_type x) const -> value_type { return std::tanh(x); }

auto TanH::derivative(value_type x) const -> value_type {
  return 1 - (this->operator()(x) * this->operator()(x));
}

auto TanH::type() const -> Activation { return Activation::TanH; }

auto TanH::to_string() const -> std::string { return "Function: TanH"; }

auto TanH::type_name() const -> std::string { return "TanH"; }

}
