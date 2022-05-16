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

#include "cbrainx/activationFunctions.hh"

#include <cmath>

namespace cbx {

// /////////////////////////////////////////////////////////////
// Activation Functions
// /////////////////////////////////////////////////////////////

// /////////////////////////////////////////////
// Interface
// /////////////////////////////////////////////

auto ArcTan::type() const -> Activation { return Activation::ArcTan; }

auto ArcTan::to_string() const -> std::string { return "Arc Tan"; }

auto ArcTan::type_name() const -> std::string { return "ArcTan"; }

auto ArcTan::operator()(value_type x) const -> value_type { return std::atan(x); }

auto ArcTan::derivative(value_type x) const -> value_type {
  value_type denominator = 1 + (x * x);
  return 1 / denominator;
}

// /////////////////////////////////////////////

auto BinaryStep::type() const -> Activation { return Activation::BinaryStep; }

auto BinaryStep::to_string() const -> std::string { return "Binary Step"; }

auto BinaryStep::type_name() const -> std::string { return "BinaryStep"; }

auto BinaryStep::operator()(value_type x) const -> value_type { return value_type(x >= 0); }

auto BinaryStep::derivative(value_type) const -> value_type { return 0; }

// /////////////////////////////////////////////

auto ELU::type() const -> Activation { return Activation::ELU; }

auto ELU::to_string() const -> std::string { return "ELU"; }

auto ELU::type_name() const -> std::string { return "ELU"; }

auto ELU::operator()(value_type x) const -> value_type { return x >= 0 ? x : ALPHA * std::expm1(x); }

auto ELU::derivative(value_type x) const -> value_type { return x >= 0 ? 1 : ALPHA * std::exp(x); }

// /////////////////////////////////////////////

auto Gaussian::type() const -> Activation { return Activation::Gaussian; }

auto Gaussian::to_string() const -> std::string { return "Gaussian"; }

auto Gaussian::type_name() const -> std::string { return "Gaussian"; }

auto Gaussian::operator()(value_type x) const -> value_type { return std::exp(-x * x); }

auto Gaussian::derivative(value_type x) const -> value_type { return -2 * x * operator()(x); }

// /////////////////////////////////////////////

auto GELU::type() const -> Activation { return Activation::GELU; }

auto GELU::to_string() const -> std::string { return "GELU"; }

auto GELU::type_name() const -> std::string { return "GELU"; }

auto GELU::operator()(value_type x) const -> value_type {
  auto sigmoid = Sigmoid{};
  return x * sigmoid(C * x);
}

auto GELU::derivative(value_type x) const -> value_type {
  auto sigmoid = Sigmoid{};
  auto swish = Swish{};
  return sigmoid(C * x) + swish(C * x) * (1 - sigmoid(C * x));
}

// /////////////////////////////////////////////

auto LeakyReLU::type() const -> Activation { return Activation::LeakyReLU; }

auto LeakyReLU::to_string() const -> std::string { return "Leaky ReLU"; }

auto LeakyReLU::type_name() const -> std::string { return "LeakyReLU"; }

auto LeakyReLU::operator()(value_type x) const -> value_type { return std::max(M * x, x); }

auto LeakyReLU::derivative(value_type x) const -> value_type { return x >= 0 ? 1 : M; }

// /////////////////////////////////////////////

auto Linear::type() const -> Activation { return Activation::Linear; }

auto Linear::to_string() const -> std::string { return "Linear"; }

auto Linear::type_name() const -> std::string { return "Linear"; }

auto Linear::operator()(value_type x) const -> value_type { return 0.01F * x; }

auto Linear::derivative(value_type) const -> value_type { return 0.01F; }

// /////////////////////////////////////////////

auto ReLU::type() const -> Activation { return Activation::ReLU; }

auto ReLU::to_string() const -> std::string { return "ReLU"; }

auto ReLU::type_name() const -> std::string { return "ReLU"; }

auto ReLU::operator()(value_type x) const -> value_type { return std::max(0.0F, x); }

auto ReLU::derivative(value_type x) const -> value_type { return value_type(x >= 0); }

// /////////////////////////////////////////////

auto Sigmoid::type() const -> Activation { return Activation::Sigmoid; }

auto Sigmoid::to_string() const -> std::string { return "Sigmoid"; }

auto Sigmoid::type_name() const -> std::string { return "Sigmoid"; }

auto Sigmoid::operator()(value_type x) const -> value_type {
  value_type denominator = 1 + std::exp(-x);
  return 1 / denominator;
}

auto Sigmoid::derivative(value_type x) const -> value_type { return operator()(x) * (1 - operator()(x)); }

// /////////////////////////////////////////////

auto Softplus::type() const -> Activation { return Activation::Softplus; }

auto Softplus::to_string() const -> std::string { return "Softplus"; }

auto Softplus::type_name() const -> std::string { return "Softplus"; }

auto Softplus::operator()(value_type x) const -> value_type { return std::log1p(std::exp(x)); }

auto Softplus::derivative(value_type x) const -> value_type {
  auto sigmoid = Sigmoid{};
  return sigmoid(x);
}

// /////////////////////////////////////////////

auto Swish::type() const -> Activation { return Activation::Swish; }

auto Swish::to_string() const -> std::string { return "Swish"; }

auto Swish::type_name() const -> std::string { return "Swish"; }

auto Swish::operator()(value_type x) const -> value_type {
  auto sigmoid = Sigmoid{};
  return x * sigmoid(x);
}

auto Swish::derivative(value_type x) const -> value_type {
  auto sigmoid = Sigmoid{};
  return operator()(x) + (sigmoid(x) * (1 - operator()(x)));
}

// /////////////////////////////////////////////

auto TanH::type() const -> Activation { return Activation::TanH; }

auto TanH::to_string() const -> std::string { return "TanH"; }

auto TanH::type_name() const -> std::string { return "TanH"; }

auto TanH::operator()(value_type x) const -> value_type { return std::tanh(x); }

auto TanH::derivative(value_type x) const -> value_type { return 1 - (operator()(x) * operator()(x)); }

// /////////////////////////////////////////////////////////////
// ActFuncWrapper
// /////////////////////////////////////////////////////////////

// /////////////////////////////////////////////
// Constructors (and Destructors)
// /////////////////////////////////////////////

ActFuncWrapper::ActFuncWrapper(Activation activation) {
  switch (activation) {
    case Activation::ArcTan: {
      func_ = std::make_shared<ArcTan>();
      break;
    }
    case Activation::BinaryStep: {
      func_ = std::make_shared<BinaryStep>();
      break;
    }
    case Activation::ELU: {
      func_ = std::make_shared<ELU>();
      break;
    }
    case Activation::Gaussian: {
      func_ = std::make_shared<Gaussian>();
      break;
    }
    case Activation::GELU: {
      func_ = std::make_shared<GELU>();
      break;
    }
    case Activation::LeakyReLU: {
      func_ = std::make_shared<LeakyReLU>();
      break;
    }
    case Activation::Linear: {
      func_ = std::make_shared<Linear>();
      break;
    }
    case Activation::ReLU: {
      func_ = std::make_shared<ReLU>();
      break;
    }
    case Activation::Sigmoid: {
      func_ = std::make_shared<Sigmoid>();
      break;
    }
    case Activation::Softplus: {
      func_ = std::make_shared<Softplus>();
      break;
    }
    case Activation::Swish: {
      func_ = std::make_shared<Swish>();
      break;
    }
    case Activation::TanH: {
      func_ = std::make_shared<TanH>();
      break;
    }
  }
}

ActFuncWrapper::ActFuncWrapper(ActFuncWrapper &&other) noexcept : func_{std::move(other.func_)} {}

// /////////////////////////////////////////////
// Assignment Operators
// /////////////////////////////////////////////

auto ActFuncWrapper::operator=(ActFuncWrapper &&other) noexcept -> ActFuncWrapper & {
  func_ = std::move(other.func_);
  return *this;
}

// /////////////////////////////////////////////
// Query Functions
// /////////////////////////////////////////////

auto ActFuncWrapper::is_null() const -> bool { return func_ == nullptr; }

// /////////////////////////////////////////////
// Wrapper Interface
// /////////////////////////////////////////////

auto ActFuncWrapper::type() const -> Activation { return func_->type(); }

auto ActFuncWrapper::to_string() const -> std::string { return func_->to_string(); }

auto ActFuncWrapper::type_name() const -> std::string { return func_->type_name(); }

auto ActFuncWrapper::operator()(value_type x) const -> value_type { return func_->operator()(x); }

auto ActFuncWrapper::derivative() const -> std::function<value_type(value_type)> {
  return [this](value_type x) -> value_type {
    return func_->derivative(x);
  };
}

}
