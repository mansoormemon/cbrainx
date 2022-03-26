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

#include "cbrainx/activation_layer.hh"

#include <utility>

namespace cbx {

ActivationLayer::ActivationLayer(shape_value_t inputs, Activation activation)
    : AbstractLayer{"ACTL"}, neurons_{inputs} {
  switch (activation) {
    case Activation::ArcTan: {
      act_func_ = std::make_unique<ArcTan>();
      break;
    }
    case Activation::BinaryStep: {
      act_func_ = std::make_unique<BinaryStep>();
      break;
    }
    case Activation::ELU: {
      act_func_ = std::make_unique<ELU>();
      break;
    }
    case Activation::Gaussian: {
      act_func_ = std::make_unique<Gaussian>();
      break;
    }
    case Activation::GELU: {
      act_func_ = std::make_unique<GELU>();
      break;
    }
    case Activation::LeakyReLU: {
      act_func_ = std::make_unique<LeakyReLU>();
      break;
    }
    case Activation::Linear: {
      act_func_ = std::make_unique<Linear>();
      break;
    }
    case Activation::ReLU: {
      act_func_ = std::make_unique<ReLU>();
      break;
    }
    case Activation::Sigmoid: {
      act_func_ = std::make_unique<Sigmoid>();
      break;
    }
    case Activation::SoftPlus: {
      act_func_ = std::make_unique<SoftPlus>();
      break;
    }
    case Activation::Swish: {
      act_func_ = std::make_unique<Swish>();
      break;
    }
    case Activation::TanH: {
      act_func_ = std::make_unique<TanH>();
      break;
    }
  }
}

ActivationLayer::ActivationLayer(ActivationLayer &&other) noexcept
    : neurons_{std::exchange(other.neurons_, {})}, act_func_{std::move(other.act_func_)} {}

// /////////////////////////////////////////////////////////////

auto ActivationLayer::operator=(ActivationLayer &&other) noexcept -> ActivationLayer & {
  neurons_ = std::exchange(other.neurons_, {});
  act_func_ = std::move(other.act_func_);
  return *this;
}

// /////////////////////////////////////////////////////////////

auto ActivationLayer::neurons() const -> size_type { return neurons_; }

auto ActivationLayer::parameters() const -> size_type { return {}; }

auto ActivationLayer::property() const -> std::string { return act_func_->to_string(); }

auto ActivationLayer::type() const -> LayerType { return LayerType::Activation; }

auto ActivationLayer::output() const -> const Tensor<f32> & { return output_; }

// /////////////////////////////////////////////////////////////

auto ActivationLayer::forward_pass(const Tensor<f32> &input) -> AbstractLayer & {
  output_ = Tensor<f32>{input.shape()};
  std::transform(input.begin(), input.end(), output_.begin(), [this](const auto &x) {
    return act_func_->operator()(x);
  });
  return *this;
}

}
