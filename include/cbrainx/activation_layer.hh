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

#ifndef CBRAINX__ACTIVATION_LAYER_HH_
#define CBRAINX__ACTIVATION_LAYER_HH_

#include <memory>

#include "abstract_layer.hh"
#include "activation_functions.hh"
#include "type_aliases.hh"

namespace cbx {

enum class Activation {
  ArcTan,
  BinaryStep,
  ELU,
  Gaussian,
  GELU,
  LeakyReLU,
  Linear,
  ReLU,
  Sigmoid,
  SoftPlus,
  Swish,
  TanH
};

class ActivationLayer : public AbstractLayer {
 private:
  shape_value_t neurons_ = {};
  std::unique_ptr<ActivationFunction> act_func_ = {};

 public:
  ActivationLayer(shape_value_t inputs, Activation activation);

  ActivationLayer(const ActivationLayer &other) = delete;

  ActivationLayer(ActivationLayer &&other) noexcept;

  ~ActivationLayer() override = default;

  // /////////////////////////////////////////////////////////////

  auto operator=(const ActivationLayer &other) -> ActivationLayer & = delete;

  auto operator=(ActivationLayer &&other) noexcept -> ActivationLayer &;

  // /////////////////////////////////////////////////////////////

  [[nodiscard]] auto neurons() const -> size_type override;

  [[nodiscard]] auto parameters() const -> size_type override;

  [[nodiscard]] auto property() const -> std::string override;

  [[nodiscard]] auto type() const -> LayerType override;

  // /////////////////////////////////////////////////////////////

  [[nodiscard]] auto forward_pass(const Tensor<f32> &input) const -> Tensor<f32> override;
};

}

#endif
