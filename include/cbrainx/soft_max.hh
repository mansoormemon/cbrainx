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

#ifndef CBRAINX__SOFT_MAX_HH_
#define CBRAINX__SOFT_MAX_HH_

#include "abstract_layer.hh"
#include "tensor.hh"
#include "type_aliases.hh"

namespace cbx {

class SoftMax : public AbstractLayer {
 private:
  shape_value_t neurons_ = {};

 public:
  explicit SoftMax(shape_value_t inputs);

  SoftMax(const SoftMax &other) = delete;

  SoftMax(SoftMax &&other) noexcept;

  ~SoftMax() override = default;

  // /////////////////////////////////////////////////////////////

  auto operator=(const SoftMax &other) -> SoftMax & = delete;

  auto operator=(SoftMax &&other) noexcept -> SoftMax &;

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