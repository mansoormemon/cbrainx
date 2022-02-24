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

#ifndef CBRAINX__ABSTRACT_LAYER_HH_
#define CBRAINX__ABSTRACT_LAYER_HH_

#include <string>

#include "shape.hh"
#include "tensor.hh"
#include "type_aliases.hh"

namespace cbx {

enum class LayerType { Dense, Activation, SoftMax };

// /////////////////////////////////////////////////////////////////////////////////////////////

class AbstractLayer {
 public:
  using shape_value_t = Shape::value_type;
  using size_type = Tensor<f32>::size_type;
  using difference_type = Tensor<f32>::difference_type;

 private:
  i32 id_ = {};
  std::string name_ = "LYR";

 public:
  AbstractLayer() = default;

  explicit AbstractLayer(i32 id);

  explicit AbstractLayer(const std::string &name);

  AbstractLayer(i32 id, const std::string &name);

  virtual ~AbstractLayer() = default;

  // /////////////////////////////////////////////////////////////

  [[nodiscard]] auto id() const -> i32;

  auto set_id(i32 id) -> AbstractLayer &;

  [[nodiscard]] auto name() const -> std::string;

  auto set_name(const std::string &name) -> AbstractLayer &;

  // /////////////////////////////////////////////////////////////

  [[nodiscard]] virtual auto neurons() const -> size_type = 0;

  [[nodiscard]] virtual auto parameters() const -> size_type = 0;

  [[nodiscard]] virtual auto property() const -> std::string = 0;

  [[nodiscard]] virtual auto type() const -> LayerType = 0;

  // /////////////////////////////////////////////////////////////

  [[nodiscard]] virtual auto forward_pass(const Tensor<f32> &input) const -> Tensor<f32> = 0;

  // /////////////////////////////////////////////////////////////

  [[nodiscard]] virtual auto to_string() const -> std::string;

  [[nodiscard]] virtual auto type_name() const -> std::string;
};

}

#endif
