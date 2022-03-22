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
  using value_type = f32;
  using container = Tensor<f32>;

  using reference = container::reference;
  using const_reference = container::const_reference;

  using pointer = container::pointer;
  using const_pointer = container::const_pointer;

  using size_type = container::size_type;
  using difference_type = container::difference_type;

  using iterator = container::iterator;
  using const_iterator = container::const_iterator;

  using container_reference = container &;
  using container_const_reference = const container &;

 private:
  i32 id_ = {};
  std::string name_ = "LYR";

 protected:
  container input_ = {};
  container output_ = {};

 public:
  AbstractLayer() = default;

  AbstractLayer(const AbstractLayer &other) = delete;

  AbstractLayer(AbstractLayer &&other) noexcept;

  explicit AbstractLayer(i32 id);

  explicit AbstractLayer(std::string name);

  AbstractLayer(i32 id, std::string name);

  virtual ~AbstractLayer() = default;

  // /////////////////////////////////////////////////////////////

  auto operator=(const AbstractLayer &other) -> AbstractLayer & = delete;

  auto operator=(AbstractLayer &&other) noexcept -> AbstractLayer &;

  // /////////////////////////////////////////////////////////////

  [[nodiscard]] auto id() const -> i32;

  auto set_id(i32 id) -> AbstractLayer &;

  [[nodiscard]] auto name() const -> std::string;

  auto set_name(std::string name) -> AbstractLayer &;

  [[nodiscard]] auto input() const -> container_const_reference;

  [[nodiscard]] auto output() const -> container_const_reference;

  // /////////////////////////////////////////////////////////////

  [[nodiscard]] virtual auto neurons() const -> size_type = 0;

  [[nodiscard]] virtual auto parameters() const -> size_type = 0;

  [[nodiscard]] virtual auto property() const -> std::string = 0;

  [[nodiscard]] virtual auto type() const -> LayerType = 0;

  // /////////////////////////////////////////////////////////////

  [[nodiscard]] virtual auto forward_pass(container_const_reference input) -> AbstractLayer & = 0;

  // /////////////////////////////////////////////////////////////

  [[nodiscard]] virtual auto to_string() const -> std::string;

  [[nodiscard]] virtual auto type_name() const -> std::string;
};

}

#endif
