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

#ifndef CBRAINX__NEURAL_NETWORK_HH_
#define CBRAINX__NEURAL_NETWORK_HH_

#include <list>
#include <memory>

#include "abstract_layer.hh"
#include "type_aliases.hh"

namespace cbx {

class NeuralNetwork {
 public:
  using value_type = std::unique_ptr<AbstractLayer>;

  using reference = value_type &;
  using const_reference = const value_type &;

  using pointer = value_type *;
  using const_pointer = const value_type *;

  using container = std::list<value_type>;

  using size_type = typename container::size_type;
  using difference_type = typename container::difference_type;

  using iterator = typename container::iterator;
  using const_iterator = typename container::const_iterator;

 private:
  AbstractLayer::shape_value_t input_size_ = {};
  container layers_ = {};

  // /////////////////////////////////////////////////////////////

  static auto input_size_check(AbstractLayer::shape_value_t input_size) -> void;

 public:
  explicit NeuralNetwork(AbstractLayer::shape_value_t input_size);

  NeuralNetwork(const NeuralNetwork &other) = default;

  NeuralNetwork(NeuralNetwork &&other) noexcept;

  ~NeuralNetwork() = default;

  // /////////////////////////////////////////////////////////////

  auto operator=(const NeuralNetwork &other) -> NeuralNetwork & = default;

  auto operator=(NeuralNetwork &&other) noexcept -> NeuralNetwork &;

  // /////////////////////////////////////////////////////////////

  template <typename LayerType, typename... Args>
  auto add(Args... args) -> NeuralNetwork & {
    size_type neurons_in_last_layer = layers_.empty() ? input_size_ : layers_.back()->neurons();
    layers_.emplace_back(std::make_unique<LayerType>(neurons_in_last_layer, args...));
    layers_.back()->set_id(layers_.size());
    return *this;
  }

  // /////////////////////////////////////////////////////////////

  [[nodiscard]] auto size() const -> size_type;

  [[nodiscard]] auto total_parameters() const -> size_type;

  // /////////////////////////////////////////////////////////////

  [[nodiscard]] auto front() const -> const_reference;

  [[nodiscard]] auto front() -> reference;

  [[nodiscard]] auto back() const -> const_reference;

  [[nodiscard]] auto back() -> reference;

  // /////////////////////////////////////////////////////////////

  [[nodiscard]] auto begin() const -> const_iterator;

  [[nodiscard]] auto begin() -> iterator;

  [[nodiscard]] auto end() const -> const_iterator;

  [[nodiscard]] auto end() -> iterator;

  // /////////////////////////////////////////////////////////////

  auto pop() -> void;

  // /////////////////////////////////////////////////////////////

  [[nodiscard]] auto forward_pass(Tensor<f32> input) const -> Tensor<f32>;

  // /////////////////////////////////////////////////////////////

  auto show_summary() const -> void;
};
}

#endif
