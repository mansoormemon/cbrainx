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

#include "cbrainx/neural_network.hh"

#include <stdexcept>
#include <utility>

#include <fmt/format.h>

#include "cbrainx/table.hh"

namespace cbx {

auto NeuralNetwork::input_size_check(shape_value_t input_size) -> void {
  if (input_size == 0) {
    throw std::invalid_argument{"cbx::NeuralNetwork::input_size_check: the number of neurons in the input "
                                "layer must be greater than zero"};
  }
}

// /////////////////////////////////////////////////////////////

NeuralNetwork::NeuralNetwork(shape_value_t input_size) {
  NeuralNetwork::input_size_check(input_size);
  input_size_ = input_size;
}

NeuralNetwork::NeuralNetwork(NeuralNetwork &&other) noexcept
    : input_size_{std::exchange(other.input_size_, {})}, layers_{std::move(other.layers_)} {}

// /////////////////////////////////////////////////////////////

auto NeuralNetwork::operator=(NeuralNetwork &&other) noexcept -> NeuralNetwork & {
  input_size_ = std::exchange(other.input_size_, {});
  layers_ = std::move(other.layers_);
  return *this;
}

// /////////////////////////////////////////////////////////////

auto NeuralNetwork::size() const -> size_type { return layers_.size(); }

auto NeuralNetwork::total_parameters() const -> size_type {
  return std::accumulate<const_iterator, size_type>(layers_.begin(), layers_.end(), {},
                                                    [](const auto &acc, const auto &layer) {
                                                      return acc + layer->parameters();
                                                    });
}

// /////////////////////////////////////////////////////////////

auto NeuralNetwork::front() const -> const_reference { return layers_.front(); }

auto NeuralNetwork::front() -> reference { return layers_.front(); }

auto NeuralNetwork::back() const -> const_reference { return layers_.back(); }

auto NeuralNetwork::back() -> reference { return layers_.back(); }

// /////////////////////////////////////////////////////////////

auto NeuralNetwork::begin() const -> const_iterator { return layers_.begin(); }

auto NeuralNetwork::begin() -> iterator { return layers_.begin(); }

auto NeuralNetwork::end() const -> const_iterator { return layers_.end(); }

auto NeuralNetwork::end() -> iterator { return layers_.end(); }

// /////////////////////////////////////////////////////////////

auto NeuralNetwork::pop() -> void { layers_.pop_back(); }

// /////////////////////////////////////////////////////////////

auto NeuralNetwork::forward_pass(const Tensor<f32> &input) -> Tensor<f32> {
  auto current = input;
  for (const auto &layer : layers_) {
    current = layer->forward_pass(current).output();
  }
  return current;
}

// /////////////////////////////////////////////////////////////

auto NeuralNetwork::show_summary() const -> void {
  auto table = Table{"Layer (Type)", "Neurons", "Property"};
  table.set_caption("Model Summary");
  auto input_layer = std::initializer_list<std::string>{"INPL0 (Input)", std::to_string(input_size_), "-"};
  table.add(input_layer);
  for (const auto &layer : layers_) {
    table.add({fmt::format("{} ({})", layer->to_string(), layer->type_name()), std::to_string(layer->neurons()),
               layer->property()});
  }
  table.show(true, Table::Large);
  fmt::print("Total Parameters: {}\n", this->total_parameters());
  fmt::print("Depth: {} layer(s)\n", this->size());
  fmt::print("{}\n", std::string(Table::Large, '='));
}

}
