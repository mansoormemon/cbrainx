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

#include "cbrainx/neuralNet.hh"

#include <utility>

#include <fmt/color.h>
#include <fmt/core.h>

namespace cbx {

// /////////////////////////////////////////////
// Helpers
// /////////////////////////////////////////////

auto NeuralNet::_s_validate_input_shape(const Shape &shape) -> void {
  auto cur_rank = shape.rank();
  if (cur_rank < tensor_type::VECTOR_RANK) {
    throw RankError{
        "cbx::NeuralNet::_s_validate_input_shape: the input must be at least one dimensional [rank = {}]",
        cur_rank};
  }
}

auto NeuralNet::_m_match_input_shape(const Shape &shape) -> void {
  auto sliced_shape = Shape{shape.begin() + 1, shape.end()};
  if (input_shape_ != sliced_shape) {
    throw ShapeError{"cbx::NeuralNet::_m_match_input_shape: shapes mismatch [expected = {}, received = {}]",
                     input_shape_.to_string(), sliced_shape.to_string()};
  }
}

// /////////////////////////////////////////////
// Constructors (and Destructors)
// /////////////////////////////////////////////

NeuralNet::NeuralNet(const Shape &input_shape) {
  _s_validate_input_shape(input_shape);
  input_shape_ = input_shape;
}

NeuralNet::NeuralNet(NeuralNet &&other) noexcept
    : input_shape_{std::move(other.input_shape_)}, layers_{std::move(other.layers_)} {}

// /////////////////////////////////////////////
// Assignment Operators
// /////////////////////////////////////////////

auto NeuralNet::operator=(NeuralNet &&other) noexcept -> NeuralNet & {
  input_shape_ = std::move(other.input_shape_);
  layers_ = std::move(other.layers_);
  return *this;
}

// /////////////////////////////////////////////
// Element Access
// /////////////////////////////////////////////

auto NeuralNet::front() const -> const_reference { return layers_.front(); }

auto NeuralNet::front() -> reference { return layers_.front(); }

auto NeuralNet::back() const -> const_reference { return layers_.back(); }

auto NeuralNet::back() -> reference { return layers_.back(); }

// /////////////////////////////////////////////
// Iterators
// /////////////////////////////////////////////

auto NeuralNet::cbegin() const -> const_iterator { return layers_.cbegin(); }

auto NeuralNet::begin() const -> const_iterator { return layers_.begin(); }

auto NeuralNet::begin() -> iterator { return layers_.begin(); }

auto NeuralNet::crbegin() const noexcept -> const_reverse_iterator { return layers_.crbegin(); }

auto NeuralNet::rbegin() const noexcept -> const_reverse_iterator { return layers_.rbegin(); }

auto NeuralNet::rbegin() noexcept -> reverse_iterator { return layers_.rbegin(); }

auto NeuralNet::cend() const -> const_iterator { return layers_.cend(); }

auto NeuralNet::end() const -> const_iterator { return layers_.end(); }

auto NeuralNet::end() -> iterator { return layers_.end(); }

auto NeuralNet::crend() const noexcept -> const_reverse_iterator { return layers_.crend(); }

auto NeuralNet::rend() const noexcept -> const_reverse_iterator { return layers_.rend(); }

auto NeuralNet::rend() noexcept -> reverse_iterator { return layers_.rend(); }

// /////////////////////////////////////////////
// Query Functions
// /////////////////////////////////////////////

auto NeuralNet::size() const -> size_type { return layers_.size(); }

auto NeuralNet::total_parameters() const -> size_type {
  return std::accumulate(layers_.begin(), layers_.end(), size_type{}, [](auto acc, const auto &layer) {
    return acc + layer->parameters();
  });
}

// /////////////////////////////////////////////
// Informative
// /////////////////////////////////////////////

auto NeuralNet::show_summary() const -> void {
  enum Width { Small = 24, Medium = 32, Large = 40, XLarge = 48 };

  enum : char { Minus = '-', Plus = '+', Asterisk = '*', Hash = '#', Underscore = '_', Equal = '=' };

  auto header = {"Layer (Type)", "Neurons", "Property"};
  auto cols = header.size();

  auto attributes =
      std::list<std::pair<std::string, std::string>>{{"Total Parameters", std::to_string(total_parameters())},
                                                     {"Depth", std::to_string(size())}};

  auto col_width = Width::Medium;
  auto table_width = cols * col_width;

  auto print_caption = [table_width](str caption) {
    fmt::print(fmt::emphasis::bold, "{:^{}}\n", caption, table_width);
  };

  auto print_header = [col_width](const auto &header) {
    for (const auto &heading : header) {
      fmt::print("{:{}}", heading, col_width);
    }
    fmt::print("\n");
  };

  auto print_row = [col_width](std::initializer_list<std::string> list) {
    for (const auto &item : list) {
      fmt::print("{:{}}", item, col_width);
    }
    fmt::print("\n");
  };

  auto print_attributes = [](const auto &attributes) {
    if (attributes.empty()) {
      fmt::print("No attributes defined.\n");
    }
    for (const auto &[label, value] : attributes) {
      fmt::print("{}: {}\n", label, value);
    }
  };

  auto print_separator = [table_width](auto sep) {
    fmt::print("{}\n", std::string(table_width, sep));
  };

  print_caption("MODEL SUMMARY");
  print_separator(Equal);
  print_header(header);
  print_separator(Equal);
  print_row({"INPL0 (Input)", input_shape_.to_string(), "-"});
  print_separator(Plus);
  for (const auto &layer : layers_) {
    print_row({fmt::format("{} ({})", layer->to_string(), layer->type_name()),
               Shape{layer->neurons()}.to_string(), layer->property()});
    print_separator(Minus);
  }
  print_attributes(attributes);
  print_separator(Equal);
}

// /////////////////////////////////////////////
// Modifiers
// /////////////////////////////////////////////

auto NeuralNet::pop() -> void { layers_.pop_back(); }

// /////////////////////////////////////////////
// Core Functionality
// /////////////////////////////////////////////

auto NeuralNet::forward_pass(const tensor_type &input) -> tensor_type {
  _m_match_input_shape(input.shape());
  auto current = input;
  for (const auto &layer : layers_) {
    current = layer->forward_pass(current);
  }
  return current;
}

}
