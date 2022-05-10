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

#include <iostream>
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

auto NeuralNet::_m_match_input_shape(const Shape &shape) const -> void {
  auto sliced_shape = shape.slice(1);
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

auto NeuralNet::forward_pass(tensor_type input) const -> tensor_type {
  _m_match_input_shape(input.shape());
  for (const auto &layer : layers_) {
    // The output of one layer becomes the input of the next.
    input = layer->forward_pass(input);
  }
  return input;
}

auto NeuralNet::backward_pass(tensor_type x, tensor_type y, usize epochs, usize batch_size, Loss loss_type,
                              OptimizerWrapper optimizer) -> void {
  const auto PROGRESS_BAR_WIDTH = 36;

  auto loss_func = LossFuncWrapper{loss_type};
  auto [samples] = x.shape().unwrap<1, f32>();
  size_type batches = std::ceil(samples / batch_size);
  size_type x_stride = x.total() / samples, y_stride = y.total() / samples;

  // Initializes the metrics for an epoch.
  auto init_metrics = [epochs, PROGRESS_BAR_WIDTH](auto epoch) {
    fmt::print("Epoch {} of {}: [{: ^{}}] 0%", epoch, epochs, "", PROGRESS_BAR_WIDTH);
    std::cout << std::flush;
  };

  // Updates the metrics for an epoch.
  auto update_metrics = [this, epochs, batches, PROGRESS_BAR_WIDTH](auto epoch, auto batch, auto total_loss) {
    f32 ratio = f32(batch) / batches;
    f32 percentage = i32(ratio * 100.0);
    size_type filled = ratio * PROGRESS_BAR_WIDTH;

    fmt::print("\rEpoch {} of {}: [{}{}] {}%, mean_loss = {:.6f}", epoch, epochs, std::string(filled, '#'),
               std::string(PROGRESS_BAR_WIDTH - filled, ' '), percentage, total_loss / batch);
    std::cout << std::flush;
  };

  // Closing clause for an epoch's metrics.
  auto end_metrics = []() {
    fmt::print("\n");
  };

  for (size_type e = {}; e < epochs; ++e) {
    f32 total_loss = {};
    init_metrics(e + 1);
    for (size_type b = {}; b < batches; ++b) {
      // Offset of the current batch.
      size_type offset = b * batch_size;
      // The number of elements in the current batch.
      size_type n = std::min(batch_size, size_type(samples - offset));

      // Determine the boundaries of the current batch.
      auto x_begin = x.begin() + (offset * x_stride), y_begin = y.begin() + (offset * y_stride);

      auto x_in = tensor_type{{n, x_stride}, x_begin};
      auto y_hat = forward_pass(x_in);
      tensor_type y_true = {(loss_type == Loss::SparseCrossEntropy ? Shape{n} : Shape{n, y_stride}), y_begin};
      total_loss += loss_func(y_true, y_hat);
      auto dL = loss_func.derivative(y_true, y_hat);
      auto gradient = dL * y_hat;
      for (auto &layer : layers_ | std::views::reverse) {
        // Downstream gradient of one layer becomes the upstream gradient of the next.
        gradient = layer->backward_pass(gradient, optimizer);
      }
      // Update the optimizer's state.
      ++optimizer;
      // Update the metrics.
      update_metrics(e + 1, b + 1, total_loss);
    }
    end_metrics();
  }
}

}
