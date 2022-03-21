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

#include <cmath>
#include <list>
#include <memory>

#include "abstract_layer.hh"
#include "dataset.hh"
#include "loss_functions.hh"
#include "optimizers.hh"
#include "type_aliases.hh"
#include "utility.hh"

namespace cbx {

template <typename T>
concept ConcreteLayer = std::is_base_of_v<AbstractLayer, T> and not std::is_abstract_v<T>;

class NeuralNetwork {
 public:
  using value_type = std::shared_ptr<AbstractLayer>;

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
  shape_value_t input_size_ = {};
  container layers_ = {};

  // /////////////////////////////////////////////////////////////

  static auto input_size_check(shape_value_t input_size) -> void;

 public:
  explicit NeuralNetwork(shape_value_t input_size);

  // For future - Write a deep copy constructor.
  NeuralNetwork(const NeuralNetwork &other) = delete;

  NeuralNetwork(NeuralNetwork &&other) noexcept;

  ~NeuralNetwork() = default;

  // /////////////////////////////////////////////////////////////

  // For future - Write a deep copy assignment operator.
  auto operator=(const NeuralNetwork &other) -> NeuralNetwork & = delete;

  auto operator=(NeuralNetwork &&other) noexcept -> NeuralNetwork &;

  // /////////////////////////////////////////////////////////////

  template <ConcreteLayer LayerType, typename... Args>
  auto add(Args... args) -> const_reference {
    size_type neurons_in_last_layer = layers_.empty() ? input_size_ : layers_.back()->neurons();
    layers_.emplace_back(std::make_shared<LayerType>(neurons_in_last_layer, args...));
    layers_.back()->set_id(static_cast<i32>((layers_.size())));
    return layers_.back();
  }

  // /////////////////////////////////////////////////////////////

  [[nodiscard]] auto size() const -> size_type;

  [[nodiscard]] auto total_parameters() const -> size_type;

  // /////////////////////////////////////////////////////////////

  [[nodiscard]] auto front() const -> const_reference;

  auto front() -> reference;

  [[nodiscard]] auto back() const -> const_reference;

  auto back() -> reference;

  // /////////////////////////////////////////////////////////////

  [[nodiscard]] auto begin() const -> const_iterator;

  auto begin() -> iterator;

  [[nodiscard]] auto end() const -> const_iterator;

  auto end() -> iterator;

  // /////////////////////////////////////////////////////////////

  auto pop() -> void;

  // /////////////////////////////////////////////////////////////

  [[nodiscard]] auto forward_pass(const Tensor<f32> &input) -> Tensor<f32>;

  // /////////////////////////////////////////////////////////////

  auto show_summary() const -> void;

  // /////////////////////////////////////////////////////////////

  auto train(const Dataset &dataset, Loss loss_type, const Optimizer auto &optimizer, size_dt batch_size,
             size_dt epochs, Verbosity verbosity = Verbosity::L3) -> void {
    decltype(auto) data = dataset.data();
    decltype(auto) targets = dataset.targets();
    size_dt sample_count = dataset.samples();
    size_dt total_batches = std::ceil(static_cast<f32>(sample_count) / batch_size);
    auto loss_func = LossFunctionFactory::make(loss_type);

    verbose(Verbosity::L2, verbosity, "Initiating training...\n");
    verbose(Verbosity::L2, verbosity, "Samples: {}\n", sample_count);
    verbose(Verbosity::L2, verbosity, "Batch size: {}, Total batches: {}\n", batch_size, total_batches);
    verbose(Verbosity::L2, verbosity, "{}\n", loss_func->to_string());

    auto out = this->forward_pass(data);

    for (size_dt epoch = {}; epoch < epochs; ++epoch) {
      verbose(Verbosity::L3, verbosity, "Epoch {} of {}: [\n", epoch + 1, epochs);

      for (size_dt batch = {}; batch < total_batches; ++batch) {
        auto sample_stride = out.total() / sample_count;
        auto offset = batch * batch_size;
        auto n = std::min(batch_size, sample_count - offset);

        auto out_begin = out.begin() + (offset * sample_stride);
        auto out_end = out_begin + (n * sample_stride);

        auto targets_begin = targets.begin() + (offset * sample_stride);
        auto mean_loss = loss_func->calculate(out_begin, out_end, targets_begin);

        verbose(Verbosity::L3, verbosity, "\tBatch # {}/{}: {{ ", batch + 1, total_batches);
        verbose(Verbosity::L3, verbosity, "samples={}", n);
        verbose(Verbosity::L3, verbosity, ", mean_loss: {} }}\n", mean_loss);
      }
      verbose(Verbosity::L3, verbosity, "]\n");
    }
    verbose(Verbosity::L2, verbosity, "Training complete!\n");
  }
};

}

#endif
