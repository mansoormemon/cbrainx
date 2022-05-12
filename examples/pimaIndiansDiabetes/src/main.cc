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

// Goal: A logistic regression model to predict where or not a person has diabetes.
// Reference: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

#include <fstream>
#include <iostream>

#include <cbrainx/cbrainx.hh>
#include <fmt/color.h>
#include <fmt/format.h>

#include "rapidcsv/rapidcsv.h"

using tensorF32 = cbx::Tensor<cbx::f32>;
using tensorU8 = cbx::Tensor<cbx::u8>;

auto read_dataset(cbx::str path, cbx::usize max_samples) -> std::pair<tensorF32, tensorF32> {
  auto file = rapidcsv::Document{path, rapidcsv::LabelParams{0, -1}, rapidcsv::SeparatorParams{','},
                                 rapidcsv::ConverterParams{true}};

  cbx::usize sample_count = std::min(max_samples, file.GetRowCount());
  cbx::usize features = file.GetColumnCount() - 1;

  auto data = tensorF32{{sample_count, features}};
  auto labels = tensorF32{{sample_count}}.reshape(2);

  for (cbx::usize i = {}; i < sample_count; ++i) {
    auto vec = file.GetRow<cbx::f32>(i);
    std::copy(vec.begin(), vec.end() - 1, data.begin() + (i * features));
    labels[i] = vec.back();
  }

  return {data, labels};
}

auto print_info(cbx::str msg, const tensorF32 &data, const tensorF32 &labels) -> void {
  fmt::print("{} => [\ndata = {},\nlabels = {}\n]\n", msg, data.meta_info(), labels.meta_info());
}

template <typename T>
auto print(const cbx::Tensor<T> &tensor, cbx::i32 count) -> void {
  auto [_, dim] = tensor.shape().template unwrap<2, cbx::i32>();
  cbx::i32 i = {};
  for (auto x : std::ranges::take_view{tensor, count * dim}) {
    fmt::print("{:<{}}", x, 16);
    i += 1;
    if (i % dim == 0) {
      fmt::print("\n");
    }
  }
}

template <typename T, typename U>
auto measure_accuracy(const cbx::Tensor<T> &truth, const cbx::Tensor<U> &predictions) {
  cbx::usize correct = {};
  for (cbx::usize i = {}; i < truth.total(); ++i) {
    correct += truth[i] == predictions[i];
  }
  return cbx::f32(correct) / truth.total();
}

auto main() -> cbx::i32 {
  const auto MAX_TRAINING_SAMPLES = -1;

  auto dataset_path = "res/diabetes.csv";

  auto [train_x, train_y] = read_dataset(dataset_path, MAX_TRAINING_SAMPLES);

  auto watch = cbx::Stopwatch{};

  fmt::print(fmt::emphasis::bold,
             "┌{0:─^{2}}┐\n"
             "│{1: ^{2}}│\n"
             "└{0:─^{2}}┘\n",
             "", "Datasets", 20);
  print_info("Training", train_x, train_y);

  auto [_, input_size] = train_x.shape().unwrap<2>();

  auto net = cbx::NeuralNet{{input_size}};
  net.add<cbx::DenseLayer>(48);
  net.add<cbx::ActivationLayer>(cbx::Activation::ArcTan);
  net.add<cbx::DenseLayer>(32);
  net.add<cbx::ActivationLayer>(cbx::Activation::LeakyReLU);
  net.add<cbx::DenseLayer>(24);
  net.add<cbx::ActivationLayer>(cbx::Activation::LeakyReLU);
  net.add<cbx::DenseLayer>(12);
  net.add<cbx::ActivationLayer>(cbx::Activation::TanH);
  net.add<cbx::DenseLayer>(1);
  net.add<cbx::ActivationLayer>(cbx::Activation::Sigmoid);
  net.show_summary();

  std::cout << "Running forward pass..." << std::endl;
  watch.start();
  auto out = net.forward_pass(train_x);
  watch.stop();
  std::cout << "Forward pass complete!" << std::endl;
  std::cout << "Time taken: " << watch.get_duration<std::chrono::seconds>() << "s." << std::endl;

  auto predictions = out | [](auto x) {
    return x >= 0.5;
  };

  auto lossFunc = cbx::BinaryCrossEntropy{};
  std::cout << "Loss (Before training): " << lossFunc(train_y, out) << std::endl;
  std::cout << "Accuracy: " << measure_accuracy(train_y, predictions) << std::endl;

  std::cout << "Output => " << out.meta_info() << std::endl;

  auto n = 5;
  std::cout << "Printing first " << n << " outputs..." << std::endl;
  print(out, n);

  net.backward_pass(train_x, train_y, 50, 1, cbx::Loss::BinaryCrossEntropy,
                    cbx::OptimizerWrapper{cbx::Optimizer::GradientDescent, 2e-4});

  out = net.forward_pass(train_x);
  predictions = out | [](auto x) {
    return x >= 0.5;
  };

  std::cout << "Loss (After training): " << lossFunc(train_y, out) << std::endl;
  std::cout << "Accuracy: " << measure_accuracy(train_y, predictions) << std::endl;

  std::cout << "Printing first " << n << " outputs..." << std::endl;
  print(out, n);

  return {};
}
