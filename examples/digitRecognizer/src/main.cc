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

// Goal: A Neural Network to Recognize Handwritten Digits
// Reference:
// https://www.digitalocean.com/community/tutorials/how-to-build-a-neural-network-to-recognize-handwritten-digits-with-tensorflow

#include <fstream>
#include <iostream>

#include <cbrainx/cbrainx.hh>
#include <fmt/color.h>
#include <fmt/format.h>

using tensorF32 = cbx::Tensor<cbx::f32>;
using tensorU8 = cbx::Tensor<cbx::u8>;

auto read_int(std::fstream &file) -> cbx::i32 {
  auto reverse_int = [](cbx::i32 i) {
    cbx::u8 b1 = {}, b2 = {}, b3 = {}, b4 = {};
    b1 = i & 255;
    b2 = (i >> 8) & 255;
    b3 = (i >> 16) & 255;
    b4 = (i >> 24) & 255;
    return (cbx::i32(b1) << 24) + (cbx::i32(b2) << 16) + (cbx::i32(b3) << 8) + b4;
  };

  cbx::i32 num = {};
  file.read(reinterpret_cast<char *>(&num), sizeof(cbx::i32));
  // Flip endianness.
  return reverse_int(num);
}

// File Signature:
//
// [offset] [type]          [value]          [description]
// 0000     32 bit integer  0x00000803(2051) magic number
// 0004     32 bit integer  60000            number of images
// 0008     32 bit integer  28               number of rows
// 0012     32 bit integer  28               number of columns
// 0016     unsigned byte   ??               pixel
// 0017     unsigned byte   ??               pixel
// ........
// xxxx     unsigned byte   ??               pixel
//
// The training set contains 60000 samples, and the test set 10000 samples.
//
// Reference: http://yann.lecun.com/exdb/mnist
auto read_images(cbx::str path, cbx::i32 max_samples) -> tensorF32 {
  auto file = std::fstream{path, std::ios::in | std::ios::binary};

  // Ignore magic number.
  file.seekg(4);

  cbx::u32 sample_num = std::min(max_samples, read_int(file));
  cbx::u32 img_width = read_int(file);
  cbx::u32 img_height = read_int(file);

  // Create a tensor after normalizing samples.
  return tensorF32::custom({sample_num, img_height * img_width}, [&file]() {
    cbx::u8 byte = {};
    file >> byte;
    return byte / 255.0;
  });
}

// File signature:
//
// [offset] [type]          [value]          [description]
// 0000     32 bit integer  0x00000801(2049) magic number (MSB first)
// 0004     32 bit integer  60000            number of items
// 0008     unsigned byte   ??               label
// 0009     unsigned byte   ??               label
//........
// xxxx     unsigned byte   ??               label
//
// The labels values are 0 to 9.
//
// Reference: http://yann.lecun.com/exdb/mnist
auto read_labels(cbx::str path, cbx::i32 max_samples) -> tensorF32 {
  auto file = std::fstream{path, std::ios::in | std::ios::binary};

  // Ignore magic number.
  file.seekg(4);

  cbx::u32 sample_num = std::min(max_samples, read_int(file));

  return tensorF32::custom({sample_num}, [&file]() {
    return file.get();
  });
}

auto print_info(cbx::str msg, const tensorF32 &images, const tensorF32 &labels) -> void {
  fmt::print("{} => [\nimages = {},\nlabels = {}\n]\n", msg, images.meta_info(), labels.meta_info());
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

auto argmax(const tensorF32 &input) -> tensorU8 {
  auto samples = input.is_matrix() ? input.shape().front() : 1;
  auto neurons = input.shape().back();
  auto result = tensorU8{{samples}};
  for (cbx::usize i = {}; i < samples; i += 1) {
    auto begin = input.begin() + (neurons * i);
    auto end = begin + neurons;
    auto it = std::max_element(begin, end);
    result[i] = std::distance(begin, it);
  }
  return result;
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
  const auto MAX_TRAINING_SAMPLES = 512;
  const auto MAX_TESTING_SAMPLES = 256;

  auto train_images_path = "res/train/images.idx3-ubyte";
  auto train_labels_path = "res/train/labels.idx1-ubyte";

  auto test_images_path = "res/test/images.idx3-ubyte";
  auto test_labels_path = "res/test/labels.idx1-ubyte";

  auto train_images = read_images(train_images_path, MAX_TRAINING_SAMPLES);
  auto train_labels = read_labels(train_labels_path, MAX_TRAINING_SAMPLES);

  auto test_images = read_images(test_images_path, MAX_TESTING_SAMPLES);
  auto test_labels = read_labels(test_labels_path, MAX_TESTING_SAMPLES);

  auto watch = cbx::Stopwatch{};

  fmt::print(fmt::emphasis::bold,
             "┌{0:─^{2}}┐\n"
             "│{1: ^{2}}│\n"
             "└{0:─^{2}}┘\n",
             "", "Datasets", 20);
  print_info("Training", train_images, train_labels);
  print_info("Testing", test_images, test_labels);

  auto [_, input_size] = train_images.shape().unwrap<2>();

  auto net = cbx::NeuralNet{{input_size}};
  net.add<cbx::DenseLayer>(512);
  net.add<cbx::ActivationLayer>(cbx::Activation::LeakyReLU);
  net.add<cbx::DenseLayer>(256);
  net.add<cbx::ActivationLayer>(cbx::Activation::ELU);
  net.add<cbx::DenseLayer>(128);
  net.add<cbx::ActivationLayer>(cbx::Activation::ArcTan);
  net.add<cbx::DenseLayer>(10);
  net.add<cbx::ActivationLayer>(cbx::Activation::TanH);
  net.add<cbx::Softmax>();
  net.show_summary();

  std::cout << "Running forward pass..." << std::endl;
  watch.start();
  auto out = net.forward_pass(test_images);
  watch.stop();
  std::cout << "Forward pass complete!" << std::endl;
  std::cout << "Time taken: " << watch.get_duration<std::chrono::seconds>() << "s." << std::endl;

  auto lossFunc = cbx::SparseCrossEntropy{};
  std::cout << "Loss: " << lossFunc(test_labels, out) << std::endl;

  auto predictions = argmax(out).reshape(2);
  std::cout << "Accuracy: " << measure_accuracy(test_labels, predictions) << std::endl;

  std::cout << "Output => " << out.meta_info() << std::endl;

  auto n = 5;
  std::cout << "Printing first " << n << " outputs..." << std::endl;
  print(out, n);

  net.backward_pass(train_images, train_labels, 10, 128, cbx::Loss::SparseCrossEntropy,
                    cbx::OptimizerWrapper{cbx::Optimizer::GradientDescent, 1e-4});

  out = net.forward_pass(test_images);
  std::cout << "Loss: " << lossFunc(test_labels, out) << std::endl;
  predictions = argmax(out).reshape(2);
  std::cout << "Accuracy: " << measure_accuracy(test_labels, predictions) << std::endl;

  std::cout << "Printing first " << n << " outputs..." << std::endl;
  print(out, n);

  return {};
}
