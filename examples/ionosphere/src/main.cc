#include <cbrainx/cbrainx.hh>

#include <iostream>
#include <vector>

#include "rapidcsv/rapidcsv.h"

namespace rapidcsv {

template <>
void Converter<cbx::f32>::ToVal(const std::string &pStr, cbx::f32 &pVal) const {
  if (pStr == "b") {
    pVal = 0;
  } else if (pStr == "g") {
    pVal = 1;
  } else {
    auto stream = std::stringstream{pStr};
    stream >> pVal;
  }
}

}

auto print_mat(const auto &mat) -> void {
  auto [rows, cols] = mat.shape().template unwrap<2>();
  std::cout << '[' << std::endl;
  for (auto r = 0UL; r < rows; ++r) {
    for (auto c = 0UL; c < cols; ++c) {
      std::cout << mat(r, c) << std::string(4, ' ');
    }
    std::cout << std::endl;
  }
  std::cout << ']' << std::endl;
};

auto read_dataset(cbx::str path_to_dataset) -> cbx::Dataset {
  auto csv_file = rapidcsv::Document{path_to_dataset, rapidcsv::LabelParams{-1, -1},
                                     rapidcsv::SeparatorParams{}, rapidcsv::ConverterParams{true}};

  cbx::shape_value_t sample_count = csv_file.GetRowCount();
  cbx::shape_value_t no_of_columns = csv_file.GetColumnCount();
  cbx::shape_value_t no_of_neurons = no_of_columns - 1;    // Last column contains labels.

  auto data = cbx::Tensor<cbx::f32>{{sample_count, no_of_neurons}};
  auto labels = cbx::Tensor<cbx::f32>{{sample_count}};

  for (cbx::shape_value_t i = {}; i < sample_count; ++i) {
    auto vec = csv_file.GetRow<cbx::f32>(i);
    std::copy(vec.begin(), vec.end() - 1, data.begin() + (i * no_of_neurons));
    labels[i] = vec.back();
  }

  return {std::move(data), std::move(labels)};
}

auto apply_threshold(const cbx::Tensor<cbx::f32> &in, cbx::f32 threshold) -> cbx::Tensor<cbx::f32> {
  return cbx::Tensor<cbx::f32>::custom(in.shape(), [threshold, it = in.begin()]() mutable {
    return *(it++) > threshold;
  });
}

auto measure_accuracy(const cbx::Dataset &dataset, const cbx::Tensor<cbx::f32> &predictions) -> cbx::f32 {
  cbx::u32 correct = {};
  auto y_hat = dataset.targets().begin();
  for (auto y : predictions) {
    correct += *(y_hat++) == y;
  }
  return static_cast<cbx::f32>(correct) / dataset.samples();
}

auto main() -> cbx::i32 {
  auto stopwatch = cbx::Stopwatch{};

  auto train_dataset = read_dataset("res/train.ionosphere.data.csv");
  auto test_dataset = read_dataset("res/test.ionosphere.data.csv");

  auto [_, neurons] = train_dataset.data().shape().unwrap<2>();

  auto net = cbx::NeuralNetwork{neurons};
  net.add<cbx::DenseLayer>(34);
  net.add<cbx::ActivationLayer>(cbx::Activation::Swish);
  net.add<cbx::DenseLayer>(1);
  net.add<cbx::ActivationLayer>(cbx::Activation::Sigmoid);
  net.show_summary();

  auto optimizer = cbx::GradientDescent{10e3};

  stopwatch.start();
  net.train(train_dataset, cbx::Loss::MeanSquaredError, optimizer, 64, 5);
  stopwatch.stop();

  std::cout << "train_dataset: " << train_dataset.meta_info() << std::endl;

  auto test_out = net.forward_pass(test_dataset.data());

  std::cout << "test_dataset: " << test_dataset.meta_info() << std::endl;
  std::cout << "test_out: " << test_out.meta_info() << std::endl;

  auto predictions = apply_threshold(test_out, 0.5);
  auto accuracy = measure_accuracy(test_dataset, predictions);
  std::cout << "Accuracy: " << std::round(accuracy * 100) << "%" << std::endl;
  std::cout << "Time taken for training: " << stopwatch.get_interval<std::chrono::milliseconds>() << " ms."
            << std::endl;

  return {};
}
