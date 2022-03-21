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

#include "cbrainx/loss_functions.hh"

#include <algorithm>

namespace cbx {

auto MeanSquaredError::calculate(x_iter_type x_begin, x_iter_type x_end, y_iter_type y_begin) const -> f32 {
  auto n = std::distance(x_begin, x_end);
  f32 total_loss = {};
  while (x_begin != x_end) {
    auto observed = *x_begin;
    auto predicted = *y_begin;
    total_loss += (predicted - observed) * (predicted - observed);
    ++x_begin, ++y_begin;
  }
  return total_loss / n;
}

auto MeanSquaredError::derivative(x_iter_type x_begin, x_iter_type x_end, y_iter_type y_begin) const -> f32 {
  auto n = std::distance(x_begin, x_end);
  f32 gradient = {};
  while (x_begin != x_end) {
    auto observed = *x_begin;
    auto predicted = *y_begin;
    gradient += 2 * (predicted - observed);
    ++x_begin, ++y_begin;
  }
  return gradient / n;
}

auto MeanSquaredError::type() const -> Loss { return Loss::MeanSquaredError; }

auto MeanSquaredError::to_string() const -> std::string { return "Loss function: Mean Squared Error"; }

auto MeanSquaredError::type_name() const -> std::string { return "MeanSquaredError"; }

// /////////////////////////////////////////////////////////////

auto BinaryCrossEntropy::calculate(x_iter_type x_begin, x_iter_type x_end, y_iter_type y_begin) const -> f32 {
  auto n = std::distance(x_begin, x_end);
  f32 total_loss = {};
  while (x_begin != x_end) {
    auto probability = *x_begin;
    auto label = *y_begin;
    total_loss += -((label * std::log(probability)) + ((1 - label) * std::log(1 - probability)));
    ;
    ++x_begin, ++y_begin;
  }
  return total_loss / n;
}

auto BinaryCrossEntropy::derivative(x_iter_type x_begin, x_iter_type x_end, y_iter_type y_begin) const -> f32 {}

auto BinaryCrossEntropy::type() const -> Loss { return Loss::BinaryCrossEntropy; }

auto BinaryCrossEntropy::to_string() const -> std::string { return "Loss function: Binary Cross Entropy"; }

auto BinaryCrossEntropy::type_name() const -> std::string { return "BinaryCrossEntropy"; }

// /////////////////////////////////////////////////////////////

auto CategoricalCrossEntropy::calculate(x_iter_type x_begin, x_iter_type x_end, y_iter_type y_begin) const
    -> f32 {
  auto n = std::distance(x_begin, x_end);
  auto positive_class_iter = std::find(y_begin, y_begin + n, 1);
  auto positive_class_index = std::distance(y_begin, positive_class_iter);
  return -std::log(x_begin[positive_class_index]);
}

auto CategoricalCrossEntropy::derivative(x_iter_type x_begin, x_iter_type x_end, y_iter_type y_begin) const
    -> f32 {}

auto CategoricalCrossEntropy::type() const -> Loss { return Loss::CategoricalCrossEntropy; }

auto CategoricalCrossEntropy::to_string() const -> std::string {
  return "Loss function: Categorical Cross Entropy";
}

auto CategoricalCrossEntropy::type_name() const -> std::string { return "CategoricalCrossEntropy"; }

// /////////////////////////////////////////////////////////////

auto SparseCrossEntropy::calculate(x_iter_type x_begin, x_iter_type, y_iter_type y_begin) const -> f32 {
  auto positive_class_index = *y_begin;
  return -std::log(x_begin[positive_class_index]);
}

auto SparseCrossEntropy::derivative(x_iter_type x_begin, x_iter_type x_end, y_iter_type y_begin) const -> f32 {}

auto SparseCrossEntropy::type() const -> Loss { return Loss::SparseCrossEntropy; }

auto SparseCrossEntropy::to_string() const -> std::string { return "Loss function: Sparse Cross Entropy"; }

auto SparseCrossEntropy::type_name() const -> std::string { return "SparseCrossEntropy"; }

// /////////////////////////////////////////////////////////////

auto LossFunctionFactory::make(Loss loss_type) -> std::shared_ptr<LossFunction> {
  switch (loss_type) {
    case Loss::MeanSquaredError: {
      return std::make_shared<MeanSquaredError>();
    }
    case Loss::BinaryCrossEntropy: {
      return std::make_shared<BinaryCrossEntropy>();
    }
    case Loss::CategoricalCrossEntropy: {
      return std::make_shared<CategoricalCrossEntropy>();
    }
    case Loss::SparseCrossEntropy: {
      return std::make_shared<SparseCrossEntropy>();
    }
  }
}

}
