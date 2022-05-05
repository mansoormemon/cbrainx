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

#include "cbrainx/lossFunctions.hh"

#include "cbrainx/exceptions.hh"

namespace cbx {

// /////////////////////////////////////////////////////////////
// LossFunction
// /////////////////////////////////////////////////////////////

// /////////////////////////////////////////////
// Helpers
// /////////////////////////////////////////////

auto LossFunction::_s_check_rank_range(usize rank, usize lower_bound, usize upper_bound) -> void {
  if (rank < lower_bound or rank > upper_bound) {
    throw ShapeError{"cbx::LossFunction::_s_check_rank_in_range: rank = {} must be in the range [{}, {}]", rank,
                     lower_bound, upper_bound};
  }
}

auto LossFunction::_s_check_shape_equality(const Shape &a, const Shape &b) -> void {
  if (a != b) {
    throw ShapeError{"cbx::LossFunction::_s_check_shape_equality: a = {} must be equal to b = {}",
                     a.to_string(), b.to_string()};
  }
}

// /////////////////////////////////////////////////////////////
// Loss Functions
// /////////////////////////////////////////////////////////////

// /////////////////////////////////////////////
// Interface
// /////////////////////////////////////////////

auto MeanSquaredError::type() const -> Loss { return Loss::MeanSquaredError; }

auto MeanSquaredError::to_string() const -> std::string { return "Mean Squared Error"; }

auto MeanSquaredError::type_name() const -> std::string { return "MeanSquaredError"; }

auto MeanSquaredError::operator()(const tensor_type &y_true, const tensor_type &y_pred) const -> value_type {
  // This function implements the subsequent operation.
  //
  // Formula: Ĺ = 1 / n  ⅀ [ὶ = 1, n] (Ýὶ - Yὶ)²
  //
  // where:
  //  Ĺ  - Loss function
  //  Y  - Observations
  //  Ý  - Predictions
  //  Yὶ - ὶᵗʰ observed value   => Range = (-∞, ∞; 𝟄 R)
  //  Ýὶ - ὶᵗʰ predicted value  => Range = (-∞, ∞; 𝟄 R)
  //  n  - Number of observations
  //
  // For a batch of samples, the mean loss is returned.

  _s_check_rank_range(y_true.rank(), tensor_type::SCALAR_RANK, tensor_type::MATRIX_RANK);
  _s_check_shape_equality(y_true.shape(), y_pred.shape());

  auto y_true_begin = y_true.begin(), y_true_end = y_true.end();
  auto y_pred_begin = y_pred.begin();

  value_type total_quadratic_loss = {};
  while (y_true_begin != y_true_end) {
    auto truth = *y_true_begin, pred = *y_pred_begin;
    total_quadratic_loss += (pred - truth) * (pred - truth);
    ++y_true_begin, ++y_pred_begin;
  }
  return total_quadratic_loss / y_true.total();
}

auto MeanSquaredError::derivative(const tensor_type &y_true, const tensor_type &y_pred) const -> value_type {
  // This function implements the subsequent operation.
  //
  // Derivative: ẟ / ẟÝὶ Ĺ = 1 / n ⅀ [ὶ = 1, n] 2 . (Ýὶ - Yὶ)
  //
  // where:
  //  Ĺ  - Loss function
  //  Y  - Observations
  //  Ý  - Predictions
  //  Yὶ - ὶᵗʰ observed value   => Range = (-∞, ∞; 𝟄 R)
  //  Ýὶ - ὶᵗʰ predicted value  => Range = (-∞, ∞; 𝟄 R)
  //  n  - Number of observations
  //
  // For a batch of samples, the mean loss is returned.

  _s_check_rank_range(y_true.rank(), tensor_type::SCALAR_RANK, tensor_type::MATRIX_RANK);
  _s_check_shape_equality(y_true.shape(), y_pred.shape());

  auto y_true_begin = y_true.begin(), y_true_end = y_true.end();
  auto y_pred_begin = y_pred.begin();

  value_type gradient = {};
  while (y_true_begin != y_true_end) {
    auto truth = *y_true_begin, pred = *y_pred_begin;
    gradient += 2 * (pred - truth);
    ++y_true_begin, ++y_pred_begin;
  }
  return gradient / y_true.total();
}

// /////////////////////////////////////////////

auto BinaryCrossEntropy::type() const -> Loss { return Loss::BinaryCrossEntropy; }

auto BinaryCrossEntropy::to_string() const -> std::string { return "Binary Cross Entropy"; }

auto BinaryCrossEntropy::type_name() const -> std::string { return "BinaryCrossEntropy"; }

auto BinaryCrossEntropy::operator()(const tensor_type &y_true, const tensor_type &y_pred) const -> value_type {
  // This function implements the subsequent operation.
  //
  // Formula: Ĺ = -1 / n  ⅀ [ὶ = 1, n] Yὶ . ln(Ýὶ) + (1 - Yὶ) . ln(1 - Ýὶ)
  //
  // where:
  //  Ĺ  - Loss function
  //  Y  - Observations
  //  Ý  - Predicted probabilities
  //  Yὶ - Label of the ὶᵗʰ class                  => Range = {0, 1}
  //  Ýὶ - Predicted probability of the ὶᵗʰ class  => Range = [0, 1; 𝟄 R]
  //  n  - Number of classes
  //
  // For a batch of samples, the mean loss is returned.

  _s_check_rank_range(y_true.rank(), tensor_type::SCALAR_RANK, tensor_type::MATRIX_RANK);
  _s_check_shape_equality(y_true.shape(), y_pred.shape());

  const auto EPSILON = std::numeric_limits<value_type>::epsilon();

  auto y_true_begin = y_true.begin(), y_true_end = y_true.end();
  auto y_pred_begin = y_pred.begin();

  value_type total_logarithmic_loss = {};
  while (y_true_begin != y_true_end) {
    auto truth = *y_true_begin, pred = std::clamp(*y_pred_begin, EPSILON, 1 - EPSILON);
    total_logarithmic_loss -= truth * std::log(pred) + (1 - truth) * std::log(1 - pred);
    ++y_true_begin, ++y_pred_begin;
  }
  return total_logarithmic_loss / y_true.total();
}

auto BinaryCrossEntropy::derivative(const tensor_type &y_true, const tensor_type &y_pred) const -> value_type {
  // This function implements the subsequent operation.
  //
  // Derivative: ẟ / ẟÝὶ Ĺ = -1 / n ⅀ [ὶ = 1, n] Yὶ / Ýὶ + (1 - Yὶ) / (1 - Ýὶ)
  //
  // where:
  //  Ĺ  - Loss function
  //  Y  - Observations
  //  Ý  - Predicted probabilities
  //  Yὶ - Label of the ὶᵗʰ class                  => Range = {0, 1}
  //  Ýὶ - Predicted probability of the ὶᵗʰ class  => Range = [0, 1; 𝟄 R]
  //  n  - Number of classes
  //
  // For a batch of samples, the mean loss is returned.

  _s_check_rank_range(y_true.rank(), tensor_type::SCALAR_RANK, tensor_type::MATRIX_RANK);
  _s_check_shape_equality(y_true.shape(), y_pred.shape());

  const auto EPSILON = std::numeric_limits<value_type>::epsilon();

  auto y_true_begin = y_true.begin(), y_true_end = y_true.end();
  auto y_pred_begin = y_pred.begin();

  value_type gradient = {};
  while (y_true_begin != y_true_end) {
    auto truth = *y_true_begin, pred = std::clamp(*y_pred_begin, EPSILON, 1 - EPSILON);
    gradient -= truth / pred - (1 - truth) / (1 - pred);
    ++y_true_begin, ++y_pred_begin;
  }
  return gradient / y_true.total();
}

// /////////////////////////////////////////////

auto CategoricalCrossEntropy::type() const -> Loss { return Loss::CategoricalCrossEntropy; }

auto CategoricalCrossEntropy::to_string() const -> std::string { return "Categorical Cross Entropy"; }

auto CategoricalCrossEntropy::type_name() const -> std::string { return "CategoricalCrossEntropy"; }

auto CategoricalCrossEntropy::operator()(const tensor_type &y_true, const tensor_type &y_pred) const
    -> value_type {
  // This function implements the subsequent operation.
  //
  // Formula: Ĺ = -ln(Ý०)
  //
  // where:
  //  Ĺ  - Loss function
  //  Y  - Observations                                 => Criteria = {A one-hot vector}
  //  Ý  - Predicted probabilities                      => Criteria = {Accumulates to 1.0}
  //  Y० - Positive class                               => Range = {1}
  //  Ý० - Predicted probability of the positive class  => Range = [0, 1; 𝟄 R]
  //  n  - Number of classes
  //
  // For a batch of samples, the mean loss is returned.

  _s_check_rank_range(y_true.rank(), tensor_type::VECTOR_RANK, tensor_type::MATRIX_RANK);
  _s_check_shape_equality(y_true.shape(), y_pred.shape());

  const auto EPSILON = std::numeric_limits<value_type>::epsilon();
  const auto EYE = 1.0F;

  auto [samples] = y_true.is_matrix() ? y_true.shape().unwrap<1>() : Shape::SCALAR_SIZE;

  value_type total_logarithmic_loss = {};
  for (usize i = {}; i < y_true.total(); ++i) {
    auto truth = y_true[i], pred = std::clamp(y_pred[i], EPSILON, 1 - EPSILON);
    if (truth == EYE) {
      total_logarithmic_loss -= std::log(pred);
    }
  }

  return total_logarithmic_loss / samples;
}

auto CategoricalCrossEntropy::derivative(const tensor_type &y_true, const tensor_type &y_pred) const
    -> value_type {
  // This function implements the subsequent operation.
  //
  // Derivative: ẟ / ẟÝ० Ĺ = -1 / Ý०
  //
  // where:
  //  Ĺ  - Loss function
  //  Y  - Observations                                 => Criteria = {A one-hot vector}
  //  Ý  - Predicted probabilities                      => Criteria = {Accumulates to 1.0}
  //  Y० - Positive class                               => Range = {1}
  //  Ý० - Predicted probability of the positive class  => Range = [0, 1; 𝟄 R]
  //  n  - Number of classes
  //
  // For a batch of samples, the mean loss is returned.

  _s_check_rank_range(y_true.rank(), tensor_type::VECTOR_RANK, tensor_type::MATRIX_RANK);
  _s_check_shape_equality(y_pred.shape(), y_true.shape());

  const auto EPSILON = std::numeric_limits<value_type>::epsilon();
  const auto EYE = 1.0F;

  auto samples = y_true.is_matrix() ? y_true.shape().at(0) : Shape::SCALAR_SIZE;

  value_type gradient = {};
  for (usize i = {}; i < y_true.total(); ++i) {
    auto truth = y_true[i], pred = std::clamp(y_pred[i], EPSILON, 1 - EPSILON);
    if (truth == EYE) {
      gradient -= 1 / pred;
    }
  }

  return gradient / samples;
}

// /////////////////////////////////////////////

auto SparseCrossEntropy::type() const -> Loss { return Loss::SparseCrossEntropy; }

auto SparseCrossEntropy::to_string() const -> std::string { return "Sparse Cross Entropy"; }

auto SparseCrossEntropy::type_name() const -> std::string { return "SparseCrossEntropy"; }

auto SparseCrossEntropy::operator()(const tensor_type &y_true, const tensor_type &y_pred) const -> value_type {
  // This function implements the subsequent operation.
  //
  // Formula: Ĺ = -ln(Ý०)
  //
  // where:
  //  Ĺ  - Loss function
  //  Y  - Observations                                          => Criteria = {
  //                                                                  Sparse,
  //                                                                  Y.rank is one lower than Ý.rank
  //                                                                }
  //  Ý  - Predicted probabilities                               => Criteria = {Accumulates to 1.0}
  //  Y० - Index of the positive class                           => Range = [0, n; 𝟄 Z)
  //  Ý० - Predicted probability of the positive class (=Ý[Y०])  => Range = [0, 1; 𝟄 R]
  //  n  - Number of classes
  //
  // For a batch of samples, the mean loss is returned.

  _s_check_rank_range(y_pred.rank(), tensor_type::VECTOR_RANK, tensor_type::MATRIX_RANK);
  _s_check_shape_equality(y_true.shape(), y_pred.shape().slice(0, y_pred.rank() - 1));

  const auto EPSILON = std::numeric_limits<value_type>::epsilon();

  auto samples = y_pred.is_matrix() ? y_pred.shape().at(1) : Shape::SCALAR_SIZE;
  value_type total_logarithmic_loss = {};
  for (usize i = {}; i < y_true.total(); ++i) {
    auto truth = y_true[i];
    usize j = i * samples + truth;
    auto pred = std::clamp(y_pred[j], EPSILON, 1 - EPSILON);
    total_logarithmic_loss -= std::log(pred);
  }
  return total_logarithmic_loss / y_true.total();
}

auto SparseCrossEntropy::derivative(const tensor_type &y_true, const tensor_type &y_pred) const -> value_type {
  // This function implements the subsequent operation.
  //
  // Derivative: ẟ / ẟÝ० Ĺ = -1 / Ý०
  //
  // where:
  //  Ĺ  - Loss function
  //  Y  - Observations                                          => Criteria = {
  //                                                                  Sparse,
  //                                                                  Y.rank is one lower than Ý.rank
  //                                                                }
  //  Ý  - Predicted probabilities                               => Criteria = {Accumulates to 1.0}
  //  ὶ० - Index of the positive class                           => Range = [0, n; 𝟄 Z)
  //  Ý० - Predicted probability of the positive class (=Ý[ὶ०])  => Range = [0, 1; 𝟄 R]
  //  n  - Number of classes
  //
  // For a batch of samples, the mean loss is returned.

  _s_check_rank_range(y_pred.rank(), tensor_type::VECTOR_RANK, tensor_type::MATRIX_RANK);
  _s_check_shape_equality(y_true.shape(), y_pred.shape().slice(0, y_pred.rank() - 1));

  const auto EPSILON = std::numeric_limits<value_type>::epsilon();

  auto samples = y_pred.is_matrix() ? y_pred.shape().at(1) : Shape::SCALAR_SIZE;
  value_type gradient = {};
  for (usize i = {}; i < y_true.total(); ++i) {
    auto truth = y_true[i];
    usize j = i * samples + truth;
    auto pred = std::clamp(y_pred[j], EPSILON, 1 - EPSILON);
    gradient -= 1 / pred;
  }
  return gradient / y_true.total();
}

// /////////////////////////////////////////////////////////////
// LossFuncWrapper
// /////////////////////////////////////////////////////////////

// /////////////////////////////////////////////
// Constructors (and Destructors)
// /////////////////////////////////////////////

LossFuncWrapper::LossFuncWrapper(Loss loss) {
  switch (loss) {
    case Loss::MeanSquaredError: {
      func_ = std::make_shared<MeanSquaredError>();
      break;
    }
    case Loss::BinaryCrossEntropy: {
      func_ = std::make_shared<BinaryCrossEntropy>();
      break;
    }
    case Loss::CategoricalCrossEntropy: {
      func_ = std::make_shared<CategoricalCrossEntropy>();
      break;
    }
    case Loss::SparseCrossEntropy: {
      func_ = std::make_shared<SparseCrossEntropy>();
      break;
    }
  }
}

LossFuncWrapper::LossFuncWrapper(LossFuncWrapper &&other) noexcept : func_{std::move(other.func_)} {}

// /////////////////////////////////////////////
// Assignment Operators
// /////////////////////////////////////////////

auto LossFuncWrapper::operator=(LossFuncWrapper &&other) noexcept -> LossFuncWrapper & {
  func_ = std::move(other.func_);
  return *this;
}

// /////////////////////////////////////////////
// Query Functions
// /////////////////////////////////////////////

auto LossFuncWrapper::is_null() const -> bool { return func_ == nullptr; }

// /////////////////////////////////////////////
// Wrapper Interface
// /////////////////////////////////////////////

auto LossFuncWrapper::type() const -> Loss { return func_->type(); }

auto LossFuncWrapper::to_string() const -> std::string { return func_->to_string(); }

auto LossFuncWrapper::type_name() const -> std::string { return func_->type_name(); }

auto LossFuncWrapper::operator()(const tensor_type &y_true, const tensor_type &y_pred) const -> value_type {
  return func_->operator()(y_true, y_pred);
}

auto LossFuncWrapper::derivative(const tensor_type &y_true, const tensor_type &y_pred) const -> value_type {
  return func_->derivative(y_true, y_pred);
}

}
