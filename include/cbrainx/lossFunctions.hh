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

#ifndef CBRAINX__LOSS_FUNCTIONS_HH_
#define CBRAINX__LOSS_FUNCTIONS_HH_

#include <memory>
#include <string>

#include "shape.hh"
#include "tensor.hh"
#include "typeAliases.hh"

namespace cbx {

/// \brief Supported loss functions.
///
/// \see LossFunction
enum class Loss { MeanSquaredError, BinaryCrossEntropy, CategoricalCrossEntropy, SparseCrossEntropy };

/// \brief The `LossFunction` class defines a standard interface for all loss functions.
///
/// \details
/// A loss function is a mathematical function that assesses how well a model performs with the current
/// parameters. Sometimes, the loss function is used interchangeably with the cost function.
///
/// \see Loss
struct LossFunction {
 protected:
  // /////////////////////////////////////////////
  // Helpers
  // /////////////////////////////////////////////

  /// \brief Checks if the given rank (inclusively) falls within the specified range.
  /// \param[in] rank The rank to be checked.
  /// \param[in] lower_bound, upper_bound The range [\p lower_bound, \p upper_bound]
  ///
  /// \details
  /// This function throws an exception if \p rank is out of range.
  ///
  /// \throws RankError
  static auto _s_check_rank_range(usize rank, usize lower_bound, usize upper_bound) -> void;

  /// \brief Checks if two shapes are equal.
  /// \param[in] a, b The shapes to be compared for equality.
  ///
  /// \details
  /// This function throws an exception if \p a and \p b are unequal.
  ///
  /// \throws ShapeError
  static auto _s_check_shape_equality(const Shape &a, const Shape &b) -> void;

 public:
  using value_type = f32;

  using tensor_type = Tensor<value_type>;

  // /////////////////////////////////////////////
  // Interface
  // /////////////////////////////////////////////

  /// \brief Returns the type of the loss function.
  /// \return The type of the loss function.
  [[nodiscard]] virtual auto type() const -> Loss = 0;

  /// \brief Returns the pretty name of the function as a string.
  /// \return The pretty name of the function.
  [[nodiscard]] virtual auto to_string() const -> std::string = 0;

  /// \brief Returns the type name of the function as a string.
  /// \return The type name of the function.
  [[nodiscard]] virtual auto type_name() const -> std::string = 0;

  /// \brief The function call operator.
  /// \param[in] y_true The observed values.
  /// \param[in] y_pred The predicted values.
  /// \return The mean loss.
  [[nodiscard]] virtual auto operator()(const tensor_type &y_true, const tensor_type &y_pred) const
      -> value_type = 0;

  /// \brief Returns the derivative of the function w.r.t. \p y_pred.
  /// \param[in] y_true The observed values.
  /// \param[in] y_pred The predicted values.
  /// \return The derivative of the function w.r.t. \p y_pred.
  [[nodiscard]] virtual auto derivative(const tensor_type &y_true, const tensor_type &y_pred) const
      -> value_type = 0;
};

/// \brief The `MeanSquaredError` loss function.
///
/// \details
/// This functor performs the subsequent operation.
///
/// Formula: Ĺ = 1 / n  ⅀ [ὶ = 1, n] (Ýὶ - Yὶ)²
///
/// And,
///
/// Derivative: ẟĹ / ẟÝὶ = 1 / n ⅀ [ὶ = 1, n] 2 . (Ýὶ - Yὶ)
///
/// where:
///  Ĺ  - Loss function
///  Y  - Observations
///  Ý  - Predictions
///  Yὶ - ὶᵗʰ observed value   => Range = (-∞, ∞; 𝟄 R)
///  Ýὶ - ὶᵗʰ predicted value  => Range = (-∞, ∞; 𝟄 R)
///  n  - Number of observations
///
/// For a batch of samples, the mean loss is returned.
///
/// \note If the input is invalid, unexpected results may occur.
///
/// \see Loss LossFunction
struct MeanSquaredError : public LossFunction {
  /// \brief Returns the type of the loss function.
  /// \return The type of the loss function.
  [[nodiscard]] auto type() const -> Loss override;

  /// \brief Returns the pretty name of the function as a string.
  /// \return The pretty name of the function.
  [[nodiscard]] auto to_string() const -> std::string override;

  /// \brief Returns the type name of the function as a string.
  /// \return The type name of the function.
  [[nodiscard]] auto type_name() const -> std::string override;

  /// \brief The function call operator.
  /// \param[in] y_true The observed values.
  /// \param[in] y_pred The predicted values.
  /// \return The mean loss.
  ///
  /// \throws RankError
  /// \throws ShapeError
  [[nodiscard]] auto operator()(const tensor_type &y_true, const tensor_type &y_pred) const
      -> value_type override;

  /// \brief Returns the derivative of the function w.r.t. \p y_pred.
  /// \param[in] y_true The observed values.
  /// \param[in] y_pred The predicted values.
  /// \return The derivative of the function w.r.t. \p y_pred.
  ///
  /// \throws RankError
  /// \throws ShapeError
  [[nodiscard]] auto derivative(const tensor_type &y_true, const tensor_type &y_pred) const
      -> value_type override;
};

/// \brief The `BinaryCrossEntropy` loss function.
///
/// \details
/// This functor performs the subsequent operation.
///
/// Formula: Ĺ = -1 / n  ⅀ [ὶ = 1, n] Yὶ . ln(Ýὶ) + (1 - Yὶ) . ln(1 - Ýὶ)
///
/// And,
///
/// Derivative: ẟĹ / ẟÝὶ = -1 / n ⅀ [ὶ = 1, n] Yὶ / Ýὶ + (1 - Yὶ) / (1 - Ýὶ)
///
/// where:
///  Ĺ  - Loss function
///  Y  - Observations
///  Ý  - Predicted probabilities
///  Yὶ - Label of the ὶᵗʰ class                  => Range = {0, 1}
///  Ýὶ - Predicted probability of the ὶᵗʰ class  => Range = [0, 1; 𝟄 R]
///  n  - Number of classes
///
/// For a batch of samples, the mean loss is returned.
///
/// \note If the input is invalid, unexpected results may occur.
///
/// \see Loss LossFunction
struct BinaryCrossEntropy : public LossFunction {
  /// \brief Returns the type of the loss function.
  /// \return The type of the loss function.
  [[nodiscard]] auto type() const -> Loss override;

  /// \brief Returns the pretty name of the function as a string.
  /// \return The pretty name of the function.
  [[nodiscard]] auto to_string() const -> std::string override;

  /// \brief Returns the type name of the function as a string.
  /// \return The type name of the function.
  [[nodiscard]] auto type_name() const -> std::string override;

  /// \brief The function call operator.
  /// \param[in] y_true The observed values.
  /// \param[in] y_pred The predicted values.
  /// \return The mean loss.
  ///
  /// \throws RankError
  /// \throws ShapeError
  [[nodiscard]] auto operator()(const tensor_type &y_true, const tensor_type &y_pred) const
      -> value_type override;

  /// \brief Returns the derivative of the function w.r.t. \p y_pred.
  /// \param[in] y_true The observed values.
  /// \param[in] y_pred The predicted values.
  /// \return The derivative of the function w.r.t. \p y_pred.
  ///
  /// \throws RankError
  /// \throws ShapeError
  [[nodiscard]] auto derivative(const tensor_type &y_true, const tensor_type &y_pred) const
      -> value_type override;
};

/// \brief The `CategoricalCrossEntropy` loss function.
///
/// \details
/// This functor performs the subsequent operation.
///
/// Formula: Ĺ = -ln(Ý०)
///
/// And,
///
/// Derivative: ẟĹ / ẟÝ० = -1 / Ý०
///
/// where:
///  Ĺ  - Loss function
///  Y  - Observations                                 => Criteria = {A one-hot vector}
///  Ý  - Predicted probabilities                      => Criteria = {Accumulates to 1.0}
///  Y० - Positive class                               => Range = {1}
///  Ý० - Predicted probability of the positive class  => Range = [0, 1; 𝟄 R]
///  n  - Number of classes
///
/// For a batch of samples, the mean loss is returned.
///
/// \note If the input is invalid, unexpected results may occur.
///
/// \see Loss LossFunction
struct CategoricalCrossEntropy : public LossFunction {
  /// \brief Returns the type of the loss function.
  /// \return The type of the loss function.
  [[nodiscard]] auto type() const -> Loss override;

  /// \brief Returns the pretty name of the function as a string.
  /// \return The pretty name of the function.
  [[nodiscard]] auto to_string() const -> std::string override;

  /// \brief Returns the type name of the function as a string.
  /// \return The type name of the function.
  [[nodiscard]] auto type_name() const -> std::string override;

  /// \brief The function call operator.
  /// \param[in] y_true The observed values.
  /// \param[in] y_pred The predicted values.
  /// \return The mean loss.
  ///
  /// \throws RankError
  /// \throws ShapeError
  [[nodiscard]] auto operator()(const tensor_type &y_true, const tensor_type &y_pred) const
      -> value_type override;

  /// \brief Returns the derivative of the function w.r.t. \p y_pred.
  /// \param[in] y_true The observed values.
  /// \param[in] y_pred The predicted values.
  /// \return The derivative of the function w.r.t. \p y_pred.
  ///
  /// \throws RankError
  /// \throws ShapeError
  [[nodiscard]] auto derivative(const tensor_type &y_true, const tensor_type &y_pred) const
      -> value_type override;
};

/// \brief The `SparseCrossEntropy` loss function.
///
/// \details
/// This functor performs the subsequent operation.
///
/// Formula: Ĺ = -ln(Ý०)
///
/// And,
///
/// Derivative: ẟĹ / ẟÝ० = -1 / Ý०
///
/// where:
///  Ĺ  - Loss function
///  Y  - Observations                                          => Criteria = {
///                                                                  Sparse,
///                                                                  Y.rank is one lower than Ý.rank
///                                                                }
///  Ý  - Predicted probabilities                               => Criteria = {Accumulates to 1.0}
///  ὶ० - Index of the positive class                           => Range = [0, n; 𝟄 Z)
///  Ý० - Predicted probability of the positive class (=Ý[ὶ०])  => Range = [0, 1; 𝟄 R]
///  n  - Number of classes
///
/// For a batch of samples, the mean loss is returned.
///
/// \note If the input is invalid, unexpected results may occur.
///
/// \see Loss LossFunction
struct SparseCrossEntropy : public LossFunction {
  /// \brief Returns the type of the loss function.
  /// \return The type of the loss function.
  [[nodiscard]] auto type() const -> Loss override;

  /// \brief Returns the pretty name of the function as a string.
  /// \return The pretty name of the function.
  [[nodiscard]] auto to_string() const -> std::string override;

  /// \brief Returns the type name of the function as a string.
  /// \return The type name of the function.
  [[nodiscard]] auto type_name() const -> std::string override;

  /// \brief The function call operator.
  /// \param[in] y_true The observed values.
  /// \param[in] y_pred The predicted values.
  /// \return The mean loss.
  ///
  /// \throws RankError
  /// \throws ShapeError
  [[nodiscard]] auto operator()(const tensor_type &y_true, const tensor_type &y_pred) const
      -> value_type override;

  /// \brief Returns the derivative of the function w.r.t. \p y_pred.
  /// \param[in] y_true The observed values.
  /// \param[in] y_pred The predicted values.
  /// \return The derivative of the function w.r.t. \p y_pred.
  ///
  /// \throws RankError
  /// \throws ShapeError
  [[nodiscard]] auto derivative(const tensor_type &y_true, const tensor_type &y_pred) const
      -> value_type override;
};

}

#endif
