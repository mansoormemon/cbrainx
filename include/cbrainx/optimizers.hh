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

#ifndef CBRAINX__OPTIMIZERS_HH_
#define CBRAINX__OPTIMIZERS_HH_

#include <memory>

#include "tensor.hh"
#include "typeAliases.hh"

namespace cbx {

/// \brief Supported optimizers.
enum class Optimizer { GradientDescent };

/// \brief The `AbstractOptimizer` class defines a standard interface for all optimizers.
///
/// \details
/// The goal of optimizers is to diminish the loss by updating weights and biases, including other trainable
/// parameters of the model. This process is known as minimizing the cost function or training.
///
/// \see Optimizer
class AbstractOptimizer {
 public:
  using value_type = f32;

  using tensor_type = Tensor<value_type>;

 protected:
  /// \brief A counter to track how many times the parameters have been updated.
  ///
  /// \note One iteration corresponds to one complete backward pass.
  u32 iterations_ = {};

 public:
  // /////////////////////////////////////////////
  // Constructors and Destructors
  // /////////////////////////////////////////////

  /// \brief Default constructor.
  AbstractOptimizer() = default;

  /// \brief Default copy constructor.
  /// \param[in] other Source optimizer.
  AbstractOptimizer(const AbstractOptimizer &other) = default;

  /// \brief Default destructor.
  virtual ~AbstractOptimizer() = default;

  // /////////////////////////////////////////////
  // Assignment Operators
  // /////////////////////////////////////////////

  /// \brief Default copy assignment operator.
  /// \param[in] other Source optimizer.
  /// \return A reference to self.
  auto operator=(const AbstractOptimizer &other) -> AbstractOptimizer & = default;

  // /////////////////////////////////////////////
  // Query Functions
  // /////////////////////////////////////////////

  /// \brief Returns the number of iterations.
  /// \return The number of iterations.
  [[nodiscard]] auto iterations() const -> u32;

  /// \brief Returns the type of the optimizer.
  /// \return The type of the optimizer.
  ///
  /// \see Optimizer
  [[nodiscard]] virtual auto type() const -> Optimizer = 0;

  // /////////////////////////////////////////////
  // Informative
  // /////////////////////////////////////////////

  /// \brief Returns a string with information about the optimizer's properties.
  /// \return Information about the optimizer's properties as a string.
  [[nodiscard]] virtual auto property() const -> std::string = 0;

  /// \brief Returns the pretty name of the optimizer as a string.
  /// \return The pretty name of the optimizer.
  [[nodiscard]] virtual auto to_string() const -> std::string = 0;

  /// \brief Returns the type name of the optimizer as a string.
  /// \return The type name of the optimizer.
  [[nodiscard]] virtual auto type_name() const -> std::string = 0;

  /// \brief Returns meta-information about the optimizer as a string.
  /// \return A string containing meta-information about the optimizer.
  [[nodiscard]] virtual auto meta_info() const -> std::string;

  // /////////////////////////////////////////////
  // Interface
  // /////////////////////////////////////////////

  /// \brief Updates iteration count.
  /// \return A reference to self.
  virtual auto operator++() -> AbstractOptimizer &;

  /// \brief Resets the optimizer to its initial state.
  /// \return A reference to self.
  virtual auto reset() -> AbstractOptimizer &;

  /// \brief Updates the given set of parameters.
  /// \param[in] params The parameters to be updated.
  /// \param[in] gradient The gradient of \p params w.r.t. the loss function.
  virtual auto update_params(tensor_type &params, const tensor_type &gradient) -> void = 0;
};

/// \brief The `GradientDescent` class implements the gradient descent optimization algorithm with a decaying
/// learning rate.
///
/// \details
/// Gradient Descent is the most effective algorithm for training and optimizing a model which uses calculus to
/// find the local or global minimum of the cost function. It computes the gradient of the loss function for
/// each parameter and iteratively updates them until it converges to the values that incur the minimum cost.
/// Although the global minimum is the preferred outcome, a local minimum is also satisfactory.
///
/// The parameters are updated using the following rule.
///
/// Formula: Ŵ = Ŵ - ⍺ . ẟ / ẟŴ Ĺ
///
/// where:
///  Ŵ        - Trainable parameters
///  ⍺        - Learning rate
///  ẟ / ẟŴ Ĺ - Gradient of the loss function w.r.t. Ŵ
///
/// With each iteration, the learning rate decays using the following rule.
///
/// Rule: ⍺ = ȹ . 1 / (1 + Ɣ . ὶ)
///
/// where:
///  ⍺ - New learning rate
///  ȹ - Initial learning rate
///  Ɣ - Decay rate
///  ὶ - Iterations
///
/// \see Optimizer AbstractOptimizer
class GradientDescent : public AbstractOptimizer {
 private:
  /// \brief The initial learning rate.
  f32 learning_rate_ = {};

  /// \brief The current learning rate.
  f32 alpha_ = {};

  /// \brief The decay rate of the learning rate.
  f32 decay_rate_ = {};

 public:
  // /////////////////////////////////////////////
  // Constructors and Destructors
  // /////////////////////////////////////////////

  /// \brief Default constructor.
  GradientDescent() = default;

  /// \brief Default copy constructor.
  /// \param[in] other Source `GradientDescent` object.
  GradientDescent(const GradientDescent &other) = default;

  /// \brief Parameterized constructor.
  /// \param[in] learning_rate The initial learning rate.
  /// \param[in] decay_rate The decay rate of the learning rate.
  explicit GradientDescent(f32 learning_rate, f32 decay_rate = 0.0);

  /// \brief Default destructor.
  ~GradientDescent() override = default;

  // /////////////////////////////////////////////
  // Assignment Operators
  // /////////////////////////////////////////////

  /// \brief Default copy assignment operator.
  /// \param[in] other Source `GradientDescent` object.
  /// \return A reference to self.
  auto operator=(const GradientDescent &other) -> GradientDescent & = default;

  // /////////////////////////////////////////////
  // Query Functions
  // /////////////////////////////////////////////

  /// \brief Returns the type of the optimizer.
  /// \return The type of the optimizer.
  ///
  /// \see Optimizer
  [[nodiscard]] auto type() const -> Optimizer override;

  // /////////////////////////////////////////////
  // Informative
  // /////////////////////////////////////////////

  /// \brief Returns a string with information about the optimizer's properties.
  /// \return Information about the optimizer's properties as a string.
  [[nodiscard]] auto property() const -> std::string override;

  /// \brief Returns the pretty name of the optimizer as a string.
  /// \return The pretty name of the optimizer.
  [[nodiscard]] auto to_string() const -> std::string override;

  /// \brief Returns the type name of the optimizer as a string.
  /// \return The type name of the optimizer.
  [[nodiscard]] auto type_name() const -> std::string override;

  /// \brief Returns meta-information about the optimizer as a string.
  /// \return A string containing meta-information about the optimizer.
  [[nodiscard]] auto meta_info() const -> std::string override;

  // /////////////////////////////////////////////
  // Interface
  // /////////////////////////////////////////////

  /// \brief Updates iteration count.
  /// \return A reference to self.
  auto operator++() -> AbstractOptimizer & override;

  /// \brief Resets the optimizer to its initial state.
  /// \return A reference to self.
  auto reset() -> AbstractOptimizer & override;

  /// \brief Updates the given set of parameters.
  /// \param[in] params The parameters to be updated.
  /// \param[in] gradient The gradient of \p params w.r.t. the loss function.
  auto update_params(tensor_type &params, const tensor_type &gradient) -> void override;
};

/// \brief The `OptimizerWrapper` class wraps an optimizer and allows you to switch between different types at
/// runtime.
///
/// \see Optimizer AbstractOptimizer
class OptimizerWrapper {
 public:
  using value_type = AbstractOptimizer::value_type;

  using tensor_type = AbstractOptimizer::tensor_type;

 private:
  /// \brief Shared pointer to the optimizer.
  std::shared_ptr<AbstractOptimizer> optimizer_ = {};

 public:
  // /////////////////////////////////////////////
  // Constructors and Destructors
  // /////////////////////////////////////////////

  /// \brief Parameterized constructor.
  /// \tparam Args The data type of arguments.
  /// \param[in] optimizer The type of optimizer.
  /// \param[in] args Parameter list for the constructor of \p optimizer.
  template <typename... Args>
  explicit OptimizerWrapper(Optimizer optimizer, Args... args) {
    switch (optimizer) {
      case Optimizer::GradientDescent: {
        optimizer_ = std::make_shared<GradientDescent>(args...);
        break;
      }
    }
  }

  /// \brief Default copy constructor.
  /// \param[in] other Source wrapper.
  OptimizerWrapper(const OptimizerWrapper &other) = default;

  /// \brief Default destructor.
  virtual ~OptimizerWrapper() = default;

  // /////////////////////////////////////////////
  // Assignment Operators
  // /////////////////////////////////////////////

  /// \brief Default copy assignment operator.
  /// \param[in] other Source wrapper.
  /// \return A reference to self.
  auto operator=(const OptimizerWrapper &other) -> OptimizerWrapper & = default;

  // /////////////////////////////////////////////
  // Query Functions
  // /////////////////////////////////////////////

  /// \brief Returns the number of iterations.
  /// \return The number of iterations.
  [[nodiscard]] auto iterations() const -> u32;

  /// \brief Returns the type of the optimizer.
  /// \return The type of the optimizer.
  ///
  /// \see Optimizer
  [[nodiscard]] auto type() const -> Optimizer;

  /// \brief Returns whether or not the underlying shared pointer is null.
  /// \return True if the underlying shared pointer is null.
  [[nodiscard]] auto is_null() const -> bool;

  // /////////////////////////////////////////////
  // Informative
  // /////////////////////////////////////////////

  /// \brief Returns a string with information about the optimizer's properties.
  /// \return Information about the optimizer's properties as a string.
  [[nodiscard]] auto property() const -> std::string;

  /// \brief Returns the pretty name of the optimizer as a string.
  /// \return The pretty name of the optimizer.
  [[nodiscard]] auto to_string() const -> std::string;

  /// \brief Returns the type name of the optimizer as a string.
  /// \return The type name of the optimizer.
  [[nodiscard]] auto type_name() const -> std::string;

  /// \brief Returns meta-information about the optimizer as a string.
  /// \return A string containing meta-information about the optimizer.
  [[nodiscard]] auto meta_info() const -> std::string;

  // /////////////////////////////////////////////
  // Wrapper Interface
  // /////////////////////////////////////////////

  /// \brief Updates iteration count.
  /// \return A reference to self.
  auto operator++() -> OptimizerWrapper &;

  /// \brief Resets the optimizer to its initial state.
  /// \return A reference to self.
  auto reset() -> OptimizerWrapper &;

  /// \brief Updates the given set of parameters.
  /// \param[in] params The parameters to be updated.
  /// \param[in] gradient The gradient of \p params w.r.t. the loss function.
  auto update_params(tensor_type &params, const tensor_type &gradient) -> void;
};

}

#endif
