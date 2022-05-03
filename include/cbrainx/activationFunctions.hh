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

#ifndef CBRAINX__ACTIVATION_FUNCTIONS_HH_
#define CBRAINX__ACTIVATION_FUNCTIONS_HH_

#include <functional>
#include <memory>
#include <string>

#include "typeAliases.hh"

namespace cbx {

/// \brief Supported activation functions.
enum class Activation {
  ArcTan,
  BinaryStep,
  ELU,
  Gaussian,
  GELU,
  LeakyReLU,
  Linear,
  ReLU,
  Sigmoid,
  Softplus,
  Swish,
  TanH
};

/// \brief The `ActivationFunction` class defines a standard interface for all activation functions.
///
/// \see Activation
struct ActivationFunction {
  using value_type = f32;

  // /////////////////////////////////////////////
  // Interface
  // /////////////////////////////////////////////

  /// \brief Returns the type of the activation function.
  /// \return The type of the activation function.
  [[nodiscard]] virtual auto type() const -> Activation = 0;

  /// \brief Returns the pretty name of the function as a string.
  /// \return The pretty name of the function.
  [[nodiscard]] virtual auto to_string() const -> std::string = 0;

  /// \brief Returns the type name of the function as a string.
  /// \return The type name of the function.
  [[nodiscard]] virtual auto type_name() const -> std::string = 0;

  /// \brief The function call operator.
  /// \param[in] x X coordinate.
  /// \return The output of the function at \p x.
  [[nodiscard]] virtual auto operator()(value_type x) const -> value_type = 0;

  /// \brief Returns the derivative of the function at \p x.
  /// \param[in] x X coordinate.
  /// \return The derivative of the function at \p x.
  [[nodiscard]] virtual auto derivative(value_type x) const -> value_type = 0;
};

/// \brief The `ActFuncWrapper` class wraps an activation function and allows you to switch between different
/// types at runtime.
///
/// \see Activation ActivationFunction
class ActFuncWrapper {
 public:
  using value_type = ActivationFunction::value_type;

 private:
  /// \brief Shared pointer to the activation function.
  std::shared_ptr<ActivationFunction> func_ = {};

 public:
  // /////////////////////////////////////////////
  // Constructors and Destructors
  // /////////////////////////////////////////////

  /// \brief Default constructor.
  ActFuncWrapper() = default;

  /// \brief Parameterized constructor.
  /// \param[in] activation
  explicit ActFuncWrapper(Activation activation);

  /// \brief Default copy constructor.
  /// \param[in] other Source wrapper.
  ActFuncWrapper(const ActFuncWrapper &other) = default;

  /// \brief Move constructor.
  /// \param[in] other Source wrapper.
  ActFuncWrapper(ActFuncWrapper &&other) noexcept;

  /// \brief Default destructor.
  ~ActFuncWrapper() = default;

  // /////////////////////////////////////////////
  // Assignment Operators
  // /////////////////////////////////////////////

  /// \brief Default copy assignment operator.
  /// \param[in] other Source wrapper.
  /// \return A reference to self.
  auto operator=(const ActFuncWrapper &other) -> ActFuncWrapper & = default;

  /// \brief Move assignment operator.
  /// \param[in] other Source wrapper.
  /// \return A reference to self.
  auto operator=(ActFuncWrapper &&other) noexcept -> ActFuncWrapper &;

  // /////////////////////////////////////////////
  // Wrapper Interface
  // /////////////////////////////////////////////

  /// \brief Returns the type of the activation function.
  /// \return The type of the activation function.
  [[nodiscard]] auto type() const -> Activation;

  /// \brief Returns the pretty name of the function as a string.
  /// \return The pretty name of the function.
  [[nodiscard]] auto to_string() const -> std::string;

  /// \brief Returns the type name of the function as a string.
  /// \return The type name of the function.
  [[nodiscard]] auto type_name() const -> std::string;

  /// \brief The function call operator.
  /// \param[in] x X coordinate.
  /// \return The output of the function at \p x.
  [[nodiscard]] auto operator()(value_type x) const -> value_type;

  /// \brief Returns the derivative of the function.
  /// \return Derivative function of the function.
  [[nodiscard]] auto derivative() const -> std::function<value_type(value_type)>;
};

/// \brief `ArcTan` activation function.
struct ArcTan : public ActivationFunction {
  /// \brief Returns the type of the activation function.
  /// \return The type of the activation function.
  [[nodiscard]] auto type() const -> Activation override;

  /// \brief Returns the pretty name of the function as a string.
  /// \return The pretty name of the function.
  [[nodiscard]] auto to_string() const -> std::string override;

  /// \brief Returns the type name of the function as a string.
  /// \return The type name of the function.
  [[nodiscard]] auto type_name() const -> std::string override;

  /// \brief The function call operator.
  /// \param[in] x X coordinate.
  /// \return The output of the function at \p x.
  [[nodiscard]] auto operator()(value_type x) const -> value_type override;

  /// \brief Returns the derivative of the function at \p x.
  /// \param[in] x X coordinate.
  /// \return The derivative of the function at \p x.
  [[nodiscard]] auto derivative(value_type x) const -> value_type override;
};

/// \brief `BinaryStep` activation function.
struct BinaryStep : public ActivationFunction {
  /// \brief Returns the type of the activation function.
  /// \return The type of the activation function.
  [[nodiscard]] auto type() const -> Activation override;

  /// \brief Returns the pretty name of the function as a string.
  /// \return The pretty name of the function.
  [[nodiscard]] auto to_string() const -> std::string override;

  /// \brief Returns the type name of the function as a string.
  /// \return The type name of the function.
  [[nodiscard]] auto type_name() const -> std::string override;

  /// \brief The function call operator.
  /// \param[in] x X coordinate.
  /// \return The output of the function at \p x.
  [[nodiscard]] auto operator()(value_type x) const -> value_type override;

  /// \brief Returns the derivative of the function at \p x.
  /// \param[in] x X coordinate.
  /// \return The derivative of the function at \p x.
  [[nodiscard]] auto derivative(value_type x) const -> value_type override;
};

/// \brief `ELU` activation function.
struct ELU : public ActivationFunction {
  /// \brief Alpha constant.
  static constexpr const value_type ALPHA = 1.0F;

  /// \brief Returns the type of the activation function.
  /// \return The type of the activation function.

  [[nodiscard]] auto type() const -> Activation override;

  /// \brief Returns the pretty name of the function as a string.
  /// \return The pretty name of the function.
  [[nodiscard]] auto to_string() const -> std::string override;

  /// \brief Returns the type name of the function as a string.
  /// \return The type name of the function.
  [[nodiscard]] auto type_name() const -> std::string override;

  /// \brief The function call operator.
  /// \param[in] x X coordinate.
  /// \return The output of the function at \p x.
  [[nodiscard]] auto operator()(value_type x) const -> value_type override;

  /// \brief Returns the derivative of the function at \p x.
  /// \param[in] x X coordinate.
  /// \return The derivative of the function at \p x.
  [[nodiscard]] auto derivative(value_type x) const -> value_type override;
};

/// \brief `Gaussian` activation function.
struct Gaussian : public ActivationFunction {
  /// \brief Returns the type of the activation function.
  /// \return The type of the activation function.
  [[nodiscard]] auto type() const -> Activation override;

  /// \brief Returns the pretty name of the function as a string.
  /// \return The pretty name of the function.
  [[nodiscard]] auto to_string() const -> std::string override;

  /// \brief Returns the type name of the function as a string.
  /// \return The type name of the function.
  [[nodiscard]] auto type_name() const -> std::string override;

  /// \brief The function call operator.
  /// \param[in] x X coordinate.
  /// \return The output of the function at \p x.
  [[nodiscard]] auto operator()(value_type x) const -> value_type override;

  /// \brief Returns the derivative of the function at \p x.
  /// \param[in] x X coordinate.
  /// \return The derivative of the function at \p x.
  [[nodiscard]] auto derivative(value_type x) const -> value_type override;
};

/// \brief `GELU` activation function.
///
/// \note This class implements a close approximation of the GELU function.
struct GELU : public ActivationFunction {
  /// \brief Approximation constant.
  static constexpr const value_type C = 1.872F;

  /// \brief Returns the type of the activation function.
  /// \return The type of the activation function.
  [[nodiscard]] auto type() const -> Activation override;

  /// \brief Returns the pretty name of the function as a string.
  /// \return The pretty name of the function.
  [[nodiscard]] auto to_string() const -> std::string override;

  /// \brief Returns the type name of the function as a string.
  /// \return The type name of the function.
  [[nodiscard]] auto type_name() const -> std::string override;

  /// \brief The function call operator.
  /// \param[in] x X coordinate.
  /// \return The output of the function at \p x.
  [[nodiscard]] auto operator()(value_type x) const -> value_type override;

  /// \brief Returns the derivative of the function at \p x.
  /// \param[in] x X coordinate.
  /// \return The derivative of the function at \p x.
  [[nodiscard]] auto derivative(value_type x) const -> value_type override;
};

/// \brief `LeakyReLU` activation function.
struct LeakyReLU : public ActivationFunction {
  /// \brief Slope constant.
  static constexpr const value_type M = 0.1F;

  /// \brief Returns the type of the activation function.
  /// \return The type of the activation function.
  [[nodiscard]] auto type() const -> Activation override;

  /// \brief Returns the pretty name of the function as a string.
  /// \return The pretty name of the function.
  [[nodiscard]] auto to_string() const -> std::string override;

  /// \brief Returns the type name of the function as a string.
  /// \return The type name of the function.
  [[nodiscard]] auto type_name() const -> std::string override;

  /// \brief The function call operator.
  /// \param[in] x X coordinate.
  /// \return The output of the function at \p x.
  [[nodiscard]] auto operator()(value_type x) const -> value_type override;

  /// \brief Returns the derivative of the function at \p x.
  /// \param[in] x X coordinate.
  /// \return The derivative of the function at \p x.
  [[nodiscard]] auto derivative(value_type x) const -> value_type override;
};

/// \brief `Linear` activation function.
struct Linear : public ActivationFunction {
  /// \brief Returns the type of the activation function.
  /// \return The type of the activation function.
  [[nodiscard]] auto type() const -> Activation override;

  /// \brief Returns the pretty name of the function as a string.
  /// \return The pretty name of the function.
  [[nodiscard]] auto to_string() const -> std::string override;

  /// \brief Returns the type name of the function as a string.
  /// \return The type name of the function.
  [[nodiscard]] auto type_name() const -> std::string override;

  /// \brief The function call operator.
  /// \param[in] x X coordinate.
  /// \return The output of the function at \p x.
  [[nodiscard]] auto operator()(value_type x) const -> value_type override;

  /// \brief Returns the derivative of the function at \p x.
  /// \param[in] x X coordinate.
  /// \return The derivative of the function at \p x.
  [[nodiscard]] auto derivative(value_type x) const -> value_type override;
};

/// \brief `ReLU` activation function.
struct ReLU : public ActivationFunction {
  /// \brief Returns the type of the activation function.
  /// \return The type of the activation function.
  [[nodiscard]] auto type() const -> Activation override;

  /// \brief Returns the pretty name of the function as a string.
  /// \return The pretty name of the function.
  [[nodiscard]] auto to_string() const -> std::string override;

  /// \brief Returns the type name of the function as a string.
  /// \return The type name of the function.
  [[nodiscard]] auto type_name() const -> std::string override;

  /// \brief The function call operator.
  /// \param[in] x X coordinate.
  /// \return The output of the function at \p x.
  [[nodiscard]] auto operator()(value_type x) const -> value_type override;

  /// \brief Returns the derivative of the function at \p x.
  /// \param[in] x X coordinate.
  /// \return The derivative of the function at \p x.
  [[nodiscard]] auto derivative(value_type x) const -> value_type override;
};

/// \brief `Sigmoid` activation function.
struct Sigmoid : public ActivationFunction {
  /// \brief Returns the type of the activation function.
  /// \return The type of the activation function.
  [[nodiscard]] auto type() const -> Activation override;

  /// \brief Returns the pretty name of the function as a string.
  /// \return The pretty name of the function.
  [[nodiscard]] auto to_string() const -> std::string override;

  /// \brief Returns the type name of the function as a string.
  /// \return The type name of the function.
  [[nodiscard]] auto type_name() const -> std::string override;

  /// \brief The function call operator.
  /// \param[in] x X coordinate.
  /// \return The output of the function at \p x.
  [[nodiscard]] auto operator()(value_type x) const -> value_type override;

  /// \brief Returns the derivative of the function at \p x.
  /// \param[in] x X coordinate.
  /// \return The derivative of the function at \p x.
  [[nodiscard]] auto derivative(value_type x) const -> value_type override;
};

/// \brief `Softplus` activation function.
struct Softplus : public ActivationFunction {
  /// \brief Returns the type of the activation function.
  /// \return The type of the activation function.
  [[nodiscard]] auto type() const -> Activation override;

  /// \brief Returns the pretty name of the function as a string.
  /// \return The pretty name of the function.
  [[nodiscard]] auto to_string() const -> std::string override;

  /// \brief Returns the type name of the function as a string.
  /// \return The type name of the function.
  [[nodiscard]] auto type_name() const -> std::string override;

  /// \brief The function call operator.
  /// \param[in] x X coordinate.
  /// \return The output of the function at \p x.
  [[nodiscard]] auto operator()(value_type x) const -> value_type override;

  /// \brief Returns the derivative of the function at \p x.
  /// \param[in] x X coordinate.
  /// \return The derivative of the function at \p x.
  [[nodiscard]] auto derivative(value_type x) const -> value_type override;
};

/// \brief `Swish` activation function.
struct Swish : public ActivationFunction {
  /// \brief Returns the type of the activation function.
  /// \return The type of the activation function.
  [[nodiscard]] auto type() const -> Activation override;

  /// \brief Returns the pretty name of the function as a string.
  /// \return The pretty name of the function.
  [[nodiscard]] auto to_string() const -> std::string override;

  /// \brief Returns the type name of the function as a string.
  /// \return The type name of the function.
  [[nodiscard]] auto type_name() const -> std::string override;

  /// \brief The function call operator.
  /// \param[in] x X coordinate.
  /// \return The output of the function at \p x.
  [[nodiscard]] auto operator()(value_type x) const -> value_type override;

  /// \brief Returns the derivative of the function at \p x.
  /// \param[in] x X coordinate.
  /// \return The derivative of the function at \p x.
  [[nodiscard]] auto derivative(value_type x) const -> value_type override;
};

/// \brief `TanH` activation function.
struct TanH : public ActivationFunction {
  /// \brief Returns the type of the activation function.
  /// \return The type of the activation function.
  [[nodiscard]] auto type() const -> Activation override;

  /// \brief Returns the pretty name of the function as a string.
  /// \return The pretty name of the function.
  [[nodiscard]] auto to_string() const -> std::string override;

  /// \brief Returns the type name of the function as a string.
  /// \return The type name of the function.
  [[nodiscard]] auto type_name() const -> std::string override;

  /// \brief The function call operator.
  /// \param[in] x X coordinate.
  /// \return The output of the function at \p x.
  [[nodiscard]] auto operator()(value_type x) const -> value_type override;

  /// \brief Returns the derivative of the function at \p x.
  /// \param[in] x X coordinate.
  /// \return The derivative of the function at \p x.
  [[nodiscard]] auto derivative(value_type x) const -> value_type override;
};

}

#endif
