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

#ifndef CBRAINX__ACTIVATION_FUNCTIONS_HH_
#define CBRAINX__ACTIVATION_FUNCTIONS_HH_

#include <string>

#include "type_aliases.hh"

namespace cbx {

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
  SoftPlus,
  Swish,
  TanH
};

struct ActivationFunction {
  using value_type = f32;

  // /////////////////////////////////////////////////////////////

  [[nodiscard]] virtual auto operator()(value_type x) const -> value_type = 0;

  [[nodiscard]] virtual auto derivative(value_type) const -> value_type = 0;

  [[nodiscard]] virtual auto type() const -> Activation = 0;

  [[nodiscard]] virtual auto to_string() const -> std::string = 0;

  [[nodiscard]] virtual auto type_name() const -> std::string = 0;
};

// /////////////////////////////////////////////////////////////

struct ArcTan : public ActivationFunction {
  [[nodiscard]] auto operator()(value_type x) const -> value_type override;

  [[nodiscard]] auto derivative(value_type) const -> value_type override;

  [[nodiscard]] auto type() const -> Activation override;

  [[nodiscard]] auto to_string() const -> std::string override;

  [[nodiscard]] auto type_name() const -> std::string override;
};

// /////////////////////////////////////////////////////////////

struct BinaryStep : public ActivationFunction {
  [[nodiscard]] auto operator()(value_type x) const -> value_type override;

  [[nodiscard]] auto derivative(value_type) const -> value_type override;

  [[nodiscard]] auto type() const -> Activation override;

  [[nodiscard]] auto to_string() const -> std::string override;

  [[nodiscard]] auto type_name() const -> std::string override;
};

// /////////////////////////////////////////////////////////////

struct ELU : public ActivationFunction {
  static constexpr const value_type ALPHA = 1.0F;

  // /////////////////////////////////////////////////////////////

  [[nodiscard]] auto operator()(value_type x) const -> value_type override;

  [[nodiscard]] auto derivative(value_type) const -> value_type override;

  [[nodiscard]] auto type() const -> Activation override;

  [[nodiscard]] auto to_string() const -> std::string override;

  [[nodiscard]] auto type_name() const -> std::string override;
};

// /////////////////////////////////////////////////////////////

struct Gaussian : public ActivationFunction {
  [[nodiscard]] auto operator()(value_type x) const -> value_type override;

  [[nodiscard]] auto derivative(value_type) const -> value_type override;

  [[nodiscard]] auto type() const -> Activation override;

  [[nodiscard]] auto to_string() const -> std::string override;

  [[nodiscard]] auto type_name() const -> std::string override;
};

// /////////////////////////////////////////////////////////////

struct GELU : public ActivationFunction {
  static constexpr const value_type C = 1.872F;

  // /////////////////////////////////////////////////////////////

  [[nodiscard]] auto operator()(value_type x) const -> value_type override;

  [[nodiscard]] auto derivative(value_type) const -> value_type override;

  [[nodiscard]] auto type() const -> Activation override;

  [[nodiscard]] auto to_string() const -> std::string override;

  [[nodiscard]] auto type_name() const -> std::string override;
};
// /////////////////////////////////////////////////////////////

struct LeakyReLU : public ActivationFunction {
  static constexpr const value_type M = 0.1F;

  [[nodiscard]] auto operator()(value_type x) const -> value_type override;

  [[nodiscard]] auto derivative(value_type) const -> value_type override;

  [[nodiscard]] auto type() const -> Activation override;

  [[nodiscard]] auto to_string() const -> std::string override;

  [[nodiscard]] auto type_name() const -> std::string override;
};

// /////////////////////////////////////////////////////////////

struct Linear : public ActivationFunction {
  [[nodiscard]] auto operator()(value_type x) const -> value_type override;

  [[nodiscard]] auto derivative(value_type) const -> value_type override;

  [[nodiscard]] auto type() const -> Activation override;

  [[nodiscard]] auto to_string() const -> std::string override;

  [[nodiscard]] auto type_name() const -> std::string override;
};

// /////////////////////////////////////////////////////////////

struct ReLU : public ActivationFunction {
  [[nodiscard]] auto operator()(value_type x) const -> value_type override;

  [[nodiscard]] auto derivative(value_type) const -> value_type override;

  [[nodiscard]] auto type() const -> Activation override;

  [[nodiscard]] auto to_string() const -> std::string override;

  [[nodiscard]] auto type_name() const -> std::string override;
};

// /////////////////////////////////////////////////////////////

struct Sigmoid : public ActivationFunction {
  [[nodiscard]] auto operator()(value_type x) const -> value_type override;

  [[nodiscard]] auto derivative(value_type) const -> value_type override;

  [[nodiscard]] auto type() const -> Activation override;

  [[nodiscard]] auto to_string() const -> std::string override;

  [[nodiscard]] auto type_name() const -> std::string override;
};

// /////////////////////////////////////////////////////////////

struct SoftPlus : public ActivationFunction {
  [[nodiscard]] auto operator()(value_type x) const -> value_type override;

  [[nodiscard]] auto derivative(value_type) const -> value_type override;

  [[nodiscard]] auto type() const -> Activation override;

  [[nodiscard]] auto to_string() const -> std::string override;

  [[nodiscard]] auto type_name() const -> std::string override;
};

// /////////////////////////////////////////////////////////////

struct Swish : public ActivationFunction {
  [[nodiscard]] auto operator()(value_type x) const -> value_type override;

  [[nodiscard]] auto derivative(value_type) const -> value_type override;

  [[nodiscard]] auto type() const -> Activation override;

  [[nodiscard]] auto to_string() const -> std::string override;

  [[nodiscard]] auto type_name() const -> std::string override;
};

// /////////////////////////////////////////////////////////////

struct TanH : public ActivationFunction {
  [[nodiscard]] auto operator()(value_type x) const -> value_type override;

  [[nodiscard]] auto derivative(value_type) const -> value_type override;

  [[nodiscard]] auto type() const -> Activation override;

  [[nodiscard]] auto to_string() const -> std::string override;

  [[nodiscard]] auto type_name() const -> std::string override;
};

}

#endif
