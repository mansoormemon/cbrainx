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

#include "type_aliases.hh"

namespace cbx {

struct ActivationFunction {
  using value_type = f32;

  // /////////////////////////////////////////////////////////////

  [[nodiscard]] virtual auto operator()(value_type x) const -> value_type = 0;

  [[nodiscard]] virtual auto derivative(value_type) const -> value_type = 0;

  [[nodiscard]] virtual auto to_string() -> str = 0;
};

// /////////////////////////////////////////////////////////////

struct ArcTan : public ActivationFunction {
  [[nodiscard]] auto operator()(value_type x) const -> value_type override;

  [[nodiscard]] auto derivative(value_type) const -> value_type override;

  [[nodiscard]] auto to_string() -> str override;
};

// /////////////////////////////////////////////////////////////

struct BinaryStep : public ActivationFunction {
  [[nodiscard]] auto operator()(value_type x) const -> value_type override;

  [[nodiscard]] auto derivative(value_type) const -> value_type override;

  [[nodiscard]] auto to_string() -> str;
};

// /////////////////////////////////////////////////////////////

struct ELU : public ActivationFunction {
  static constexpr const value_type ALPHA = 1.0F;

  // /////////////////////////////////////////////////////////////

  [[nodiscard]] auto operator()(value_type x) const -> value_type override;

  [[nodiscard]] auto derivative(value_type) const -> value_type override;

  [[nodiscard]] auto to_string() -> str override;
};

// /////////////////////////////////////////////////////////////

struct Gaussian : public ActivationFunction {
  [[nodiscard]] auto operator()(value_type x) const -> value_type override;

  [[nodiscard]] auto derivative(value_type) const -> value_type override;

  [[nodiscard]] auto to_string() -> str override;
};

// /////////////////////////////////////////////////////////////

struct GELU : public ActivationFunction {
  static constexpr const value_type C = 1.872F;

  // /////////////////////////////////////////////////////////////

  [[nodiscard]] auto operator()(value_type x) const -> value_type override;

  [[nodiscard]] auto derivative(value_type) const -> value_type override;

  [[nodiscard]] auto to_string() -> str override;
};
// /////////////////////////////////////////////////////////////

struct LeakyReLU : public ActivationFunction {
  static constexpr const value_type M = 0.1F;

  [[nodiscard]] auto operator()(value_type x) const -> value_type override;

  [[nodiscard]] auto derivative(value_type) const -> value_type override;

  [[nodiscard]] auto to_string() -> str override;
};

// /////////////////////////////////////////////////////////////

struct Linear : public ActivationFunction {
  [[nodiscard]] auto operator()(value_type x) const -> value_type override;

  [[nodiscard]] auto derivative(value_type) const -> value_type override;

  [[nodiscard]] auto to_string() -> str;
};

// /////////////////////////////////////////////////////////////

struct ReLU : public ActivationFunction {
  [[nodiscard]] auto operator()(value_type x) const -> value_type override;

  [[nodiscard]] auto derivative(value_type) const -> value_type override;

  [[nodiscard]] auto to_string() -> str override;
};

// /////////////////////////////////////////////////////////////

struct Sigmoid : public ActivationFunction {
  [[nodiscard]] auto operator()(value_type x) const -> value_type override;

  [[nodiscard]] auto derivative(value_type) const -> value_type override;

  [[nodiscard]] auto to_string() -> str;
};

// /////////////////////////////////////////////////////////////

struct SoftPlus : public ActivationFunction {
  [[nodiscard]] auto operator()(value_type x) const -> value_type override;

  [[nodiscard]] auto derivative(value_type) const -> value_type override;

  [[nodiscard]] auto to_string() -> str override;
};

// /////////////////////////////////////////////////////////////

struct Swish : public ActivationFunction {
  [[nodiscard]] auto operator()(value_type x) const -> value_type override;

  [[nodiscard]] auto derivative(value_type) const -> value_type override;

  [[nodiscard]] auto to_string() -> str override;
};

// /////////////////////////////////////////////////////////////

struct TanH : public ActivationFunction {
  [[nodiscard]] auto operator()(value_type x) const -> value_type override;

  [[nodiscard]] auto derivative(value_type) const -> value_type override;

  [[nodiscard]] auto to_string() -> str override;
};

}

#endif
