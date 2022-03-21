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

#ifndef CBRAINX__LOSS_FUNCTIONS_HH_
#define CBRAINX__LOSS_FUNCTIONS_HH_

#include <memory>
#include <string>

#include "dataset.hh"
#include "shape.hh"
#include "tensor.hh"
#include "type_aliases.hh"

namespace cbx {

enum class Loss { MeanSquaredError, BinaryCrossEntropy, CategoricalCrossEntropy, SparseCrossEntropy };

class LossFunction {
 public:
  using x_iter_type = Dataset::data_container_const_iterator;
  using y_iter_type = Dataset::targets_container_const_iterator;

  // /////////////////////////////////////////////////////////////

  [[nodiscard]] virtual auto calculate(x_iter_type x_begin, x_iter_type x_end, y_iter_type y_begin) const
      -> f32 = 0;

  [[nodiscard]] virtual auto derivative(x_iter_type x_begin, x_iter_type x_end, y_iter_type y_begin) const
      -> f32 = 0;

  [[nodiscard]] virtual auto type() const -> Loss = 0;

  [[nodiscard]] virtual auto to_string() const -> std::string = 0;

  [[nodiscard]] virtual auto type_name() const -> std::string = 0;
};

// /////////////////////////////////////////////////////////////

class MeanSquaredError : public LossFunction {
 public:
  [[nodiscard]] auto calculate(x_iter_type x_begin, x_iter_type x_end, y_iter_type y_begin) const
      -> f32 override;

  [[nodiscard]] auto derivative(x_iter_type x_begin, x_iter_type x_end, y_iter_type y_begin) const
      -> f32 override;

  [[nodiscard]] auto type() const -> Loss override;

  [[nodiscard]] auto to_string() const -> std::string override;

  [[nodiscard]] auto type_name() const -> std::string override;
};

// /////////////////////////////////////////////////////////////

class BinaryCrossEntropy : public LossFunction {
 public:
  [[nodiscard]] auto calculate(x_iter_type x_begin, x_iter_type x_end, y_iter_type y_begin) const
      -> f32 override;

  [[nodiscard]] auto derivative(x_iter_type x_begin, x_iter_type x_end, y_iter_type y_begin) const
      -> f32 override;

  [[nodiscard]] auto type() const -> Loss override;

  [[nodiscard]] auto to_string() const -> std::string override;

  [[nodiscard]] auto type_name() const -> std::string override;
};

// /////////////////////////////////////////////////////////////

class CategoricalCrossEntropy : public LossFunction {
 public:
  [[nodiscard]] auto calculate(x_iter_type x_begin, x_iter_type x_end, y_iter_type y_begin) const
      -> f32 override;

  [[nodiscard]] auto derivative(x_iter_type x_begin, x_iter_type x_end, y_iter_type y_begin) const
      -> f32 override;

  [[nodiscard]] auto type() const -> Loss override;

  [[nodiscard]] auto to_string() const -> std::string override;

  [[nodiscard]] auto type_name() const -> std::string override;
};

// /////////////////////////////////////////////////////////////

class SparseCrossEntropy : public LossFunction {
 public:
  [[nodiscard]] auto calculate(x_iter_type x_begin, x_iter_type, y_iter_type y_begin) const -> f32 override;

  [[nodiscard]] auto derivative(x_iter_type x_begin, x_iter_type x_end, y_iter_type y_begin) const
      -> f32 override;

  [[nodiscard]] auto type() const -> Loss override;

  [[nodiscard]] auto to_string() const -> std::string override;

  [[nodiscard]] auto type_name() const -> std::string override;
};

// /////////////////////////////////////////////////////////////

class LossFunctionFactory {
 public:
  [[nodiscard]] static auto make(Loss loss_type) -> std::shared_ptr<LossFunction>;
};

}

#endif
