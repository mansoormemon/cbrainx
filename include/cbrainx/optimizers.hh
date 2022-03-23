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

#ifndef CBRAINX__OPTIMIZERS_HH_
#define CBRAINX__OPTIMIZERS_HH_

#include <memory>

#include "tensor.hh"
#include "type_aliases.hh"

namespace cbx {

class Optimizer {
 public:
  virtual auto update_params(Tensor<f32>::iterator dest_begin, Tensor<f32>::iterator dest_end,
                             Tensor<f32>::const_iterator gradients_begin) -> void = 0;
};

class GradientDescent : public Optimizer {
 private:
  f64 learning_rate_ = {};

 public:
  GradientDescent() = default;
  explicit GradientDescent(f64 learning_rate) : learning_rate_{learning_rate} {}

  auto update_params(Tensor<f32>::iterator dest_begin, Tensor<f32>::iterator dest_end,
                     Tensor<f32>::const_iterator gradients_begin) -> void override {
    std::transform(dest_begin, dest_end, gradients_begin, dest_begin, [this](auto x, auto xd) {
      return x - (learning_rate_ * xd);
    });
  }
};

class OptimizerFactory {
 public:
  template <typename T, typename... Args>
  static auto make(Args... args) -> std::shared_ptr<Optimizer> {
    return std::make_shared<T>(args...);
  }
};

}

#endif
