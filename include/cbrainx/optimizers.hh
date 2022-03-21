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

namespace cbx {

template <typename T>
concept Optimizer = true;

class GradientDescent {
 private:
  f64 learning_rate = {};

 public:
  GradientDescent() = default;
  GradientDescent(f64 learning_rate) {}

  auto update_gradients() -> void {}
};

}

#endif
