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

#ifndef CBRAINX__MATRIX_HH_
#define CBRAINX__MATRIX_HH_

#include <algorithm>

#include "shape.hh"
#include "tensor.hh"
#include "type_aliases.hh"
#include "type_concepts.hh"

namespace cbx {

class Matrix {
 public:
  static constexpr Shape::size_type DIMENSIONS = 2;

 private:
  static auto rank_check(Shape::size_type rank) -> void;

  static auto shape_equality_check(const Shape &a, const Shape &b) -> void;

 public:
  template <typename T = f32>
  [[nodiscard]] static auto make() -> Tensor<T> {
    return Tensor<T>{{1, 1}};
  }

  template <typename T = f32>
  [[nodiscard]] static auto make(Shape::value_type rows, Shape::value_type cols) -> Tensor<T> {
    return Tensor<T>{{rows, cols}};
  }

  // /////////////////////////////////////////////////////////////

  template <typename T = f32>
  [[nodiscard]] static auto zeros(Shape::value_type rows, Shape::value_type cols) -> Tensor<T> {
    return Tensor<T>::zeros({rows, cols});
  }

  template <typename T = f32>
  [[nodiscard]] static auto ones(Shape::value_type rows, Shape::value_type cols) -> Tensor<T> {
    return Tensor<T>::ones({rows, cols});
  }

  template <typename T = f32>
  [[nodiscard]] static auto fill(Shape::value_type rows, Shape::value_type cols, T value) -> Tensor<T> {
    return Tensor<T>::fill({rows, cols}, value);
  }

  template <typename T = f32>
  [[nodiscard]] static auto random(Shape::value_type rows, Shape::value_type cols, u64 seed = 1,
                                   T lower_bound = 0, T upper_bound = 1) -> Tensor<T> {
    return Tensor<T>::random({rows, cols}, seed, lower_bound, upper_bound);
  }

  template <typename T = f32, typename Lambda>
  [[nodiscard]] static auto custom(Shape::value_type rows, Shape::value_type cols, Lambda func) -> Tensor<T> {
    return Tensor<T>::custom({rows, cols}, func);
  }

  template <typename T = f32>
  [[nodiscard]] static auto scalar(Shape::value_type size, T value) -> Tensor<T> {
    const auto EXTRA_STEP = 1;
    auto mat = Tensor<T>{{size, size}};
    for (auto it = mat.begin(), end = mat.end(); it < end; it += size + EXTRA_STEP) {
      *it = value;
    }
    return mat;
  }

  template <typename T = f32>
  [[nodiscard]] static auto identity(Shape::value_type size) -> Tensor<T> {
    return Matrix::scalar<T>(size, 1);
  }

  // /////////////////////////////////////////////////////////////

  template <typename T, typename U>
  static auto add(Tensor<T> &a, const Tensor<U> &b) -> Tensor<T> & {
    Matrix::rank_check(a.rank());
    Matrix::rank_check(b.rank());
    Matrix::shape_equality_check(a.shape(), b.shape());
    std::transform(a.begin(), a.end(), b.begin(), a.begin(), [](const auto &a_x, const auto &b_x) {
      return a_x + b_x;
    });
    return a;
  }

  template <typename T>
  static auto add(Tensor<T> &a, Number auto b) -> Tensor<T> & {
    Matrix::rank_check(a.rank());
    std::transform(a.begin(), a.end(), a.begin(), [b](const auto &a_x) {
      return a_x + b;
    });
    return a;
  }

  // /////////////////////////////////////////////////////////////

  template <typename T, typename U>
  static auto subtract(Tensor<T> &a, const Tensor<U> &b) -> Tensor<T> & {
    Matrix::rank_check(a.rank());
    Matrix::rank_check(b.rank());
    Matrix::shape_equality_check(a.shape(), b.shape());
    std::transform(a.begin(), a.end(), b.begin(), a.begin(), [](const auto &a_x, const auto &b_x) {
      return a_x - b_x;
    });
    return a;
  }

  template <typename T>
  static auto subtract(Tensor<T> &a, Number auto b) -> Tensor<T> & {
    Matrix::rank_check(a.rank());
    std::transform(a.begin(), a.end(), a.begin(), [b](const auto &a_x) {
      return a_x - b;
    });
    return a;
  }

  // /////////////////////////////////////////////////////////////

  template <typename T>
  static auto multiply(Tensor<T> &a, Number auto factor) -> Tensor<T> & {
    Matrix::rank_check(a.rank());
    std::transform(a.begin(), a.end(), a.begin(), [factor](const auto &a_x) {
      return a_x * factor;
    });
    return a;
  }
};

}

#endif
