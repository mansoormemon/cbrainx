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
#include <cmath>
#include <thread>
#include <vector>

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

  static auto multiplication_compatibility_check(Shape::value_type c1, Shape::value_type r2) -> void;

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

  template <typename T, typename U>
  static auto multiply(const Tensor<T> &a, const Tensor<U> &b, bool multithreading = true) -> Tensor<T> {
    Matrix::rank_check(a.rank());
    Matrix::rank_check(b.rank());

    auto [r1, c1] = a.shape().template unwrap<2>();
    auto [r2, c2] = b.shape().template unwrap<2>();

    Matrix::multiplication_compatibility_check(c1, r2);

    auto rows = r1, cols = c2, common_index = c1;
    auto product = Matrix::make<T>(rows, cols);

    // Estimates how many rows each thread will be assigned based on the number of rows in the product matrix.
    auto calculate_rows_per_thread = [](auto rows) -> Shape::size_type {
      const auto BITS_IN_BYTE = 8;
      // Arbitrarily establish a relation between thread count and matrix size.
      auto ARCHITECTURE_BITS = sizeof(std::ptrdiff_t) * BITS_IN_BYTE;
      auto factor = ARCHITECTURE_BITS * std::log(rows + BITS_IN_BYTE);
      return std::floor(factor);
    };

    // Calculates how many threads will be required based on the number of rows assigned to each thread.
    auto calculate_threads_required = [](f32 rows, auto rows_per_thread) -> Shape::size_type {
      return std::ceil(rows / rows_per_thread);
    };

    // The actual implementation for matrix multiplication without any multithreading witchcraft. It's a primary
    // schoolbook algorithm disencumbered by any optimization.
    auto impl = [&a, &b, &product, cols, common_index](auto row_start, auto row_count) {
      auto row_end = row_start + row_count;
      for (auto r = row_start; r < row_end; ++r) {
        for (Shape::value_type c = {}; c < cols; ++c) {
          for (Shape::value_type k = {}; k < common_index; ++k) {
            product(r, c) += a(r, k) * b(k, c);
          }
        }
      }
    };

    // If multithreading is unsought, ordinarily call the implementation lambda and return the product.
    if (not multithreading) {
      impl(Shape::size_type{}, rows);
      return product;
    }

    auto rows_per_thread = calculate_rows_per_thread(rows);
    auto threads_required = calculate_threads_required(rows, rows_per_thread);

    // Bookkeeping threads to call later for joining into the main thread.
    auto threads = std::vector<std::thread>{};
    threads.reserve(threads_required);

    // Construct each thread with implementation lambda, position, and span of the thread in the product matrix.
    for (Shape::size_type current_row = {}; current_row < rows; current_row += rows_per_thread) {
      auto distance = rows - current_row;
      auto thread_rows_span = std::min(rows_per_thread, distance);
      threads.emplace_back(impl, current_row, thread_rows_span);
    }

    // Call to join the main thread.
    for (auto &thread : threads) {
      thread.join();
    }

    return product;
  }
};

}

#endif
