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

#ifndef CBRAINX__TENSOR_HH_
#define CBRAINX__TENSOR_HH_

#include <algorithm>
#include <iterator>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>

#include <fmt/format.h>

#include "exceptions.hh"
#include "shape.hh"
#include "type_aliases.hh"
#include "type_concepts.hh"

namespace cbx {

/**
 * @brief The <b>Tensor</b> class is a convenient structure to represent an n-dimensional tensor.
 *
 * @tparam T The datatype of the tensor (must be arithmetic).
 */
template <Number T = f32>
class Tensor {
 public:
  using value_type = T;

  using container = std::vector<value_type>;

  using reference = typename container::reference;
  using const_reference = typename container::const_reference;

  using pointer = typename container::pointer;
  using const_pointer = typename container::const_pointer;

  using size_type = typename container::size_type;
  using difference_type = typename container::difference_type;

  using iterator = typename container::iterator;
  using const_iterator = typename container::const_iterator;

  // /////////////////////////////////////////////////////////////

  static constexpr shape_size_t SCALAR_RANK = 0;

 private:
  Shape shape_ = {};
  container data_ = container(shape_.total());

  // /////////////////////////////////////////////////////////////

  constexpr auto linear_range_check(size_type index) const -> void {
    auto total = this->total();
    if (index >= total) {
      throw std::out_of_range{fmt::format(
          "cbx::Tensor::linear_range_check: `index(={}) >= this->total()(={})` is true", index, total)};
    }
  }

  constexpr auto shape_compatibility_check(const Shape &other) const -> void {
    // The total number of elements for both shapes must match.
    if (not shape_.is_compatible(other)) {
      throw ShapeError{"cbx::Tensor::shape_compatibility_check: `this->shape().is_compatible(other)` is false"};
    }
  }

  template <Integer... Args>
  constexpr auto dimensionality_check(Args... indices) const -> void {
    auto indices_count = sizeof...(indices);
    auto rank = this->rank();
    // The number of indices must be equal to the rank of the tensor.
    if (indices_count != rank) {
      throw RankError{fmt::format(
          "cbx::Tensor::dimensionality_check: indices(count={}) contradict the rank(={}) of the tensor",
          indices_count, rank)};
    }
  }

  template <Integer... Args>
  constexpr auto dimensional_range_check(Args... indices) const -> void {
    this->dimensionality_check(indices...);

    auto ilist_indices = std::initializer_list<shape_value_t>{static_cast<shape_value_t>(indices)...};
    for (auto shape_it = shape_.begin(); auto dim_index : ilist_indices) {
      if (dim_index >= *shape_it) {
        throw std::out_of_range{fmt::format("cbx::Tensor::dimensional_range_check: `dim_index(={}) >= "
                                            "this->shape()[dimension(={})](={})` is true",
                                            dim_index, std::distance(shape_.begin(), shape_it), *shape_it)};
      }
      ++shape_it;
    }
  }

  template <Integer... Args>
  [[nodiscard]] constexpr auto get_linearized_index(Args... indices) const -> size_type {
    this->dimensional_range_check(indices...);

    auto ilist_indices = std::initializer_list<shape_value_t>{static_cast<shape_value_t>(indices)...};
    size_type linearized_index = {};
    auto stride = Shape::UNIT_DIMENSION_SIZE;
    auto shape_r_it = std::reverse_iterator{shape_.end()};
    for (auto indices_r_it = std::reverse_iterator{ilist_indices.end()},
              indices_r_end = std::reverse_iterator{ilist_indices.begin()};
         indices_r_it != indices_r_end; ++indices_r_it) {
      linearized_index += *indices_r_it * stride;
      stride *= *shape_r_it;
      ++shape_r_it;
    }
    return linearized_index;
  }

 public:
  constexpr Tensor() = default;

  constexpr Tensor(const Tensor &other) = default;

  constexpr Tensor(Tensor &&other) noexcept : shape_{std::move(other.shape_)}, data_(std::move(other.data_)) {}

  explicit Tensor(Shape shape, value_type value = {})
      : shape_{std::move(shape)}, data_(shape_.total(), value) {}

  constexpr ~Tensor() = default;

  // /////////////////////////////////////////////////////////////

  constexpr auto operator=(const Tensor &other) -> Tensor & = default;

  constexpr auto operator=(Tensor &&other) noexcept -> Tensor & {
    shape_ = std::move(other.shape_);
    data_ = std::move(other.data_);
    return *this;
  }

  auto operator=(Shape shape) -> Tensor & {
    shape_ = std::move(shape);
    data_ = container(shape_.total());
    return *this;
  }

  // /////////////////////////////////////////////////////////////

  [[nodiscard]] constexpr auto operator[](size_type index) const noexcept -> const_reference {
    return data_[index];
  }

  constexpr auto operator[](size_type index) noexcept -> reference { return data_[index]; }

  [[nodiscard]] constexpr auto at(size_type index) const -> const_reference {
    this->linear_range_check(index);
    return data_[index];
  }

  constexpr auto at(size_type index) -> reference {
    this->linear_range_check(index);
    return data_[index];
  }

  template <Integer... Args>
  auto operator()(Args... indices) const -> const_reference {
    return data_[get_linearized_index(indices...)];
  }

  template <Integer... Args>
  auto operator()(Args... indices) -> reference {
    return data_[get_linearized_index(indices...)];
  }

  // /////////////////////////////////////////////////////////////

  [[nodiscard]] constexpr auto shape() const noexcept -> const Shape & { return shape_; }

  [[nodiscard]] constexpr auto data() const noexcept -> const_pointer { return data_.data(); }

  constexpr auto data() noexcept -> pointer { return data_.data(); }

  [[nodiscard]] constexpr auto total() const noexcept -> size_type { return data_.size(); }

  [[nodiscard]] constexpr auto rank() const noexcept -> size_type { return shape_.dimensions(); }

  // /////////////////////////////////////////////////////////////

  [[nodiscard]] constexpr auto begin() const noexcept -> const_iterator { return data_.begin(); }

  constexpr auto begin() noexcept -> iterator { return data_.begin(); }

  [[nodiscard]] constexpr auto end() const noexcept -> const_iterator { return data_.end(); }

  constexpr auto end() noexcept -> iterator { return data_.end(); }

  // /////////////////////////////////////////////////////////////

  [[nodiscard]] constexpr auto is_scalar() const noexcept -> bool { return shape_.is_scalar(); }

  // /////////////////////////////////////////////////////////////

  constexpr auto reshape(const Shape &shape) -> Tensor & {
    this->shape_compatibility_check(shape);
    shape_ = shape;
    return *this;
  }

  auto crampy_reshape(size_type new_rank) -> Tensor & {
    auto cur_rank = this->rank();
    if (cur_rank == new_rank) {
      return *this;
    }
    if (new_rank == SCALAR_RANK) {
      auto new_shape = Shape{};
      this->shape_compatibility_check(new_shape);
      shape_ = std::move(new_shape);
    } else if (new_rank < cur_rank) {
      auto new_shape = shape_.clone().resize(new_rank);
      auto last_index = new_rank - 1;
      auto cramped_dim_val = std::accumulate(shape_.begin() + last_index, shape_.end(),
                                             Shape::UNIT_DIMENSION_SIZE, std::multiplies());
      new_shape.set_dimension(last_index, cramped_dim_val);

      this->shape_compatibility_check(new_shape);
      shape_ = std::move(new_shape);
    } else {
      shape_.resize(new_rank);
    }
    return *this;
  }

  constexpr auto flatten() -> Tensor & { return this->crampy_reshape(1); }

  // /////////////////////////////////////////////////////////////

  [[nodiscard]] constexpr auto clone() const -> Tensor { return *this; }

  constexpr auto swap(Tensor &other) noexcept -> Tensor & {
    shape_.swap(other.shape_);
    data_.swap(other.data_);
    return *this;
  }

  // /////////////////////////////////////////////////////////////

  [[nodiscard]] auto meta_info() const -> std::string {
    return fmt::format("{{ total={}, rank={}, shape={}, is_scalar={} }}", this->total(), this->rank(),
                       this->shape().to_string(), this->is_scalar());
  }

  // /////////////////////////////////////////////////////////////////////////////////////////////

  [[nodiscard]] static constexpr auto zeros(const Shape &shape) -> Tensor { return Tensor{shape}; }

  [[nodiscard]] static constexpr auto ones(const Shape &shape) -> Tensor { return Tensor{shape, 1}; }

  [[nodiscard]] static constexpr auto fill(const Shape &shape, value_type value) -> Tensor {
    return Tensor{shape, value};
  }

  [[nodiscard]] static auto random(const Shape &shape, u32 seed = 1U, value_type lower_bound = 0,
                                   value_type upper_bound = 1) -> Tensor {
    auto tensor = Tensor{shape};
    auto randomizer = std::default_random_engine(seed);
    auto engine = std::mt19937_64{randomizer()};
    using distributer_type =
        std::conditional_t<std::is_integral_v<value_type>, std::uniform_int_distribution<value_type>,
                           std::uniform_real_distribution<value_type>>;
    auto distributor = distributer_type{lower_bound, upper_bound};
    std::generate(tensor.begin(), tensor.end(), [&engine, &distributor]() {
      return distributor(engine);
    });
    return tensor;
  }

  template <typename Lambda>
  [[nodiscard]] static auto custom(const Shape &shape, Lambda func) -> Tensor {
    auto tensor = Tensor{shape};
    std::generate(tensor.begin(), tensor.end(), func);
    return tensor;
  }

  template <std::input_iterator I_It>
  [[nodiscard]] static auto copy(const Shape &shape, I_It src_first, I_It src_last) -> Tensor {
    auto tensor = Tensor{shape};
    std::copy(src_first, src_last, tensor.begin());
    return tensor;
  }
};

}

#endif
