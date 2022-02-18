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

#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>

#include "shape.hh"
#include "type_aliases.hh"
#include "type_concepts.hh"

#include <fmt/format.h>

namespace cbx {

/**
 * @brief The <b>Tensor</b> class is a convenient structure to represent an n-dimensional tensor.
 *
 * @tparam T The datatype of the tensor (must be arithmetic).
 */
template <Number T = f32>
class Tensor {
 public:
  using container = std::vector<T>;

  using value_type = typename container::value_type;

  using reference = typename container::reference;
  using const_reference = typename container::const_reference;

  using pointer = typename container::pointer;
  using const_pointer = typename container::const_pointer;

  using size_type = typename container::size_type;
  using difference_type = typename container::difference_type;

  using iterator = typename container::iterator;
  using const_iterator = typename container::const_iterator;

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
      throw std::invalid_argument{
          "cbx::Tensor::shape_compatibility_check: `this->shape().is_compatible(other)` is false"};
    }
  }

  template <Integer... Args>
  constexpr auto dimensionality_check(Args... indices) const -> void {
    auto indices_count = sizeof...(indices);
    auto rank = this->rank();
    // The number of indices must be equal to the rank of the tensor.
    if (indices_count != rank) {
      throw std::invalid_argument{fmt::format(
          "cbx::Tensor::dimensionality_check: indices(count={}) contradict the rank(={}) of the tensor",
          indices_count, rank)};
    }
  }

  template <Integer... Args>
  constexpr auto dimensional_range_check(Args... indices) const -> void {
    this->dimensionality_check(indices...);

    [[maybe_unused]] auto dimension = Shape::size_type{};
    // An anonymous lambda that checks the range of each dimension by recursively unpacking the parameter pack.
    (
        [this, &dimension](auto &dim_index) {
          auto dimension_size = shape_[dimension];
          if (auto i = static_cast<Shape::size_type>(dim_index); i >= dimension_size) {
            throw std::out_of_range{fmt::format("cbx::Tensor::dimensional_range_check: `dim_index(={}) >= "
                                                "this->shape()[dimension(={})](={})` is true",
                                                i, dimension, dimension_size)};
          }
          dimension += 1;
        }(indices),
        ...);
  }

  template <Integer... Args>
  [[nodiscard]] constexpr auto get_linearized_index(Args... indices) const -> size_type {
    this->dimensional_range_check(indices...);

    auto linearized_index = size_type{};

    [[maybe_unused]] auto dimension = size_type{};
    // An anonymous lambda that computes a linear index from the given non-linear indices by recursively
    // expanding the parameter pack. The recursion propagates from the higher to lower dimensions.
    (
        [this, &linearized_index, &dimension](auto dim_index) {
          linearized_index += dim_index
                            * std::accumulate(shape_.begin() + dimension + 1, shape_.end(),
                                              Shape::UNIT_DIMENSION_SIZE, std::multiplies());
          dimension += 1;
        }(indices),
        ...);

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
