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

#ifndef CBRAINX__SHAPE_HH_
#define CBRAINX__SHAPE_HH_

#include <iterator>
#include <ranges>
#include <string>
#include <tuple>
#include <vector>

#include "type_aliases.hh"

namespace cbx {

/**
 * @brief The <b>Shape</b> class represents the shape of a tensor. The length of each axis determines the shape
 * of a tensor.
 * @details An important thing to note is that zero is not acceptable as an axis length, which is justifiable
 * for two reasons. First, a length zero for an axis appears ambiguous. Second, the shape of a tensor determines
 * how much memory it gets, which would no longer be true if the product of its magnitudes became zero.
 */
class Shape {
 public:
  using value_type = usize;

  using container = std::vector<value_type>;

  using reference = container::reference;
  using const_reference = container::const_reference;

  using pointer = container::pointer;
  using const_pointer = container::const_pointer;

  using size_type = container::size_type;
  using difference_type = container::difference_type;

  using iterator = container::iterator;
  using const_iterator = container::const_iterator;

  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

  // /////////////////////////////////////////////////////////////

  /**
   * @brief Rank of a scalar.
   */
  static constexpr size_type SCALAR_RANK = 0;

  /**
   * @brief Rank of a vector/array.
   */
  static constexpr size_type VECTOR_RANK = 1;

  /**
   * @brief Rank of a matrix.
   */
  static constexpr size_type MATRIX_RANK = 2;

  /**
   * @brief Size of a scalar.
   */
  static constexpr size_type SCALAR_SIZE = 1;

 private:
  /**
   * @brief Container to hold the length of each axis.
   */
  container data_ = {};

  // /////////////////////////////////////////////////////////////

  /**
   * @brief Performs bounds checking.
   * @param index The index to be checked.
   *
   * @details This function will throw an exception only if @p index is out of range.
   */
  auto check_bounds(size_type index) const -> void;

  /**
   * @brief Performs rank checking.
   * @param N The rank to be checked.
   *
   * @details This function will throw an exception only if:
   * @code
   * N > this->rank()
   * @endcode
   */
  auto check_rank(size_type N) const -> void;

  /**
   * @brief Performs validation check(s).
   * @param value The axis size to be validated.
   *
   * @details This function will throw an exception only if the input value is zero, which is justifiable for
   * two reasons. First, a length zero for an axis appears ambiguous. Second, the shape of a tensor determines
   * how much memory it gets, which would no longer be true if the product of its magnitudes became zero.
   */
  static auto validate_input(value_type value) -> void;

  /**
   * @brief Performs validation check(s) for a range.
   * @param first The beginning of the range to validate.
   * @param last The ending of the range to validate.
   *
   * @details Same as scalar counter part.
   */
  template <std::input_iterator I_It>
  static auto validate_input(I_It first, I_It last) -> void {
    for (std::input_iterator auto it = first; it != last; ++it) {
      Shape::validate_input(*it);
    }
  }

 public:
  Shape() = default;

  Shape(const Shape &other) = default;

  Shape(Shape &&other) noexcept;

  Shape(std::initializer_list<value_type> ilist);

  template <std::input_iterator I_It>
  Shape(I_It first, I_It last) {
    Shape::validate_input(first, last);
    data_.assign(first, last);
  }

  explicit Shape(const std::ranges::range auto &range) {
    Shape::validate_input(range.begin(), range.end());
    data_.assign(range.begin(), range.end());
  }

  ~Shape() = default;

  // /////////////////////////////////////////////////////////////

  auto operator=(const Shape &other) -> Shape & = default;

  auto operator=(Shape &&other) noexcept -> Shape &;

  // /////////////////////////////////////////////////////////////

  /**
   * @brief Accesses elements at the specified index.
   * @param index The index to be accessed.
   * @return Element at the specified index.
   *
   * @note This function does not perform bounds checking.
   */
  [[nodiscard]] auto operator[](size_type index) const noexcept -> const_reference;

  /**
   * @brief Accesses elements at the specified index.
   * @param index The index to be accessed.
   * @return Element at the specified index.
   *
   * @note This function performs bounds checking.
   */
  [[nodiscard]] auto at(size_type index) const -> const_reference;

  // /////////////////////////////////////////////////////////////

  /**
   * @brief Returns the underlying pointer to data in memory.
   * @return A const-qualified pointer to data.
   */
  [[nodiscard]] auto data() const noexcept -> const_pointer;

  /**
   * @brief Returns the underlying container holding the data.
   * @return A const-qualified reference to container.
   */
  [[nodiscard]] auto underlying_container() const noexcept -> const container &;

  /**
   * @brief Returns the rank of the shape i.e. its number of dimensions.
   * @return Rank of shape.
   */
  [[nodiscard]] auto rank() const noexcept -> size_type;

  /**
   * @brief Sets the length of the specified axis.
   * @param index The index of the axis.
   * @param value The length of the axis.
   */
  auto set_axis(size_type index, value_type value) -> Shape &;

  // /////////////////////////////////////////////////////////////

  /**
   * @brief Returns the length of first axis.
   * @return Length of first axis.
   *
   * @note Behaviour is undefined for scalars.
   */
  [[nodiscard]] auto front() const -> const_reference;

  /**
   * @brief Returns the length of last axis.
   * @return Length of last axis.
   *
   * @note Behaviour is undefined for scalars.
   */
  [[nodiscard]] auto back() const -> const_reference;

  // /////////////////////////////////////////////////////////////

  /**
   * @brief Returns a const-qualified iterator pointing to the beginning of the shape.
   * @return A const-qualified iterator to beginning.
   */
  [[nodiscard]] auto begin() const noexcept -> const_iterator;

  /**
   * @brief Returns a const-qualified reverse iterator pointing to the reverse beginning of the shape.
   * @return A const-qualified reverse iterator to reverse beginning.
   */
  [[nodiscard]] auto rbegin() const noexcept -> const_reverse_iterator;

  /**
   * @brief Returns a const-qualified iterator pointing to the ending of the shape.
   * @return A const-qualified iterator to ending.
   */
  [[nodiscard]] auto end() const noexcept -> const_iterator;

  /**
   * @brief Returns a const-qualified reverse iterator pointing to the reverse ending of the shape.
   * @return A const-qualified reverse iterator to reverse ending.
   */
  [[nodiscard]] auto rend() const noexcept -> const_reverse_iterator;

  // /////////////////////////////////////////////////////////////

  /**
   * @brief Checks if the shape can be attributed to a scalar.
   * @return
   * @code
   * this->rank() == SCALAR_RANK
   * @endcode
   */
  [[nodiscard]] auto is_scalar() const noexcept -> bool;

  /**
   * @brief Checks if the shape can be attributed to a vector or array.
   * @return
   * @code
   * this->rank() == VECTOR_RANK
   * @endcode
   */
  [[nodiscard]] auto is_vector() const noexcept -> bool;

  /**
   * @brief Checks if the shape can be attributed to a matrix.
   * @return
   * @code
   * this->rank() == MATRIX_RANK
   * @endcode
   */
  [[nodiscard]] auto is_matrix() const noexcept -> bool;

  /**
   * @brief Checks if two shapes are equivalent i.e. they have the same number of total elements.
   * @param other The shape to be compared for equivalence.
   * @return
   * @code
   * this->total == other.total()
   * @endcode
   */
  [[nodiscard]] auto is_equivalent(const Shape &other) const noexcept -> bool;

  // /////////////////////////////////////////////////////////////

  /**
   * @brief Returns the total number of elements.
   * @return Total number of elements.
   *
   * @note This function returns SCALAR_SIZE if the shape represents a scalar.
   */
  [[nodiscard]] auto total() const noexcept -> size_type;

  // /////////////////////////////////////////////////////////////

  /**
   * @brief Resizes the shape to the specified rank.
   */
  auto resize(size_type rank) -> Shape &;

  /**
   * @brief Swaps current shape with the other.
   * @param other The shape with which the swap will be performed.
   */
  auto swap(Shape &other) noexcept -> Shape &;

  // /////////////////////////////////////////////////////////////

  /**
   * @brief Clones the current shape.
   * @return A clone of the current shape.
   */
  [[nodiscard]] auto clone() const -> Shape;

 private:
  /**
   * @brief Helper for unwrap().
   * @tparam T Type for static_casting.
   * @tparam Indices Size of tuple.
   * @return A tuple of axes lengths.
   */
  template <typename T, size_type... Indices>
  constexpr auto unwrap_helper(std::index_sequence<Indices...>) const {
    return std::make_tuple(static_cast<T>(data_[Indices])...);
  }

 public:
  /**
   * @brief Returns first @p N axes lengths as a tuple.
   * @returns A tuple of @p N elements each of type @p T.
   */
  template <size_type N, typename T = value_type>
  constexpr auto unwrap() const {
    this->check_rank(N);
    return unwrap_helper<T>(std::make_index_sequence<N>());
  }

  // /////////////////////////////////////////////////////////////

  /**
   * @brief Returns a string containing meta information.
   * @return A meta information string.
   */
  [[nodiscard]] auto meta_info() const noexcept -> std::string;

  /**
   * @brief Converts the shape to a string.
   * @return Shape as a string.
   */
  [[nodiscard]] auto to_string() const noexcept -> std::string;

  // /////////////////////////////////////////////////////////////////////////////////////////////

  friend auto operator==(const Shape &a, const Shape &b) noexcept -> bool;

  friend auto operator!=(const Shape &a, const Shape &b) noexcept -> bool;
};

using shape_value_t = typename Shape::value_type;
using shape_size_t = typename Shape::size_type;

}

#endif
