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

// Copyright (c) 2021 Mansoor Ahmed Memon <mansoorahmed.one@gmail.com>

#ifndef CBRAINX__TENSOR_HH_
#define CBRAINX__TENSOR_HH_

#include <algorithm>
#include <iterator>
#include <numeric>
#include <random>
#include <ranges>
#include <stdexcept>
#include <string>
#include <utility>

#include <fmt/format.h>

#include "custom_iterators.hh"
#include "exceptions.hh"
#include "shape.hh"
#include "type_aliases.hh"
#include "type_concepts.hh"

namespace cbx {

/**
 * @brief The Tensor class represents an n-dimensional vector (or array).
 * @tparam T Data type (must be arithmetic).
 *
 * @details A tensor is a multidimensional array or an abstraction of vectors and matrices. In practice, it is a
 * container that can harbor identical numeric data in an N-dimensional space. Its shape describes the
 * dimensionality of the data. The shape must be defined at the time of instantiation.
 *
 * Consider the following code snippet:
 *
 * @code
 * cbx::Tensor<cbx::i32>{{}, 3};             // { total=1, rank=0, shape=() }
 * cbx::Tensor<cbx::f32>{{8}, 1.34};         // { total=8, rank=1, shape=(8) }
 * cbx::Tensor<cbx::f32>{{3, 4}, 4.2};       // { total=12, rank=2, shape=(3, 4) }
 * cbx::Tensor<cbx::f32>{{3, 4, 8}, 4.2};    // { total=96, rank=3, shape=(3, 4, 8) }
 * @endcode
 *
 * @note The words `dimension (plural: dimensions)` and `axis (plural: axes)` are used interchangeably
 * throughout this documentation.
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

  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

  // /////////////////////////////////////////////////////////////
  // Constants
  // /////////////////////////////////////////////////////////////

  /**
   * @brief Rank of a scalar.
   */
  static constexpr size_type SCALAR_RANK = 0;

  /**
   * @brief Rank of a vector (or array).
   */
  static constexpr size_type VECTOR_RANK = 1;

  /**
   * @brief Rank of a matrix.
   */
  static constexpr size_type MATRIX_RANK = 2;

 private:
  /**
   * @brief A flag for enabling or disabling bounds checking.
   */
  bool bounds_checking_ = true;

  /**
   * @brief Shape of data.
   */
  Shape shape_ = {};

  /**
   * @brief Container to hold the data.
   */
  container data_ = container(shape_.total());

  // /////////////////////////////////////////////////////////////
  // Helpers
  // /////////////////////////////////////////////////////////////

  /**
   * @brief Checks if two shapes are equivalent i.e. they have the same number of total elements.
   * @param a, b The shapes to be compared.
   *
   * @details This function will throw an exception only if @p b is not equivalent to @p a.
   */
  static constexpr auto _s_check_shape_equivalency(const Shape &a, const Shape &b) -> void {
    if (not a.is_equivalent(b)) {
      custom_throw<ShapeError>(
          "cbx::Tensor::_s_check_shape_equivalency: a (total={}) must be equivalent to b (total={})", a.total(),
          b.total());
    }
  }

  /**
   * @brief Checks if two shapes are equal.
   * @param a, b The shapes to be compared.
   *
   * @details This function will throw an exception only if @p a is not equal to @p b.
   */
  static constexpr auto _s_check_shape_equality(const Shape &a, const Shape &b) -> void {
    if (a != b) {
      custom_throw<ShapeError>("cbx::Tensor::_s_check_shape_equality: a (={}) must be equal to b (={})",
                               a.to_string(), b.to_string());
    }
  }

  static constexpr auto _s_check_broadcastability(const Shape &a, const Shape &b) -> void {
    if (b.rank() > a.rank() or not std::equal(b.rbegin(), b.rend(), a.rbegin())) {
      custom_throw<ShapeError>(
          "cbx::Tensor::_s_check_broadcastability: b (={}) is not broadcastable to a (={})", b.to_string(),
          a.to_string());
    }
  }

  /**
   * @brief Performs bounds checking w.r.t. total elements.
   * @param index The index to be checked.
   *
   * @details This function will throw an exception only if @p index is out of range.
   */
  constexpr auto _m_check_linear_bounds(size_type index) const -> void {
    auto total_elements = total();
    if (index >= total_elements) {
      custom_throw<std::out_of_range>(
          "cbx::Tensor::_m_check_linear_bounds: index (={}) >= this->total() (={}) ", index, total_elements);
    }
  }

  /**
   * @brief Checks if the number of the given indices is equal to the rank.
   * @tparam Args Type of indices (must be integral).
   * @param indices Co-ordinates of element in an n-dimensional space.
   *
   * @details This function will throw an exception only if the number of indices is not equal to the rank.
   */
  template <Integer... Args>
  constexpr auto _m_check_rank(Args... indices) const -> void {
    auto indices_count = sizeof...(indices);
    auto cur_rank = rank();
    if (indices_count != cur_rank) {
      custom_throw<RankError>(
          "cbx::Tensor::_m_check_rank: indices (count={}) are in contradiction with the rank (={})",
          indices_count, cur_rank);
    }
  }

  /**
   * @brief Performs bounds checking for the given location in an n-dimensional space.
   * @tparam Args Type of indices (must be integral).
   * @param indices Co-ordinates of element in an n-dimensional space.
   *
   * @details This function will throw an exception only if:
   * * Any index is out of range w.r.t to its axis, provided that bounds checking is enabled.
   * * Rank contradicts with the number of indices.
   */
  template <Integer... Args>
  auto _m_check_axes_bounds(Args... indices) const -> void {
    _m_check_rank(indices...);
    if (not bounds_checking_) {
      return;
    }
    auto il_indices = std::initializer_list<usize>{usize(indices)...};
    for (auto shape_it = shape_.begin(); auto axis_index : il_indices) {
      if (axis_index >= (*shape_it)) {
        custom_throw<std::out_of_range>(
            "cbx::Tensor::_m_check_axes_bounds: axis_index (={}) >= this->shape()[axis (={})] (={})",
            axis_index, std::distance(shape_.begin(), shape_it), *shape_it);
      }
      ++shape_it;
    }
  }

  /**
   * @brief Calculates the linear index from the given indices.
   * @tparam Args Type of indices (must be integral).
   * @param indices A variable number of indices representing the location of element in an n-dimensional space.
   *
   * @details This function will throw an exception only if any index is out of range w.r.t to its axes length.
   */
  template <Integer... Args>
  [[nodiscard]] auto _m_linear_index(Args... indices) const -> size_type {
    _m_check_axes_bounds(indices...);

    auto il_indices = std::initializer_list<usize>{usize(indices)...};
    size_type linear_index = {};
    auto stride = Shape::SCALAR_SIZE;
    auto shape_r_it = shape_.rbegin();
    for (auto indices_r_it = std::rbegin(il_indices), indices_r_end = std::rend(il_indices);
         indices_r_it != indices_r_end; ++indices_r_it) {
      linear_index += (*indices_r_it) * stride;
      stride *= (*shape_r_it);
      ++shape_r_it;
    }
    return linear_index;
  }

 public:
  // /////////////////////////////////////////////////////////////
  // Constructors and Destructors
  // /////////////////////////////////////////////////////////////

  constexpr Tensor() = default;

  constexpr Tensor(const Tensor &other) = default;

  constexpr Tensor(Tensor &&other) noexcept
      : bounds_checking_{other.bounds_checking_}, shape_{std::move(other.shape_)},
        data_(std::move(other.data_)) {}

  explicit Tensor(Shape shape, value_type value = {})
      : shape_{std::move(shape)}, data_(shape_.total(), value) {}

  template <std::input_iterator I_It>
  Tensor(Shape shape, I_It first, I_It last) : shape_{std::move(shape)}, data_{first, last} {}

  Tensor(Shape shape, const std::ranges::range auto &range)
      : shape_{std::move(shape)}, data_{range.begin(), range.end()} {}

  constexpr ~Tensor() = default;

  // /////////////////////////////////////////////////////////////
  // Assignment Operators
  // /////////////////////////////////////////////////////////////

  constexpr auto operator=(const Tensor &other) -> Tensor & = default;

  constexpr auto operator=(Tensor &&other) noexcept -> Tensor & {
    bounds_checking_ = std::exchange(other.bounds_checking_, true);
    shape_ = std::move(other.shape_);
    data_ = std::move(other.data_);
    return *this;
  }

  // /////////////////////////////////////////////////////////////
  // Element Access
  // /////////////////////////////////////////////////////////////

  /**
   * @brief Accesses element in a linear fashion at the specified index.
   * @param index The index to be accessed.
   * @return A const-qualified reference to the element at the specified index.
   *
   * @note This function neither respects dimensionality of data nor performs bounds checking.
   */
  [[nodiscard]] constexpr auto operator[](size_type index) const noexcept -> const_reference {
    return data_[index];
  }

  /**
   * @brief Accesses element in a linear fashion at the specified index.
   * @param index The index to be accessed.
   * @return A reference to the element at the specified index.
   *
   * @note This function neither respects dimensionality of data nor performs bounds checking.
   */
  constexpr auto operator[](size_type index) noexcept -> reference { return data_[index]; }

  /**
   * @brief Accesses element in a linear fashion at the specified index.
   * @param index The index to be accessed.
   * @return A const-qualified to the element at the specified index.
   *
   * @note This function does not respect dimensionality of data but performs bounds checking w.r.t. total
   * elements.
   */
  [[nodiscard]] constexpr auto at(size_type index) const -> const_reference {
    _m_check_linear_bounds(index);
    return data_[index];
  }

  /**
   * @brief Accesses element in a linear fashion at the specified index.
   * @param index The index to be accessed.
   * @return A reference to the element at the specified index.
   *
   * @note This function does not respect dimensionality of data but performs bounds checking w.r.t. total
   * elements.
   */
  constexpr auto at(size_type index) -> reference {
    _m_check_linear_bounds(index);
    return data_[index];
  }

  /**
   * @brief Accesses element in an n-dimensional space at the specified location.
   * @tparam Args Type of indices (must be integral).
   * @param indices A variable number of indices representing the location of element in an n-dimensional space.
   * @return A const-qualified reference to the element at the specified location.
   *
   * @note This function performs bounds checking if bounds checking is enabled.
   */
  template <Integer... Args>
  [[nodiscard]] constexpr auto operator()(Args... indices) const -> const_reference {
    return data_[_m_linear_index(indices...)];
  }

  /**
   * @brief Accesses element in an n-dimensional space at the specified location.
   * @tparam Args Type of indices (must be integral).
   * @param indices A variable number of indices representing the location of element in an n-dimensional space.
   * @return A reference to the element at the specified location.
   *
   * @note This function performs bounds checking if bounds checking is enabled.
   */
  template <Integer... Args>
  constexpr auto operator()(Args... indices) -> reference {
    return data_[_m_linear_index(indices...)];
  }

  // /////////////////////////////////////////////////////////////
  // Accessors and Mutators
  // /////////////////////////////////////////////////////////////

  /**
   * @brief Tell the state of bounds checking flag.
   * @return State of bounds checking flag.
   */
  [[nodiscard]] constexpr auto is_bounds_checking_enabled() const noexcept -> bool { return bounds_checking_; }

  /**
   * @brief Enables bounds checking.
   */
  constexpr auto enable_bounds_checking() noexcept -> void { bounds_checking_ = true; }

  /**
   * @brief Disables bounds checking.
   */
  constexpr auto disable_bounds_checking() noexcept -> void { bounds_checking_ = false; }

  /**
   * @brief Returns the shape of data.
   * @return A const-qualified reference to shape of data.
   */
  [[nodiscard]] constexpr auto shape() const noexcept -> const Shape & { return shape_; }

  /**
   * @brief Returns the underlying pointer to data in memory.
   * @return A const-qualified pointer to data.
   */
  [[nodiscard]] constexpr auto data() const noexcept -> const_pointer { return data_.data(); }

  /**
   * @brief Returns the underlying pointer to data in memory.
   * @return A pointer to data.
   */
  constexpr auto data() noexcept -> pointer { return data_.data(); }

  /**
   * @brief Returns the underlying container holding the data.
   * @return A const-qualified reference to container.
   */
  [[nodiscard]] constexpr auto underlying_container() const noexcept -> const container & { return data_; }

  /**
   * @brief Returns the total number of elements.
   * @return Total number of elements.
   */
  [[nodiscard]] constexpr auto total() const noexcept -> size_type { return data_.size(); }

  /**
   * @brief Returns the rank of the tensor i.e. the dimensionality of data.
   * @return Rank of shape.
   */
  [[nodiscard]] constexpr auto rank() const noexcept -> size_type { return shape_.rank(); }

  // /////////////////////////////////////////////////////////////
  // Iterators
  // /////////////////////////////////////////////////////////////

  /**
   * @brief Returns an iterator pointing to the beginning of the data in memory.
   * @return A const-qualified iterator to the beginning of the data.
   */
  [[nodiscard]] constexpr auto cbegin() const noexcept -> const_iterator { return data_.cbegin(); }

  /**
   * @brief Returns an iterator pointing to the beginning of the data in memory.
   * @return A const-qualified iterator to the beginning of the data.
   */
  [[nodiscard]] constexpr auto begin() const noexcept -> const_iterator { return data_.begin(); }

  /**
   * @brief Returns an iterator pointing to the beginning of the data in memory.
   * @return An iterator to the beginning of the data.
   */
  constexpr auto begin() noexcept -> iterator { return data_.begin(); }

  /**
   * @brief Returns a reverse iterator pointing to the reverse beginning of the data in memory.
   * @return A const-qualified reverse iterator to the reverse beginning of the data.
   */
  [[nodiscard]] constexpr auto crbegin() const noexcept -> const_reverse_iterator { return data_.crbegin(); }

  /**
   * @brief Returns a reverse iterator pointing to the reverse beginning of the data in memory.
   * @return A const-qualified reverse iterator to the reverse beginning of the data.
   */
  [[nodiscard]] constexpr auto rbegin() const noexcept -> const_reverse_iterator { return data_.rbegin(); }

  /**
   * @brief Returns a reverse iterator pointing to the reverse beginning of the data in memory.
   * @return A reverse iterator to the reverse beginning of the data.
   */
  constexpr auto rbegin() noexcept -> reverse_iterator { return data_.rbegin(); }

  /**
   * @brief Returns an iterator pointing to the ending of the data in memory.
   * @return A const-qualified iterator to the ending of the data.
   */
  [[nodiscard]] constexpr auto cend() const noexcept -> const_iterator { return data_.cend(); }

  /**
   * @brief Returns an iterator pointing to the ending of the data in memory.
   * @return A const-qualified iterator to the ending of the data.
   */
  [[nodiscard]] constexpr auto end() const noexcept -> const_iterator { return data_.end(); }

  /**
   * @brief Returns an iterator pointing to the ending of the data in memory.
   * @return A iterator to the ending of the data.
   */
  constexpr auto end() noexcept -> iterator { return data_.end(); }

  /**
   * @brief Returns a reverse iterator pointing to the reverse ending of the data in memory.
   * @return A const-qualified reverse iterator to the reverse ending of the data.
   */
  [[nodiscard]] constexpr auto crend() const noexcept -> const_reverse_iterator { return data_.crend(); }

  /**
   * @brief Returns a reverse iterator pointing to the reverse ending of the data in memory.
   * @return A const-qualified reverse iterator to the reverse ending of the data.
   */
  [[nodiscard]] constexpr auto rend() const noexcept -> const_reverse_iterator { return data_.rend(); }

  /**
   * @brief Returns a reverse iterator pointing to the reverse ending of the data in memory.
   * @return A reverse iterator to the reverse ending of the data.
   */
  constexpr auto rend() noexcept -> reverse_iterator { return data_.rend(); }

  // /////////////////////////////////////////////////////////////
  // Query Functions
  // /////////////////////////////////////////////////////////////

  /**
   * @brief Checks if the tensor represents a scalar.
   * @return
   * @code
   * this->rank() == SCALAR_RANK
   * @endcode
   */
  [[nodiscard]] constexpr auto is_scalar() const noexcept -> bool { return rank() == SCALAR_RANK; }

  /**
   * @brief Checks if the tensor represents a vector (or array).
   * @return
   * @code
   * this->rank() == VECTOR_RANK
   * @endcode
   */
  [[nodiscard]] constexpr auto is_vector() const noexcept -> bool { return rank() == VECTOR_RANK; }

  /**
   * @brief Checks if the tensor represents a matrix.
   * @return
   * @code
   * this->rank() == MATRIX_RANK
   * @endcode
   */
  [[nodiscard]] constexpr auto is_matrix() const noexcept -> bool { return rank() == MATRIX_RANK; }

  // /////////////////////////////////////////////////////////////
  // Informative
  // /////////////////////////////////////////////////////////////

  /**
   * @brief Returns a string containing meta information.
   * @return A meta information string.
   */
  [[nodiscard]] auto meta_info() const -> std::string {
    return fmt::format("{{ total={}, shape={} }}", total(), shape_.to_string());
  }

  // /////////////////////////////////////////////////////////////
  // Modifiers
  // /////////////////////////////////////////////////////////////

  /**
   * @brief Reshapes the tensor.
   * @param new_shape The new shape of the tensor.
   * @return A reference to self.
   *
   * @note This function throws an exception if the new shape is not equivalent to the current shape.
   */
  constexpr auto reshape(const Shape &new_shape) -> Tensor & {
    _s_check_shape_equivalency(shape_, new_shape);
    shape_ = new_shape;
    return *this;
  }

  /**
   * @brief Reshapes the tensor to have the specified rank.
   * @param new_rank The new rank of the tensor.
   * @return A reference to self.
   *
   * @details
   * Consider the following code snippet:
   *
   * @code
   * // A tensor of shape (3, 4, 5, 7, 8, 9, 3, 2, 1, 1, 1)
   * auto tensor = cbx::Tensor{{3, 4, 5, 7, 8, 9, 3, 2, 1, 1, 1}};
   *
   * // When new rank is lower than the current rank, the shape is cramped.
   * tensor.reshape(4); // New shape becomes (3, 4, 5, 3024)
   *
   * // When new rank is higher than the current rank, the shape is stretched.
   * tensor.reshape(6); // New shape becomes (3, 4, 5, 3024, 1, 1)
   *
   * // When new rank is equal to the current rank, nothing happens.
   * tensor.reshape(6); // Shape remains (3, 4, 5, 3024, 1, 1)
   * @endcode
   */
  auto reshape(size_type new_rank) -> Tensor & {
    auto cur_rank = rank();
    if (cur_rank == new_rank) {
      return *this;
    }
    if (new_rank == SCALAR_RANK) {
      auto new_shape = Shape{};
      _s_check_shape_equivalency(shape_, new_shape);
      shape_ = std::move(new_shape);
    } else if (new_rank < cur_rank) {
      auto new_shape = shape_.clone().resize(new_rank);
      auto back_index = new_rank - 1;
      auto cramped_axis_length =
          std::accumulate(shape_.begin() + back_index, shape_.end(), Shape::SCALAR_SIZE, std::multiplies());
      new_shape.set_axis(back_index, cramped_axis_length);
      shape_ = std::move(new_shape);
    } else {
      shape_.resize(new_rank);
    }
    return *this;
  }

  /**
   * @brief Flattens the data i.e. rank = 1.
   * @return A reference to self.
   */
  constexpr auto flatten() -> Tensor & { return reshape(1); }

  /**
   * @brief Applies the given transformation to all elements.
   * @tparam Func The type of function.
   * @param func Transformation function.
   * @return A reference to self.
   */
  template <typename Func>
  constexpr auto transform(Func func) noexcept -> Tensor & {
    std::transform(begin(), end(), begin(), func);
    return *this;
  }

  /**
   * @brief Applies the given transformation to all elements.
   * @tparam Func The type of function.
   * @param iter An input iterator for parallel iteration with `this->begin()`
   * @param func Transformation function.
   * @return A reference to self.
   */
  template <std::input_iterator I_It, typename Func>
  constexpr auto transform(I_It iter, Func func) noexcept -> Tensor & {
    std::transform(begin(), end(), iter, begin(), func);
    return *this;
  }

  /**
   * @brief Applies the given transformation to all elements.
   * @tparam Func The type of function.
   * @param iter An input iterator for parallel iteration with `this->begin()`
   * @param func Transformation function.
   * @return A reference to self.
   */
  template <std::ranges::range R, typename Func>
  constexpr auto transform(const R &range, Func func) noexcept -> Tensor & {
    std::transform(begin(), end(), range.begin(), begin(), func);
    return *this;
  }

  /**
   * @brief Applies the given transformation to all elements.
   * @tparam Func The type of function.
   * @param func Transformation function.
   * @return A reference to self.
   */
  template <typename Func>
  constexpr auto operator|=(Func func) noexcept -> Tensor & {
    return transform(func);
  }

  /**
   * @brief Returns a transformed tensor.
   * @tparam Func The type of function.
   * @param func Transformation function.
   * @return The transformed tensor.
   */
  template <typename U = value_type, typename Func>
  [[nodiscard]] constexpr auto transformed(Func func) const -> Tensor<U> {
    auto result = zeros_like<U>();
    std::transform(begin(), end(), result.begin(), func);
    return result;
  }

  /**
   * @brief Returns a transformed tensor.
   * @tparam Func The type of function.
   * @param func Transformation function.
   * @return The transformed tensor.
   */
  template <typename U = value_type, std::input_iterator I_It, typename Func>
  [[nodiscard]] constexpr auto transformed(I_It iter, Func func) const -> Tensor<U> {
    auto result = zeros_like<U>();
    std::transform(begin(), end(), iter, result.begin(), func);
    return result;
  }

  /**
   * @brief Returns a transformed tensor.
   * @tparam Func The type of function.
   * @param func Transformation function.
   * @return The transformed tensor.
   */
  template <typename U = value_type, std::ranges::range R, typename Func>
  [[nodiscard]] constexpr auto transformed(R range, Func func) const -> Tensor<U> {
    auto result = zeros_like<U>();
    std::transform(begin(), end(), range.begin(), result.begin(), func);
    return result;
  }

  /**
   * @brief Returns a transformed tensor.
   * @tparam Func The type of function.
   * @param func Transformation function.
   * @return The transformed tensor.
   */
  template <typename Func>
  [[nodiscard]] constexpr auto operator|(Func func) const -> Tensor {
    return transformed(func);
  }

  /**
   * @brief Clamps values outside the interval to its boundaries.
   * @param lower_bound The lower bound of the interval.
   * @param upper_bound The upper bound of the interval.
   * @return A reference to self.
   */
  constexpr auto clamp(value_type lower_bound, value_type upper_bound) noexcept -> Tensor & {
    return transform([lower_bound, upper_bound](auto x) {
      return std::clamp(x, lower_bound, upper_bound);
    });
  }

  /**
   * @brief Clamps values outside the interval to its boundaries and returns the result.
   * @param lower_bound The lower bound of the interval.
   * @param upper_bound The upper bound of the interval.
   * @return The clamped tensor.
   */
  [[nodiscard]] constexpr auto clamped(value_type lower_bound, value_type upper_bound) noexcept -> Tensor {
    return transformed([lower_bound, upper_bound](auto x) {
      return std::clamp(x, lower_bound, upper_bound);
    });
  }

  // /////////////////////////////////////////////////////////////
  // Arithmetic Operators
  // /////////////////////////////////////////////////////////////

  /**
   * @brief Performs an in-place addition operation.
   * @param num The number to be added.
   * @return A reference to self.
   */
  constexpr auto operator+=(Number auto num) noexcept -> Tensor & {
    return transform([num](auto x) {
      return x + num;
    });
  }

  /**
   * @brief Performs an in-place subtraction operation.
   * @param num The number to be subtracted.
   * @return A reference to self.
   */
  constexpr auto operator-=(Number auto num) noexcept -> Tensor & {
    return transform([num](auto x) {
      return x - num;
    });
  }

  /**
   * @brief Performs an in-place multiplication operation.
   * @param num The number to be multiplied.
   * @return A reference to self.
   */
  constexpr auto operator*=(Number auto num) noexcept -> Tensor & {
    return transform([num](auto x) {
      return x * num;
    });
  }

  /**
   * @brief Performs an in-place division operation.
   * @param num The number by which the tensor is to be divided.
   * @return A reference to self.
   */
  constexpr auto operator/=(Number auto num) noexcept -> Tensor & {
    return transform([num](auto x) {
      return x / num;
    });
  }

  /**
   * @brief Performs an in-place modulo (or remainder) operation.
   * @param num The operand on the right hand side of modulo operator.
   * @return A reference to self.
   */
  constexpr auto operator%=(Number auto num) noexcept -> Tensor & {
    return transform([num](auto x) {
      return x % num;
    });
  }

  /**
   * @brief Adds the given tensors and returns the result.
   * @tparam U Type of tensor on the left-hand side.
   * @param rhs Tensor on the right-hand side.
   * @return A reference to self.
   */
  template <typename U>
  constexpr auto operator+=(const Tensor<U> &rhs) -> Tensor & {
    _s_check_broadcastability(shape_, rhs.shape());
    return transform(CyclicIterator{rhs.begin(), rhs.end()}, std::plus{});
  }

  /**
   * @brief Subtracts the given tensors and returns the result.
   * @tparam U Type of tensor on the left-hand side.
   * @param rhs Tensor on the right-hand side.
   * @return A reference to self.
   */
  template <typename U>
  constexpr auto operator-=(const Tensor<U> &rhs) -> Tensor & {
    _s_check_broadcastability(shape_, rhs.shape());
    return transform(CyclicIterator{rhs.begin(), rhs.end()}, std::minus{});
  }

  /**
   * @brief Multiplies the given tensors and returns the result.
   * @tparam U Type of tensor on the left-hand side.
   * @param rhs Tensor on the right-hand side.
   * @return A reference to self.
   */
  template <typename U>
  constexpr auto operator*=(const Tensor<U> &rhs) -> Tensor & {
    _s_check_broadcastability(shape_, rhs.shape());
    return transform(CyclicIterator{rhs.begin(), rhs.end()}, std::multiplies{});
  }

  /**
   * @brief Divides the given tensors and returns the result.
   * @tparam U Type of tensor on the left-hand side.
   * @param rhs Tensor on the right-hand side.
   * @return A reference to self.
   */
  template <typename U>
  constexpr auto operator/=(const Tensor<U> &rhs) -> Tensor & {
    _s_check_broadcastability(shape_, rhs.shape());
    return transform(CyclicIterator{rhs.begin(), rhs.end()}, std::divides{});
  }

  /**
   * @brief Applies Modulus on the given tensors and returns the result.
   * @tparam U Type of tensor on the left-hand side.
   * @param rhs Tensor on the right-hand side.
   * @return A reference to self.
   */
  template <Number U>
  constexpr auto operator%=(const Tensor<U> &rhs) -> Tensor & {
    _s_check_broadcastability(shape_, rhs.shape());
    return transform(CyclicIterator{rhs.begin(), rhs.end()}, [](auto x, auto y) {
      return std::fmod(x, y);
    });
  }

  // /////////////////////////////////////////////////////////////
  // Utility
  // /////////////////////////////////////////////////////////////

  /**
   * @brief Swaps this tensor with the other.
   * @param other The tensor with which the swap will be performed.
   * @return A reference to self.
   */
  constexpr auto swap(Tensor &other) noexcept -> Tensor & {
    std::swap(bounds_checking_, other.bounds_checking_);
    shape_.swap(other.shape_);
    data_.swap(other.data_);
    return *this;
  }

  /**
   * @brief Clones the tensor.
   * @return A clone of the original tensor.
   */
  [[nodiscard]] constexpr auto clone() const -> Tensor { return *this; }

  /**
   * @brief Returns a zero-initialized tensor of the same shape.
   * @tparam U Type of tensor
   * @return Tensor of similar shape
   */
  template <typename U = value_type>
  [[nodiscard]] constexpr auto zeros_like() const -> Tensor<U> {
    return Tensor<U>{shape_};
  }

  // /////////////////////////////////////////////////////////////////////////////////////////////
  // Static Functions
  // /////////////////////////////////////////////////////////////////////////////////////////////

  // /////////////////////////////////////////////////////////////
  // Factory Functions
  // /////////////////////////////////////////////////////////////

  /**
   * @brief Creates a randomly filled tensor of the specified shape.
   * @param shape The shape of the tensor.
   * @param seed The seed of randomness.
   * @param lower_bound The lower bound of the random values.
   * @param upper_bound The upper bound of the random values.
   * @return A random tensor of the specified shape.
   */
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

  /**
   * @brief Creates a custom filled tensor of the specified shape.
   * @tparam Func The Type of function.
   * @param shape The shape of the tensor.
   * @param func The function for filling tensor.
   * @return A custom tensor of the specified shape.
   *
   * @code
   * // The following tensor contains the sequence: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
   * auto tensor = cbx::Tensor<cbx::f32>::custom({3, 4}, [n = 0]() mutable {
   *    return n += 1;
   * });
   * @endcode
   */
  template <typename Func>
  [[nodiscard]] static auto custom(const Shape &shape, Func func) -> Tensor {
    auto tensor = Tensor{shape};
    std::generate(tensor.begin(), tensor.end(), func);
    return tensor;
  }

  // /////////////////////////////////////////////////////////////////////////////////////////////
  // Friend Functions
  // /////////////////////////////////////////////////////////////////////////////////////////////

  // /////////////////////////////////////////////////////////////
  // Arithmetic Operators
  // /////////////////////////////////////////////////////////////

  /**
   * @brief Adds the given tensors and returns the result.
   * @tparam U Type of tensor on the left-hand side.
   * @param lhs Tensor on the left-hand side.
   * @param rhs Tensor on the right-hand side.
   * @return Result of addition as a tensor.
   */
  template <typename U>
  friend constexpr auto operator+(const Tensor &lhs, const Tensor<U> &rhs) -> Tensor {
    if (lhs.rank() > rhs.rank()) {
      _s_check_broadcastability(lhs.shape_, rhs.shape_);
      return lhs.transformed(CyclicIterator{rhs.begin(), rhs.end()}, std::plus{});
    } else {
      _s_check_broadcastability(rhs.shape_, lhs.shape_);
      return rhs.transformed(CyclicIterator{lhs.begin(), lhs.end()}, std::plus{});
    }
  }

  /**
   * @brief Subtracts the given tensors and returns the result.
   * @tparam U Type of tensor on the left-hand side.
   * @param lhs Tensor on the left-hand side.
   * @param rhs Tensor on the right-hand side.
   * @return Result of subtraction as a tensor.
   */
  template <typename U>
  friend constexpr auto operator-(const Tensor &lhs, const Tensor<U> &rhs) -> Tensor {
    if (lhs.rank() > rhs.rank()) {
      _s_check_broadcastability(lhs.shape_, rhs.shape_);
      return lhs.transformed(CyclicIterator{rhs.begin(), rhs.end()}, std::minus{});
    } else {
      _s_check_broadcastability(rhs.shape_, lhs.shape_);
      return rhs.transformed(CyclicIterator{lhs.begin(), lhs.end()}, [](auto y, auto x) {
        return x - y;
      });
    }
  }

  /**
   * @brief Multiplies the given tensors and returns the result.
   * @tparam U Type of tensor on the left-hand side.
   * @param lhs Tensor on the left-hand side.
   * @param rhs Tensor on the right-hand side.
   * @return Result of multiplication as a tensor.
   */
  template <typename U>
  friend constexpr auto operator*(const Tensor &lhs, const Tensor<U> &rhs) -> Tensor {
    if (lhs.rank() > rhs.rank()) {
      _s_check_broadcastability(lhs.shape_, rhs.shape_);
      return lhs.transformed(CyclicIterator{rhs.begin(), rhs.end()}, std::multiplies{});
    } else {
      _s_check_broadcastability(rhs.shape_, lhs.shape_);
      return rhs.transformed(CyclicIterator{lhs.begin(), lhs.end()}, std::multiplies{});
    }
  }

  /**
   * @brief Divides the given tensors and returns the result.
   * @tparam U Type of tensor on the left-hand side.
   * @param lhs Tensor on the left-hand side.
   * @param rhs Tensor on the right-hand side.
   * @return Result of division as a tensor.
   */
  template <typename U>
  friend constexpr auto operator/(const Tensor &lhs, const Tensor<U> &rhs) -> Tensor {
    if (lhs.rank() > rhs.rank()) {
      _s_check_broadcastability(lhs.shape_, rhs.shape_);
      return lhs.transformed(CyclicIterator{rhs.begin(), rhs.end()}, std::divides{});
    } else {
      _s_check_broadcastability(rhs.shape_, lhs.shape_);
      return rhs.transformed(CyclicIterator{lhs.begin(), lhs.end()}, [](auto y, auto x) {
        return x / y;
      });
    }
  }

  /**
   * @brief Performs modulo (or remainder) operation and returns the result.
   * @tparam U Type of tensor on the left-hand side.
   * @param lhs Tensor on the left-hand side.
   * @param rhs Tensor on the right-hand side.
   * @return Result of remainder operation as a tensor.
   */
  template <typename U>
  friend constexpr auto operator%(const Tensor &lhs, const Tensor<U> &rhs) -> Tensor {
    if (lhs.rank() > rhs.rank()) {
      _s_check_broadcastability(lhs.shape_, rhs.shape_);
      return lhs.transformed(CyclicIterator{rhs.begin(), rhs.end()}, [](auto x, auto y) {
        return std::fmod(x, y);
      });
    } else {
      _s_check_broadcastability(rhs.shape_, lhs.shape_);
      return rhs.transformed(CyclicIterator{lhs.begin(), lhs.end()}, [](auto y, auto x) {
        return std::fmod(x, y);
      });
    }
  }
};

// /////////////////////////////////////////////////////////////////////////////////////////////
// External Functions
// /////////////////////////////////////////////////////////////////////////////////////////////

// /////////////////////////////////////////////////////////////
// Arithmetic Operators
// /////////////////////////////////////////////////////////////

// /////////////////////////////////////////////////////////////
// Unary Operators
// /////////////////////////////////////////////////////////////

/**
 * @brief Unary plus operator.
 * @return Result tensor.
 */
template <typename T>
constexpr auto operator+(const Tensor<T> &tensor) noexcept -> Tensor<T> {
  return tensor;
}

/**
 * @brief Unary minus operator - Negates all the elements of a tensor.
 * @return Negated tensor.
 */
template <typename T>
constexpr auto operator-(const Tensor<T> &tensor) noexcept -> Tensor<T> {
  return tensor | std::negate{};
}

// /////////////////////////////////////////////////////////////
// Binary Operators
// /////////////////////////////////////////////////////////////

/**
 * @brief Adds the given scalar to the tensor and returns the result.
 * @tparam T Type of tensor.
 * @param tensor Source tensor.
 * @param num The number to be added.
 * @return Result of addition as a tensor.
 */
template <typename T>
constexpr auto operator+(const Tensor<T> &tensor, Number auto num) -> Tensor<T> {
  return tensor | [num](auto x) {
    return x + num;
  };
}

/**
 * @brief Adds the given scalar to the tensor and returns the result.
 * @tparam T Type of tensor.
 * @param num The number to be added.
 * @param tensor Source tensor.
 * @return Result of addition as a tensor.
 */
template <typename T>
constexpr auto operator+(Number auto num, const Tensor<T> &tensor) -> Tensor<T> {
  return tensor + num;
}

/**
 * @brief Subtracts the given scalar from the tensor and returns the result.
 * @tparam T Type of tensor.
 * @param tensor Source tensor.
 * @param num The number to be subtracted.
 * @return Result of subtraction as a tensor.
 */
template <typename T>
constexpr auto operator-(const Tensor<T> &tensor, Number auto num) -> Tensor<T> {
  return tensor | [num](auto x) {
    return x - num;
  };
}

/**
 * @brief Subtracts the given scalar from the tensor and returns the result.
 * @tparam T Type of tensor.
 * @param num The number to be subtracted.
 * @param tensor Source tensor.
 * @return Result of subtraction as a tensor.
 */
template <typename T>
constexpr auto operator-(Number auto num, const Tensor<T> &tensor) -> Tensor<T> {
  return tensor | [num](auto x) {
    return num - x;
  };
}

/**
 * @brief Multiplies the given scalar with the tensor and returns the result.
 * @tparam T Type of tensor.
 * @param tensor Source tensor.
 * @param num The number to be multiplied.
 * @return Result of multiplication as a tensor.
 */
template <typename T>
constexpr auto operator*(const Tensor<T> &tensor, Number auto num) -> Tensor<T> {
  return tensor | [num](auto x) {
    return x * num;
  };
}

/**
 * @brief Multiplies the given scalar with the tensor and returns the result.
 * @tparam T Type of tensor.
 * @param num The number to be multiplied.
 * @param tensor Source tensor.
 * @return Result of multiplication as a tensor.
 */
template <typename T>
constexpr auto operator*(Number auto num, const Tensor<T> &tensor) -> Tensor<T> {
  return tensor * num;
}

/**
 * @brief Divides the tensor with the given scalar and returns the result.
 * @tparam T Type of tensor.
 * @param tensor Source tensor.
 * @param num The number by which the tensor is to be divided.
 * @return Result of division as a tensor.
 */
template <typename T>
constexpr auto operator/(const Tensor<T> &tensor, Number auto num) -> Tensor<T> {
  return tensor | [num](auto x) {
    return x / num;
  };
}

/**
 * @brief Divides the tensor with the given scalar and returns the result.
 * @tparam T Type of tensor.
 * @param num The number by which the tensor is to be divided.
 * @param tensor Source tensor.
 * @return Result of division as a tensor.
 */
template <typename T>
constexpr auto operator/(Number auto num, const Tensor<T> &tensor) -> Tensor<T> {
  return tensor | [num](auto x) {
    return num / x;
  };
}

/**
 * @brief Performs modulo (or remainder) operation and returns the result.
 * @tparam T Type of tensor.
 * @param tensor Source tensor.
 * @param num The operand on the right hand side of modulo operator.
 * @return Result of remainder operation as a tensor.
 */
template <typename T>
constexpr auto operator%(const Tensor<T> &tensor, Number auto num) -> Tensor<T> {
  return tensor | [num](auto x) {
    return x % num;
  };
}

/**
 * @brief Performs modulo (or remainder) operation and returns the result.
 * @tparam T Type of tensor.
 * @param num The operand on the right hand side of modulo operator.
 * @param tensor Source tensor.
 * @return Result of remainder operation as a tensor.
 */
template <typename T>
constexpr auto operator%(Number auto num, const Tensor<T> &tensor) -> Tensor<T> {
  return tensor | [num](auto x) {
    return num % x;
  };
}

}

#endif
