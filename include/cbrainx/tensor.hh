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
#include <string>
#include <thread>
#include <utility>

#include <fmt/format.h>

#include "custom_iterators.hh"
#include "exceptions.hh"
#include "shape.hh"
#include "type_aliases.hh"
#include "type_concepts.hh"

namespace cbx {

/// \brief The `Tensor` class represents an n-dimensional array.
/// \tparam T Data type of the tensor (must be arithmetic).
///
/// \details
/// A tensor is a generalization of vectors and matrices to arbitrary ranks, more commonly known as a
/// multidimensional array. In practice, it is a container that can harbor uniform numeric data in an
/// N-dimensional space. The number of indices necessary to obtain individual tensor elements is its rank. For
/// example, a matrix is a tensor of rank two since it requires two indices to denote row and column. Axes are
/// the components that make up the rank, and dimension refers to the number of elements in a given axis. In
/// many instances, axis and dimensions are interchangeable, but there is a subtlety between them. In simple
/// terms, the axis represents the dimensions of data. A shape is an ordered container whose length is its rank,
/// and elements represent the dimensions of each axis. The discrete units of datum in a tensor are called its
/// scalar components or simply components or elements.
///
/// \see Shape
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

  // /////////////////////////////////////////////
  // Constants
  // /////////////////////////////////////////////

  /// \brief Rank of a scalar.
  static constexpr size_type SCALAR_RANK = 0;

  /// \brief Rank of a vector (or array).
  static constexpr size_type VECTOR_RANK = 1;

  /// \brief Rank of a matrix.
  static constexpr size_type MATRIX_RANK = 2;

 private:
  /// \brief A flag for enabling or disabling bounds checking.
  ///
  /// \note This attribute is mutable and does not account for the constness of the tensor.
  mutable bool bounds_checking_ = true;

  /// \brief Shape of data.
  Shape shape_ = {};

  /// \brief Actual data.
  container data_ = container(shape_.total());

  // /////////////////////////////////////////////
  // Helpers
  // /////////////////////////////////////////////

  /// \brief Checks if two shapes are equivalent.
  /// \param[in] a, b The shapes to be compared for equivalency.
  ///
  /// \details
  /// This function throws an exception if \p a and \p b are not equivalent.
  ///
  /// \throws ShapeError
  static constexpr auto _s_check_shape_equivalency(const Shape &a, const Shape &b) -> void {
    if (not a.is_equivalent(b)) {
      throw ShapeError{"cbx::Tensor::_s_check_shape_equivalency: a = {} [total = {}] is not equivalent to b = "
                       "{} [total = {}]",
                       a.to_string(), a.total(), b.to_string(), b.total()};
    }
  }

  /// \brief Checks if two shapes are equal.
  /// \param[in] a, b The shapes to be compared for equality.
  ///
  /// \details
  /// This function throws an exception if \p a and \p b are unequal.
  ///
  /// \throws ShapeError
  static constexpr auto _s_check_shape_equality(const Shape &a, const Shape &b) -> void {
    if (a != b) {
      throw ShapeError{"cbx::Tensor::_s_check_shape_equality: a = {} must be equal to b = {}", a.to_string(),
                       b.to_string()};
    }
  }

  /// \brief Checks if \p other is broadcastable to `this->shape()`.
  /// \param[in] other The shape to be tested for broadcastability.
  ///
  /// \details
  /// This functions throws an exception if \p other is not broadcastable to `this->shape()`.
  ///
  /// \throws ShapeError
  constexpr auto _m_check_broadcastability(const Shape &other) -> void {
    if (other.rank() > rank() or not std::equal(other.rbegin(), other.rend(), shape_.rbegin())) {
      throw ShapeError{
          "cbx::Tensor::_m_check_broadcastability: other = {} is not broadcastable to this->shape() = {}",
          other.to_string(), shape_.to_string()};
    }
  }

  /// \brief Checks if two shapes are compatible for broadcasting.
  /// \param[in] a, b The shapes to be tested for broadcastability.
  ///
  /// \details
  /// This functions throws an exception if \p a and \p b are not compatible for broadcasting.
  ///
  /// \throws ShapeError
  static constexpr auto _s_check_broadcastability(const Shape &a, const Shape &b) -> void {
    auto is_compatible = a.rank() > b.rank() ? std::equal(b.rbegin(), b.rend(), a.rbegin())
                                             : std::equal(a.rbegin(), a.rend(), b.rbegin());
    if (not is_compatible) {
      throw ShapeError{
          "cbx::Tensor::_s_check_broadcastability: a = {} and b = {} are not compatible for broadcasting",
          a.to_string(), b.to_string()};
    }
  }

  /// \brief Returns the broadcast shape if \p a and \p b are broadcastable.
  /// \param[in] a, b The shapes to be broadcasted.
  /// \return Broadcast shape.
  ///
  /// \details
  /// This functions throws an exception if \p a and \p b are not compatible for broadcasting.
  ///
  /// \throws ShapeError
  static auto _s_get_broadcast_shape(const Shape &a, const Shape &b) -> Shape {
    _s_check_broadcastability(a, b);
    return a.rank() > b.rank() ? a : b;
  }

  /// \brief Performs bounds checking w.r.t. the total number of elements.
  /// \param[in] index The index of the element.
  ///
  /// \details
  /// This function throws an exception if \p index is out of bounds.
  ///
  /// \throws IndexOutOfBounds
  constexpr auto _m_check_linear_bounds(size_type index) const -> void {
    auto total_elements = total();
    if (index >= total_elements) {
      throw IndexOutOfBoundsError{"cbx::Tensor::_m_check_linear_bounds: index = {} >= this->total() = {}",
                                  index, total_elements};
    }
  }

  /// \brief Checks if the number of indices conforms with the rank.
  /// \tparam Args Data type of the indices (must be integral).
  /// \param[in] indices Co-ordinates of the element in an n-dimensional space.
  ///
  /// \details
  /// This function throws an exception if the number of indices contradicts the rank of the tensor.
  ///
  /// \throws RankError
  template <Integer... Args>
  constexpr auto _m_check_rank(Args... indices) const -> void {
    auto indices_count = sizeof...(indices);
    auto cur_rank = rank();
    if (indices_count != cur_rank) {
      throw RankError{
          "cbx::Tensor::_m_check_rank: indices [count = {}] are in contradiction with the rank = {}",
          indices_count, cur_rank};
    }
  }

  /// \brief Performs bounds checking for the given element in an n-dimensional space.
  /// \tparam Args Data type of the indices (must be integral).
  /// \param[in] indices Co-ordinates of the element in an n-dimensional space.
  ///
  /// \details
  /// This function throws an exception if:
  ///     * Any index is out of range w.r.t to its axis, provided that bounds checking is enabled.
  ///     * The number of indices contradicts the rank.
  ///
  /// \throws RankError
  /// \throws IndexOutOfBoundsError
  template <Integer... Args>
  auto _m_check_axes_bounds(Args... indices) const -> void {
    _m_check_rank(indices...);
    if (not bounds_checking_) {
      return;
    }
    auto il_indices = std::initializer_list<usize>{usize(indices)...};
    for (auto shape_it = shape_.begin(); auto axis_index : il_indices) {
      if (axis_index >= (*shape_it)) {
        throw IndexOutOfBoundsError{
            "cbx::Tensor::_m_check_axes_bounds: axis_index = {} >= this->shape() [axis = {}] = {}", axis_index,
            std::distance(shape_.begin(), shape_it), *shape_it};
      }
      ++shape_it;
    }
  }

  /// \brief Calculates the linear index of the given indices.
  /// \tparam Args Data type of the indices (must be integral).
  /// \param[in] indices Co-ordinates of the element in an n-dimensional space.
  ///
  /// \details
  /// This function throws an exception if:
  ///     * Any index is out of range w.r.t to its axis, provided that bounds checking is enabled.
  ///     * The number of indices contradicts the rank.
  ///
  /// \throws RankError
  /// \throws IndexOutOfBoundsError
  template <Integer... Args>
  [[nodiscard]] auto _m_linear_index(Args... indices) const -> size_type {
    _m_check_axes_bounds(indices...);

    // Convert the parameter pack to an initializer list and use strides to get the linear index. A stride is a
    // span between two successive elements along a particular axis.
    auto il_indices = std::initializer_list<usize>{usize(indices)...};
    size_type linear_index = {};
    auto stride = Shape::SCALAR_SIZE;
    auto shape_r_it = shape_.rbegin();
    // Reverse iteration evades repetition when computing strides.
    for (auto indices_r_it = std::rbegin(il_indices), indices_r_end = std::rend(il_indices);
         indices_r_it != indices_r_end; ++indices_r_it) {
      linear_index += (*indices_r_it) * stride;
      stride *= (*shape_r_it);
      ++shape_r_it;
    }
    return linear_index;
  }

  /// \brief Checks if the given rank represents a matrix.
  /// \param[in] rank The rank to be checked.
  ///
  /// \details
  /// This function throws an exception if the given rank does not represent a matrix.
  ///
  /// \throws RankError
  static auto _s_matrix_rank_check(size_type rank) -> void {
    if (rank != MATRIX_RANK) {
      throw RankError{"cbx::Tensor::_s_matrix_rank_check: rank = {} does not represent a matrix", rank};
    }
  }

  /// \brief Checks if two matrices are compatible for multiplication.
  /// \param[in] c1 Columns of the first matrix.
  /// \param[in] r2 Rows of the second matrix.
  ///
  /// \details
  /// This function throws an exception if the two matrices are not compatible for multiplication.
  ///
  /// \throws ShapeError
  static auto _s_matmul_compatibility_check(size_type c1, size_type r2) -> void {
    if (c1 != r2) {
      throw ShapeError{"cbx::Tensor::_s_matmul_compatibility_check: shapes are not compatible for matrix "
                       "multiplication [c1 = {}, r2 = {}]",
                       c1, r2};
    }
  }

 public:
  // /////////////////////////////////////////////
  // Constructors and Destructors
  // /////////////////////////////////////////////

  /// \brief Default constructor.
  ///
  /// \details This constructor creates a scalar.
  constexpr Tensor() = default;

  /// \brief Copy constructor.
  /// \param[in] other Source tensor.
  constexpr Tensor(const Tensor &other) = default;

  /// \brief Move constructor.
  /// \param[in] other Source tensor.
  constexpr Tensor(Tensor &&other) noexcept
      : bounds_checking_{other.bounds_checking_}, shape_{std::move(other.shape_)},
        data_(std::move(other.data_)) {}

  /// \brief Constructs a tensor of the specified shape with the initial value \p value for all its elements.
  /// \param[in] shape The shape of the tensor.
  /// \param[in] value The initializing value for all the elements.
  explicit Tensor(const Shape &shape, value_type value = {}) : shape_{shape}, data_(shape.total(), value) {}

  /// \brief Constructs a tensor of the specified shape with the contents of the range [\p first, `last`).
  /// \param[in] shape The shape of the tensor.
  /// \param[in] first The beginning of the range to copy the data from.
  ///
  /// \note The ending of the range, i.e., `last`, will be calculated from \p shape.
  template <std::input_iterator I_It>
  Tensor(const Shape &shape, I_It first) : shape_{shape}, data_{first, first + shape.total()} {}

  /// \brief Constructs a tensor of the specified shape with the contents of \p range.
  /// \param[in] shape The shape of the tensor.
  /// \param[in] range The range to copy the data from.
  Tensor(const Shape &shape, const std::ranges::range auto &range)
      : shape_{shape}, data_{range.begin(), range.begin() + shape.total()} {}

  /// \brief Default destructor.
  constexpr ~Tensor() = default;

  // /////////////////////////////////////////////
  // Assignment Operators
  // /////////////////////////////////////////////

  /// \brief Default copy assignment operator.
  /// \param[in] other Source tensor.
  /// \return A reference to self.
  constexpr auto operator=(const Tensor &other) -> Tensor & = default;

  /// \brief Move assignment operator.
  /// \param[in] other Source tensor.
  /// \return A reference to self.
  constexpr auto operator=(Tensor &&other) noexcept -> Tensor & {
    bounds_checking_ = std::exchange(other.bounds_checking_, true);
    shape_ = std::move(other.shape_);
    data_ = std::move(other.data_);
    return *this;
  }

  // /////////////////////////////////////////////
  // Element Access
  // /////////////////////////////////////////////

  /// \brief Accesses the element at the specified index linearly.
  /// \param[in] index The index of the element.
  /// \return An immutable reference to the element at the specified index.
  ///
  /// \note This function neither respects dimensionality nor performs bounds checking.
  [[nodiscard]] constexpr auto operator[](size_type index) const noexcept -> const_reference {
    return data_[index];
  }

  /// \brief Accesses the element at the specified index linearly.
  /// \param[in] index The index of the element.
  /// \return A mutable reference to the element at the specified index.
  ///
  /// \note This function neither respects dimensionality nor performs bounds checking.
  constexpr auto operator[](size_type index) noexcept -> reference { return data_[index]; }

  /// \brief Accesses the element at the specified index linearly.
  /// \param[in] index The index of the element.
  /// \return An immutable reference to the element at the specified index.
  ///
  /// \note This function does not respect dimensionality but performs bounds checking w.r.t. the total number
  /// of elements.
  ///
  /// \throws IndexOutOfBounds
  [[nodiscard]] constexpr auto at(size_type index) const -> const_reference {
    _m_check_linear_bounds(index);
    return data_[index];
  }

  /// \brief Accesses the element at the specified index linearly.
  /// \param[in] index The index of the element.
  /// \return A mutable reference to the element at the specified index.
  ///
  /// \note This function does not respect dimensionality but performs bounds checking w.r.t. the total number
  /// of elements.
  ///
  /// \throws IndexOutOfBounds
  constexpr auto at(size_type index) -> reference {
    _m_check_linear_bounds(index);
    return data_[index];
  }

  /// \brief Accesses the element at the specified coordinates in an n-dimensional space.
  /// \tparam Args Data type of the indices (must be integral).
  /// \param[in] indices Coordinates of the element in an n-dimensional space.
  /// \return An immutable reference to the element at the specified coordinates.
  ///
  /// \note This function performs bounds checking if it is enabled.
  ///
  /// \throws RankError
  /// \throws IndexOutOfBoundsError
  template <Integer... Args>
  [[nodiscard]] constexpr auto operator()(Args... indices) const -> const_reference {
    return data_[_m_linear_index(indices...)];
  }

  /// \brief Accesses the element at the specified coordinates in an n-dimensional space.
  /// \tparam Args Data type of the indices (must be integral).
  /// \param[in] indices Coordinates of the element in an n-dimensional space.
  /// \return A mutable reference to the element at the specified coordinates.
  ///
  /// \note This function performs bounds checking if it is enabled.
  ///
  /// \throws RankError
  /// \throws IndexOutOfBoundsError
  template <Integer... Args>
  constexpr auto operator()(Args... indices) -> reference {
    return data_[_m_linear_index(indices...)];
  }

  // /////////////////////////////////////////////
  // Accessors and Mutators
  // /////////////////////////////////////////////

  /// \brief Returns whether bounds checking is enabled or not.
  /// \return True if bounds checking is enabled.
  [[nodiscard]] constexpr auto is_bounds_checking_enabled() const noexcept -> bool { return bounds_checking_; }

  /// \brief Enables bounds checking.
  constexpr auto enable_bounds_checking() const noexcept -> void { bounds_checking_ = true; }

  /// \brief Disables bounds checking.
  constexpr auto disable_bounds_checking() const noexcept -> void { bounds_checking_ = false; }

  /// \brief Returns the shape of the tensor.
  /// \return An immutable reference to the shape of the tensor.
  [[nodiscard]] constexpr auto shape() const noexcept -> const Shape & { return shape_; }

  /// \brief Returns the underlying pointer to actual data in the memory.
  /// \return An immutable pointer to the actual data.
  [[nodiscard]] constexpr auto data() const noexcept -> const_pointer { return data_.data(); }

  /// \brief Returns the underlying pointer to actual data in the memory.
  /// \return A mutable pointer to the actual data.
  constexpr auto data() noexcept -> pointer { return data_.data(); }

  /// \brief Returns the underlying container holding the actual data.
  /// \return A immutable reference to the underlying container.
  [[nodiscard]] constexpr auto underlying_container() const noexcept -> const container & { return data_; }

  /// \brief Returns the total number of elements in the tensor.
  /// \return The total number of elements.
  [[nodiscard]] constexpr auto total() const noexcept -> size_type { return data_.size(); }

  /// \brief Returns the rank of the tensor.
  /// \return Rank of the tensor.
  [[nodiscard]] constexpr auto rank() const noexcept -> size_type { return shape_.rank(); }

  // /////////////////////////////////////////////
  // Iterators
  // /////////////////////////////////////////////

  /// \brief Returns an immutable random access iterator pointing to the first element of the underlying
  /// container.
  /// \return An immutable iterator pointing to the beginning of the container.
  [[nodiscard]] constexpr auto cbegin() const noexcept -> const_iterator { return data_.cbegin(); }

  /// \brief Returns a random access iterator pointing to the first element of the underlying container.
  /// \return
  [[nodiscard]] constexpr auto begin() const noexcept -> const_iterator { return data_.begin(); }

  /// \brief Returns a random access iterator pointing to the first element of the underlying container.
  /// \return A mutable iterator pointing to the beginning of the container.
  constexpr auto begin() noexcept -> iterator { return data_.begin(); }

  /// \brief Returns an immutable reverse random access iterator pointing to the last element of the underlying
  /// container.
  /// \return An immutable reverse iterator pointing to the reverse beginning of the container.
  [[nodiscard]] constexpr auto crbegin() const noexcept -> const_reverse_iterator { return data_.crbegin(); }

  /// \brief Returns a reverse random access iterator pointing to the last element of the underlying container.
  /// \return An immutable reverse iterator pointing to the reverse beginning of the container.
  [[nodiscard]] constexpr auto rbegin() const noexcept -> const_reverse_iterator { return data_.rbegin(); }

  /// \brief Returns a reverse random access iterator pointing to the last element of the underlying container.
  /// \return A mutable reverse iterator pointing to the reverse beginning of the container.
  constexpr auto rbegin() noexcept -> reverse_iterator { return data_.rbegin(); }

  /// \brief Returns an immutable random access iterator pointing to the last element of the underlying
  /// container.
  /// \return An immutable iterator pointing to the ending of the container.
  [[nodiscard]] constexpr auto cend() const noexcept -> const_iterator { return data_.cend(); }

  /// \brief Returns a random access iterator pointing to the last element of the underlying container.
  /// \return An immutable iterator pointing to the ending of the container.
  [[nodiscard]] constexpr auto end() const noexcept -> const_iterator { return data_.end(); }

  /// \brief Returns a random access iterator pointing to the last element of the underlying container.
  /// \return A mutable iterator pointing to the ending of the container.
  constexpr auto end() noexcept -> iterator { return data_.end(); }

  /// \brief Returns an immutable reverse random access iterator pointing to the first element of the underlying
  /// container.
  /// \return An immutable reverse iterator pointing to the reverse ending of the container.
  [[nodiscard]] constexpr auto crend() const noexcept -> const_reverse_iterator { return data_.crend(); }

  /// \brief Returns a reverse random access iterator pointing to the first element of the underlying
  /// container.
  /// \return An immutable reverse iterator pointing to the reverse ending of the container.
  [[nodiscard]] constexpr auto rend() const noexcept -> const_reverse_iterator { return data_.rend(); }

  /// \brief Returns a reverse random access iterator pointing to the first element of the underlying
  /// container.
  /// \return A mutable reverse iterator pointing to the reverse ending of the container.
  constexpr auto rend() noexcept -> reverse_iterator { return data_.rend(); }

  // /////////////////////////////////////////////
  // Query Functions
  // /////////////////////////////////////////////

  /// \brief Returns whether the tensor represents a scalar or not.
  /// \return True if the tensor represents a scalar.
  [[nodiscard]] constexpr auto is_scalar() const noexcept -> bool { return rank() == SCALAR_RANK; }

  /// \brief Returns whether the tensor represents a vector or not.
  /// \return True if the tensor represents a vector.
  [[nodiscard]] constexpr auto is_vector() const noexcept -> bool { return rank() == VECTOR_RANK; }

  /// \brief Returns whether the tensor represents a matrix or not.
  /// \return True if the tensor represents a matrix.
  [[nodiscard]] constexpr auto is_matrix() const noexcept -> bool { return rank() == MATRIX_RANK; }

  // /////////////////////////////////////////////
  // Informative
  // /////////////////////////////////////////////

  /// \brief Returns meta-information about the tensor as a string.
  /// \return A string containing meta-information about the tensor.
  [[nodiscard]] auto meta_info() const -> std::string {
    return fmt::format("{{ total={}, shape={}, type={} }}", total(), shape_.to_string(),
                       typeid(value_type).name());
  }

  // /////////////////////////////////////////////
  // Modifiers
  // /////////////////////////////////////////////

  /// \brief Reshapes the tensor.
  /// \param[in] new_shape The new shape of the tensor.
  /// \return A reference to self.
  ///
  /// \details This function throws an exception if the new shape is not equivalent to `this->shape()`.
  ///
  /// \throws ShapeError
  constexpr auto reshape(const Shape &new_shape) -> Tensor & {
    _s_check_shape_equivalency(shape_, new_shape);
    shape_ = new_shape;
    return *this;
  }

  /// \brief Reshapes the tensor.
  /// \param[in] new_rank The new rank of the tensor.
  /// \param[in] modify_front If true, the front of the shape will be modified to conform with the new rank.
  /// \return A reference to self.
  ///
  /// \details
  /// Here is a code snippet to better exhibit the use of this method.
  ///
  /// ```cpp
  /// // Here is a rank-4 tensor of shape (2, 4, 5, 7).
  /// auto tensor = cbx::Tensor{{2, 4, 5, 7}};
  /// // Reshape it to a rank-2 tensor. Hence, the new shape will become (2, 140).
  /// tensor.reshape(2);
  /// // Again, reshape it to a rank-5 tensor but from the front. Hence, the new shape will become
  /// // (1, 1, 1, 2, 140).
  /// tensor.reshape(5, true);
  /// ```
  ///
  /// This function throws an exception if an equivalent shape conforming with the new rank is unattainable.
  ///
  /// \throws ShapeError
  auto reshape(size_type new_rank, bool modify_front = false) -> Tensor & {
    auto cur_rank = rank();
    if (cur_rank == new_rank) {
      return *this;
    }
    if (new_rank == SCALAR_RANK) {
      auto new_shape = Shape{};
      _s_check_shape_equivalency(shape_, new_shape);
      shape_ = std::move(new_shape);
    } else if (new_rank < cur_rank) {
      auto new_shape = shape_.clone().resize(new_rank, modify_front);

      // Calculate the total number of elements in the cramped axes of the shape and set it as the dimension of
      // the first or last axis to adjust the front or back accordingly.
      auto new_begin = shape_.begin() + (modify_front ? 0 : new_rank - 1);
      auto new_end = shape_.end() - (modify_front ? new_rank - 1 : 0);
      auto axis = modify_front ? 0 : new_rank - 1;
      auto cramped_dimension = std::accumulate(new_begin, new_end, Shape::SCALAR_SIZE, std::multiplies{});
      new_shape.set_axis(axis, cramped_dimension);
      shape_ = std::move(new_shape);
    } else {
      shape_.resize(new_rank, modify_front);
    }
    return *this;
  }

  /// \brief Flattens the tensor, i.e., reshapes it into a rank-1 tensor.
  /// \return A reference to self.
  constexpr auto flatten() -> Tensor & { return reshape(1); }

  /// \brief Applies the given transformation to all the elements of the tensor.
  /// \param[in] func The transformation function.
  /// \return A reference to self.
  constexpr auto transform(UnaryOperation auto func) noexcept -> Tensor & {
    std::transform(begin(), end(), begin(), func);
    return *this;
  }

  /// \brief Applies the given transformation to all the elements of the tensor.
  /// \param[in] first An iterator pointing to the beginning of the secondary range.
  /// \param[in] func The transformation function.
  /// \return A reference to self.
  template <std::input_iterator I_It>
  constexpr auto transform(I_It first, BinaryOperation auto func) noexcept -> Tensor & {
    std::transform(begin(), end(), first, begin(), func);
    return *this;
  }

  /// \brief Applies the given transformation to all the elements of the tensor.
  /// \param[in] range The secondary range.
  /// \param[in] func The transformation function.
  /// \return A reference to self.
  template <std::ranges::range R>
  constexpr auto transform(const R &range, BinaryOperation auto func) noexcept -> Tensor & {
    std::transform(begin(), end(), range.begin(), begin(), func);
    return *this;
  }

  /// \brief Applies the given transformation to all the elements of the tensor.
  /// \param[in] func The transformation function.
  /// \return A reference to self.
  ///
  /// \see Tensor::transform(UnaryOperation auto func)
  constexpr auto operator|=(UnaryOperation auto func) noexcept -> Tensor & { return transform(func); }

  /// \brief Applies the given transformation to all the elements and returns it as a transformed tensor.
  /// \tparam U The type of new tensor.
  /// \param[in] func The transformation function.
  /// \return The transformed tensor.
  template <typename U = value_type>
  [[nodiscard]] constexpr auto transformed(UnaryOperation auto func) const -> Tensor<U> {
    auto result = zeros_like<U>();
    std::transform(begin(), end(), result.begin(), func);
    return result;
  }

  /// \brief Applies the given transformation to all the elements and returns it as a transformed tensor.
  /// \tparam U The type of new tensor.
  /// \param[in] first An iterator pointing to the beginning of the secondary range.
  /// \param[in] func The transformation function.
  /// \return The transformed tensor.
  template <typename U = value_type, std::input_iterator I_It>
  [[nodiscard]] constexpr auto transformed(I_It first, BinaryOperation auto func) const -> Tensor<U> {
    auto result = zeros_like<U>();
    std::transform(begin(), end(), first, result.begin(), func);
    return result;
  }

  /// \brief Applies the given transformation to all the elements and returns it as a transformed tensor.
  /// \tparam U The type of new tensor.
  /// \param[in] range The secondary range.
  /// \param[in] func The transformation function.
  /// \return The transformed tensor.
  template <typename U = value_type, std::ranges::range R>
  [[nodiscard]] constexpr auto transformed(R range, BinaryOperation auto func) const -> Tensor<U> {
    auto result = zeros_like<U>();
    std::transform(begin(), end(), range.begin(), result.begin(), func);
    return result;
  }

  /// \brief Applies the given transformation to all the elements and returns it as a transformed tensor.
  /// \param[in] func The transformation function.
  /// \return The transformed tensor.
  ///
  /// \see Tensor::transformed(UnaryOperation auto func)
  [[nodiscard]] constexpr auto operator|(UnaryOperation auto func) const -> Tensor { return transformed(func); }

  /// \brief Clamps values outside the interval [\p lower_bound, \p upper_bound] to its edges.
  /// \param[in] lower_bound, upper_bound The interval boundaries.
  /// \return A reference to self.
  constexpr auto clamp(value_type lower_bound, value_type upper_bound) noexcept -> Tensor & {
    return transform([lower_bound, upper_bound](auto x) {
      return std::clamp(x, lower_bound, upper_bound);
    });
  }

  /// \brief Clamps values outside the interval [\p lower_bound, \p upper_bound] to its edges and returns it as
  /// a clamped tensor.
  /// \param[in] lower_bound, upper_bound The interval boundaries.
  /// \return The clamped tensor.
  [[nodiscard]] constexpr auto clamped(value_type lower_bound, value_type upper_bound) noexcept -> Tensor {
    return transformed([lower_bound, upper_bound](auto x) {
      return std::clamp(x, lower_bound, upper_bound);
    });
  }

  // /////////////////////////////////////////////
  // Arithmetic Operators
  // /////////////////////////////////////////////

  /// \brief Add and assign operator.
  /// \param[in] num A scalar operand.
  /// \return A reference to self.
  constexpr auto operator+=(Number auto num) noexcept -> Tensor & {
    return transform([num](auto x) {
      return x + num;
    });
  }

  /// \brief Subtract and assign operator.
  /// \param[in] num A scalar operand.
  /// \return A reference to self.
  constexpr auto operator-=(Number auto num) noexcept -> Tensor & {
    return transform([num](auto x) {
      return x - num;
    });
  }

  /// \brief Multiply and assign operator.
  /// \param[in] num A scalar operand.
  /// \return A reference to self.
  constexpr auto operator*=(Number auto num) noexcept -> Tensor & {
    return transform([num](auto x) {
      return x * num;
    });
  }

  /// \brief Divide and assign operator.
  /// \param[in] num A scalar operand.
  /// \return A reference to self.
  constexpr auto operator/=(Number auto num) noexcept -> Tensor & {
    return transform([num](auto x) {
      return x / num;
    });
  }

  /// \brief Modulus and assign operator.
  /// \param[in] num A scalar operand.
  /// \return A reference to self.
  constexpr auto operator%=(Number auto num) noexcept -> Tensor & {
    return transform([num](auto x) {
      return fmod(x, num);
    });
  }

  /// \brief Add and assign operator.
  /// \tparam U Data type of \p tensor.
  /// \param[in] tensor A tensor operand.
  /// \return A reference to self.
  ///
  /// \details
  /// This function throws an exception if \p tensor is not broadcastable to `this->shape()`.
  ///
  /// \throws ShapeError
  template <typename U>
  constexpr auto operator+=(const Tensor<U> &tensor) -> Tensor & {
    _m_check_broadcastability(tensor.shape());
    // Cyclic iterators are relatively more expensive than simple iterators. Hence, the conditional check
    // provides fairly significant optimization when `this->shape()` is identical to `tensor.shape()`.
    return rank() > tensor.rank() ? transform(make_cyclic_iterator(tensor), std::plus{})
                                  : transform(tensor.begin(), std::plus{});
  }

  /// \brief Subtract and assign operator.
  /// \tparam U Data type of \p tensor.
  /// \param[in] tensor A tensor operand.
  /// \return A reference to self.
  ///
  /// \details
  /// This function throws an exception if \p tensor is not broadcastable to `this->shape()`.
  ///
  /// \throws ShapeError
  template <typename U>
  constexpr auto operator-=(const Tensor<U> &tensor) -> Tensor & {
    _m_check_broadcastability(tensor.shape());
    // Cyclic iterators are relatively more expensive than simple iterators. Hence, the conditional check
    // provides fairly significant optimization when `this->shape()` is identical to `tensor.shape()`.
    return rank() > tensor.rank() ? transform(make_cyclic_iterator(tensor), std::minus{})
                                  : transform(tensor.begin(), std::minus{});
  }

  /// \brief Multiply and assign operator.
  /// \tparam U Data type of \p tensor.
  /// \param[in] tensor A tensor operand.
  /// \return A reference to self.
  ///
  /// \details
  /// This function throws an exception if \p tensor is not broadcastable to `this->shape()`.
  ///
  /// \throws ShapeError
  template <typename U>
  constexpr auto operator*=(const Tensor<U> &tensor) -> Tensor & {
    _m_check_broadcastability(tensor.shape());
    // Cyclic iterators are relatively more expensive than simple iterators. Hence, the conditional check
    // provides fairly significant optimization when `this->shape()` is identical to `tensor.shape()`.
    return rank() > tensor.rank() ? transform(make_cyclic_iterator(tensor), std::multiplies{})
                                  : transform(tensor.begin(), std::multiplies{});
  }

  /// \brief Divide and assign operator.
  /// \tparam U Data type of \p tensor.
  /// \param[in] tensor A tensor operand.
  /// \return A reference to self.
  ///
  /// \details
  /// This function throws an exception if \p tensor is not broadcastable to `this->shape()`.
  ///
  /// \throws ShapeError
  template <typename U>
  constexpr auto operator/=(const Tensor<U> &tensor) -> Tensor & {
    _m_check_broadcastability(tensor.shape());
    // Cyclic iterators are relatively more expensive than simple iterators. Hence, the conditional check
    // provides fairly significant optimization when `this->shape()` is identical to `tensor.shape()`.
    return rank() > tensor.rank() ? transform(make_cyclic_iterator(tensor), std::divides{})
                                  : transform(tensor.begin(), std::divides{});
  }

  /// \brief Modulus and assign operator.
  /// \tparam U Data type of \p tensor.
  /// \param[in] tensor A tensor operand.
  /// \return A reference to self.
  ///
  /// \details
  /// This function throws an exception if \p tensor is not broadcastable to `this->shape()`.
  ///
  /// \throws ShapeError
  template <typename U>
  constexpr auto operator%=(const Tensor<U> &tensor) -> Tensor & {
    _m_check_broadcastability(tensor.shape());
    auto modulus = [](auto x, auto y) {
      return std::fmod(x, y);
    };
    // Cyclic iterators are relatively more expensive than simple iterators. Hence, the conditional check
    // provides fairly significant optimization when `this->shape()` is identical to `tensor.shape()`.
    return rank() > tensor.rank() ? transform(make_cyclic_iterator(tensor), modulus)
                                  : transform(tensor.begin(), modulus);
  }

  // /////////////////////////////////////////////
  // Mathematical Operations
  // /////////////////////////////////////////////

  /// \brief Matrix multiplication.
  /// \tparam U Data type of \p tensor.
  /// \tparam resultant_value_t Data type of the resultant tensor.
  /// \param[in] tensor A tensor operand.
  /// \param[in] multithreading If true, this function will use multithreading.
  /// \return The resultant tensor.
  ///
  /// \details
  /// This function throws an exception if:
  ///     * Either of the tensors do not represent a matrix.
  ///     * The matrices are not compatible for multiplication.
  ///
  /// \throws RankError
  /// \throws ShapeError
  template <typename U, typename resultant_value_t = decltype(value_type{} * U{})>
  auto matmul(const Tensor<U> &tensor, bool multithreading = true) const -> Tensor<resultant_value_t> {
    _s_matrix_rank_check(rank());
    _s_matrix_rank_check(tensor.rank());

    auto [r1, c1] = shape_.template unwrap<2>();
    auto [r2, c2] = tensor.shape().template unwrap<2>();

    _s_matmul_compatibility_check(c1, r2);

    auto rows = r1, cols = c2, common_axis = c1;
    auto product = Tensor<resultant_value_t>::matrix(rows, cols);

    // Based on the number of rows in the product matrix, estimate how many rows will be assigned to each
    // thread.
    auto calculate_rows_per_thread = [](auto rows) -> size_type {
      // Arbitrarily establish a relation between thread count and matrix size.
      const auto ARBITRARY_CONSTANT_A = sizeof(usize);
      const auto ARBITRARY_CONSTANT_B = sizeof(i32) * (ARBITRARY_CONSTANT_A);
      auto factor = std::log(rows + ARBITRARY_CONSTANT_A);
      return std::floor(factor) * (ARBITRARY_CONSTANT_B - ARBITRARY_CONSTANT_A);
    };

    // Calculate how many threads will be required based on the number of total rows and rows assigned to each
    // thread.
    auto calculate_threads_required = [](f32 rows, auto rows_per_thread) -> size_type {
      return std::ceil(rows / rows_per_thread);
    };

    // Disable bounds checking as the loop is counter controlled, so there is no need for bounds checking.
    auto this_bounds_checking_enabled = this->is_bounds_checking_enabled();
    auto tensor_bounds_checking_enabled = tensor.is_bounds_checking_enabled();

    this->disable_bounds_checking();
    tensor.disable_bounds_checking();
    product.disable_bounds_checking();

    // The actual implementation for matrix multiplication without any multithreading witchcraft. It's a primary
    // schoolbook algorithm free from any optimization.
    auto impl = [&a = *this, &b = tensor, &product, cols, common_axis](auto row_start, auto row_count) {
      auto row_end = row_start + row_count;
      for (auto r = row_start; r < row_end; ++r) {
        for (size_type c = {}; c < cols; ++c) {
          for (size_type k = {}; k < common_axis; ++k) {
            product(r, c) += a(r, k) * b(k, c);
          }
        }
      }
    };

    // If multithreading is unsought, simply call the implementation lambda and return the product.
    if (not multithreading) {
      impl(size_type{}, rows);
      return product;
    }

    auto rows_per_thread = calculate_rows_per_thread(rows);
    auto threads_required = calculate_threads_required(rows, rows_per_thread);

    // Bookkeeping threads to call later for joining into the main thread.
    auto threads = std::vector<std::thread>{};
    threads.reserve(threads_required);

    // Construct each thread with implementation lambda and its parameters.
    for (size_type current_row = {}; current_row < rows; current_row += rows_per_thread) {
      auto distance = rows - current_row;
      auto num_of_rows = std::min(rows_per_thread, distance);
      threads.emplace_back(impl, current_row, num_of_rows);
    }

    // Call all threads to join into the main thread.
    for (auto &thread : threads) {
      thread.join();
    }

    // Return bounds checking to previous state.
    if (this_bounds_checking_enabled) {
      this->enable_bounds_checking();
    }
    if (tensor_bounds_checking_enabled) {
      tensor.disable_bounds_checking();
    }
    product.enable_bounds_checking();

    return product;
  }

  // /////////////////////////////////////////////
  // Utility
  // /////////////////////////////////////////////

  /// \brief Swaps the contents of this tensor with those of the other.
  /// \param[in] other The tensor whither the swap will transpire.
  /// \return A reference to self.
  constexpr auto swap(Tensor &other) noexcept -> Tensor & {
    std::swap(bounds_checking_, other.bounds_checking_);
    shape_.swap(other.shape_);
    data_.swap(other.data_);
    return *this;
  }

  /// \brief Clones the original tensor.
  /// \return A clone of the original tensor.
  [[nodiscard]] constexpr auto clone() const -> Tensor { return *this; }

  /// \brief Returns a zero-initialized tensor of an identical shape.
  /// \tparam U The type of new tensor.
  /// \return The new tensor.
  template <typename U = value_type>
  [[nodiscard]] constexpr auto zeros_like() const -> Tensor<U> {
    return Tensor<U>{shape_};
  }

  // /////////////////////////////////////////////////////////////
  // Static Functions
  // /////////////////////////////////////////////////////////////

  // /////////////////////////////////////////////
  // Factory Functions
  // /////////////////////////////////////////////

  /// \brief Returns a tensor of the specified shape populated with a custom arange.
  /// \param[in] shape The shape of the tensor.
  /// \param[in] func A arange generator.
  /// \return A custom tensor of the specified shape.
  [[nodiscard]] static auto custom(const Shape &shape, NullaryOperation auto func) -> Tensor {
    auto tensor = Tensor{shape};
    std::generate(tensor.begin(), tensor.end(), func);
    return tensor;
  }

  /// \brief Returns a matrix of the shape (\p row, \p col) populated with \p value.
  /// \param[in] value Initialization value.
  /// \return A matrix (or rank-2 tensor) of the specified dimensions.
  [[nodiscard]] static auto matrix(size_type row, size_type col, value_type value = {}) -> Tensor {
    return Tensor{{row, col}, value};
  }

  /// \brief Returns a tensor of the specified shape populated with ones.
  /// \param[in] shape The shape of the tensor.
  /// \return A tensor of the specified shape initialized with ones.
  [[nodiscard]] static auto ones(const Shape &shape) -> Tensor { return Tensor{shape, 1}; }

  /// \brief Returns a tensor of the specified shape populated with random values uniformly distributed on the
  /// closed interval [lower_bound, upper_bound].
  /// \param[in] shape The shape of the tensor.
  /// \param[in] seed The seed of randomness.
  /// \param[in] lower_bound, upper_bound The closed interval.
  /// \return A random tensor of the specified shape.
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

  /// \brief Returns a scalar-tensor initialized with \p value.
  /// \param[in] value Initialization value.
  /// \return A scalar (or rank-0 tensor).
  [[nodiscard]] static auto scalar(value_type value = {}) -> Tensor { return Tensor{{}, value}; }

  /// \brief Returns a tensor of the specified shape populated with the given range.
  /// \param[in] shape The shape of the tensor.
  /// \param[in] start The starting point of the range.
  /// \param[in] step The step to be taken for each consecutive element.
  /// \return A tensor filled with the given range.
  [[nodiscard]] static auto arange(const Shape &shape, value_type start = 0, value_type step = 1) -> Tensor {
    return Tensor::custom(shape, [step, n = start]() mutable {
      return std::exchange(n, n + step);
    });
  }

  /// \brief Returns a vector populated with \p value.
  /// \param[in] size The size of the vector.
  /// \param[in] value Initialization value.
  /// \return A vector (or rank-1 tensor) filled with \p value.
  [[nodiscard]] static auto vector(size_type size, value_type value = {}) -> Tensor {
    return Tensor{{size}, value};
  }

  /// \brief Returns a tensor of the specified shape populated with zeros.
  /// \param[in] shape The shape of the tensor.
  /// \return A tensor of the specified shape initialized with zeros.
  [[nodiscard]] static auto zeros(const Shape &shape) -> Tensor { return Tensor{shape}; }

  // /////////////////////////////////////////////////////////////
  // Friend Functions
  // /////////////////////////////////////////////////////////////

  // /////////////////////////////////////////////
  // Arithmetic Operators
  // /////////////////////////////////////////////

  /// \brief Addition operator.
  /// \tparam U Data type of \p b.
  /// \tparam resultant_value_t Data type of the resultant tensor.
  /// \param[in] a, b The operands.
  /// \return The resultant tensor.
  template <typename U, typename resultant_value_t = decltype(value_type{} + U{})>
  friend constexpr auto operator+(const Tensor &a, const Tensor<U> &b) -> Tensor<resultant_value_t> {
    auto broadcast_shape = _s_get_broadcast_shape(a.shape(), b.shape());
    auto resultant = Tensor<resultant_value_t>{broadcast_shape};
    // Cyclic iterators are relatively more expensive than simple iterators. Hence, the `else` branch provides
    // fairly significant optimization when `a` and `b` have identical shapes.
    if (a.rank() > b.rank()) {
      std::transform(a.begin(), a.end(), make_cyclic_iterator(b), resultant.begin(), std::plus{});
    } else if (b.rank() > a.rank()) {
      std::transform(b.begin(), b.end(), make_cyclic_iterator(a), resultant.begin(), std::plus{});
    } else {
      std::transform(a.begin(), a.end(), b.begin(), resultant.begin(), std::plus{});
    }
    return resultant;
  }

  /// \brief Subtraction operator.
  /// \tparam U Data type of \p b.
  /// \tparam resultant_value_t Data type of the resultant tensor.
  /// \param[in] a, b The operands.
  /// \return The resultant tensor.
  ///
  /// \details
  /// This function throws an exception if \p a and \p b are incompatible for broadcasting.
  ///
  /// \throws ShapeError
  template <typename U, typename resultant_value_t = decltype(value_type{} - U{})>
  friend constexpr auto operator-(const Tensor &a, const Tensor<U> &b) -> Tensor<resultant_value_t> {
    auto broadcast_shape = _s_get_broadcast_shape(a.shape(), b.shape());
    auto resultant = Tensor<resultant_value_t>{broadcast_shape};
    // Cyclic iterators are relatively more expensive than simple iterators. Hence, the `else` branch provides
    // fairly significant optimization when `a` and `b` have identical shapes.
    if (a.rank() > b.rank()) {
      std::transform(a.begin(), a.end(), make_cyclic_iterator(b), resultant.begin(), std::minus{});
    } else if (b.rank() > a.rank()) {
      std::transform(b.begin(), b.end(), make_cyclic_iterator(a), resultant.begin(), [](auto y, auto x) {
        return x - y;
      });
    } else {
      std::transform(a.begin(), a.end(), b.begin(), resultant.begin(), std::minus{});
    }
    return resultant;
  }

  /// \brief Multiplication operator.
  /// \tparam U Data type of \p b.
  /// \tparam resultant_value_t Data type of the resultant tensor.
  /// \param[in] a, b The operands.
  /// \return The resultant tensor.
  ///
  /// \details
  /// This function throws an exception if \p a and \p b are incompatible for broadcasting.
  ///
  /// \throws ShapeError
  template <typename U, typename resultant_value_t = decltype(value_type{} * U{})>
  friend constexpr auto operator*(const Tensor &a, const Tensor<U> &b) -> Tensor<resultant_value_t> {
    auto broadcast_shape = _s_get_broadcast_shape(a.shape(), b.shape());
    auto resultant = Tensor<resultant_value_t>{broadcast_shape};
    // Cyclic iterators are relatively more expensive than simple iterators. Hence, the `else` branch provides
    // fairly significant optimization when `a` and `b` have identical shapes.
    if (a.rank() > b.rank()) {
      std::transform(a.begin(), a.end(), make_cyclic_iterator(b), resultant.begin(), std::multiplies{});
    } else if (b.rank() > a.rank()) {
      std::transform(b.begin(), b.end(), make_cyclic_iterator(a), resultant.begin(), std::multiplies{});
    } else {
      std::transform(a.begin(), a.end(), b.begin(), resultant.begin(), std::multiplies{});
    }
    return resultant;
  }

  /// \brief Division operator.
  /// \tparam U Data type of \p b.
  /// \tparam resultant_value_t Data type of the resultant tensor.
  /// \param[in] a, b The operands.
  /// \return The resultant tensor.
  ///
  /// \details
  /// This function throws an exception if \p a and \p b are incompatible for broadcasting.
  ///
  /// \throws ShapeError
  template <typename U, typename resultant_value_t = decltype(value_type{} / U{})>
  friend constexpr auto operator/(const Tensor &a, const Tensor<U> &b) -> Tensor<resultant_value_t> {
    auto broadcast_shape = _s_get_broadcast_shape(a.shape(), b.shape());
    auto resultant = Tensor<resultant_value_t>{broadcast_shape};
    // Cyclic iterators are relatively more expensive than simple iterators. Hence, the `else` branch provides
    // fairly significant optimization when `a` and `b` have identical shapes.
    if (a.rank() > b.rank()) {
      std::transform(a.begin(), a.end(), make_cyclic_iterator(b), resultant.begin(), std::divides{});
    } else if (b.rank() > a.rank()) {
      std::transform(b.begin(), b.end(), make_cyclic_iterator(a), resultant.begin(), [](auto y, auto x) {
        return x / y;
      });
    } else {
      std::transform(a.begin(), a.end(), b.begin(), resultant.begin(), std::divides{});
    }
    return resultant;
  }

  /// \brief Modulus operator.
  /// \tparam U Data type of \p b.
  /// \tparam resultant_value_t Data type of the resultant tensor.
  /// \param[in] a, b The operands.
  /// \return The resultant tensor.
  ///
  /// \details
  /// This function throws an exception if \p a and \p b are incompatible for broadcasting.
  ///
  /// \throws ShapeError
  template <typename U, typename resultant_value_t = decltype(std::fmod(value_type{}, U{}))>
  friend constexpr auto operator%(const Tensor &a, const Tensor<U> &b) -> Tensor<resultant_value_t> {
    auto broadcast_shape = _s_get_broadcast_shape(a.shape(), b.shape());
    auto resultant = Tensor<resultant_value_t>{broadcast_shape};
    // Cyclic iterators are relatively more expensive than simple iterators. Hence, the `else` branch provides
    // fairly significant optimization when `a` and `b` have identical shapes.
    if (a.rank() > b.rank()) {
      std::transform(a.begin(), a.end(), make_cyclic_iterator(b), resultant.begin(), [](auto x, auto y) {
        return std::fmod(x, y);
      });
    } else if (b.rank() > a.rank()) {
      std::transform(b.begin(), b.end(), make_cyclic_iterator(a), resultant.begin(), [](auto y, auto x) {
        return std::fmod(x, y);
      });
    } else {
      std::transform(a.begin(), a.end(), b.begin(), resultant.begin(), [](auto x, auto y) {
        return std::fmod(x, y);
      });
    }
    return resultant;
  }
};

// /////////////////////////////////////////////////////////////
// External Functions
// /////////////////////////////////////////////////////////////

// /////////////////////////////////////////////
// Arithmetic Operators
// /////////////////////////////////////////////

// /////////////////////////////////
// Unary Operators
// /////////////////////////////////

/// \brief Unary Plus operator.
/// \tparam T Data type of \p tensor.
/// \param[in] tensor A tensor operand.
/// \return The resultant tensor.
template <typename T>
constexpr auto operator+(const Tensor<T> &tensor) noexcept -> Tensor<T> {
  return tensor;
}

/// \brief Unary Minus operator.
/// \tparam T Data type of \p tensor.
/// \param[in] tensor A tensor operand.
/// \return The resultant tensor.
template <typename T>
constexpr auto operator-(const Tensor<T> &tensor) noexcept -> Tensor<T> {
  return tensor | std::negate{};
}

// /////////////////////////////////
// Binary Operators
// /////////////////////////////////

/// \brief Addition operator.
/// \tparam T Data type of \p tensor.
/// \tparam N Data type of \p num.
/// \tparam resultant_value_t Data type of the resultant tensor.
/// \param[in] tensor, num The operands.
/// \return The resultant tensor.
template <typename T, Number N, typename resultant_value_t = decltype(T{} + N{})>
constexpr auto operator+(const Tensor<T> &tensor, N num) -> Tensor<resultant_value_t> {
  return tensor.template transformed<resultant_value_t>([num](auto x) {
    return x + num;
  });
}

/// \brief Addition operator.
/// \tparam N Data type of \p num.
/// \tparam T Data type of \p tensor.
/// \tparam resultant_value_t Data type of the resultant tensor.
/// \param[in] num, tensor The operands.
/// \return The resultant tensor.
template <Number N, typename T, typename resultant_value_t = decltype(N{} + T{})>
constexpr auto operator+(N num, const Tensor<T> &tensor) -> Tensor<resultant_value_t> {
  return tensor + num;
}

/// \brief Subtraction operator.
/// \tparam T Data type of \p tensor.
/// \tparam N Data type of \p num.
/// \tparam resultant_value_t Data type of the resultant tensor.
/// \param[in] tensor, num The operands.
/// \return The resultant tensor.
template <typename T, Number N, typename resultant_value_t = decltype(T{} - N{})>
constexpr auto operator-(const Tensor<T> &tensor, N num) -> Tensor<resultant_value_t> {
  return tensor.template transformed<resultant_value_t>([num](auto x) {
    return x - num;
  });
}

/// \brief Subtraction operator.
/// \tparam N Data type of \p num.
/// \tparam T Data type of \p tensor.
/// \tparam resultant_value_t Data type of the resultant tensor.
/// \param[in] num, tensor The operands.
/// \return The resultant tensor.
template <Number N, typename T, typename resultant_value_t = decltype(N{} - T{})>
constexpr auto operator-(N num, const Tensor<T> &tensor) -> Tensor<resultant_value_t> {
  return tensor.template transformed<resultant_value_t>([num](auto x) {
    return num - x;
  });
}

/// \brief Multiplication operator.
/// \tparam T Data type of \p tensor.
/// \tparam N Data type of \p num.
/// \tparam resultant_value_t Data type of the resultant tensor.
/// \param[in] tensor, num The operands.
/// \return The resultant tensor.
template <typename T, Number N, typename resultant_value_t = decltype(T{} * N{})>
constexpr auto operator*(const Tensor<T> &tensor, N num) -> Tensor<resultant_value_t> {
  return tensor.template transformed<resultant_value_t>([num](auto x) {
    return x * num;
  });
}

/// \brief Multiplication operator.
/// \tparam N Data type of \p num.
/// \tparam T Data type of \p tensor.
/// \tparam resultant_value_t Data type of the resultant tensor.
/// \param[in] num, tensor The operands.
/// \return The resultant tensor.
template <Number N, typename T, typename resultant_value_t = decltype(N{} * T{})>
constexpr auto operator*(N num, const Tensor<T> &tensor) -> Tensor<resultant_value_t> {
  return tensor * num;
}

/// \brief Division operator.
/// \tparam T Data type of \p tensor.
/// \tparam N Data type of \p num.
/// \tparam resultant_value_t Data type of the resultant tensor.
/// \param[in] tensor, num The operands.
/// \return The resultant tensor.
template <typename T, Number N, typename resultant_value_t = decltype(T{} / N{})>
constexpr auto operator/(const Tensor<T> &tensor, N num) -> Tensor<resultant_value_t> {
  return tensor.template transformed<resultant_value_t>([num](auto x) {
    return x / num;
  });
}

/// \brief Division operator.
/// \tparam N Data type of \p num.
/// \tparam T Data type of \p tensor.
/// \tparam resultant_value_t Data type of the resultant tensor.
/// \param[in] num, tensor The operands.
/// \return The resultant tensor.
template <Number N, typename T, typename resultant_value_t = decltype(N{} / T{})>
constexpr auto operator/(N num, const Tensor<T> &tensor) -> Tensor<resultant_value_t> {
  return tensor.template transformed<resultant_value_t>([num](auto x) {
    return num / x;
  });
}

/// \brief Modulus operator.
/// \tparam T Data type of \p tensor.
/// \tparam N Data type of \p num.
/// \tparam resultant_value_t Data type of the resultant tensor.
/// \param[in] tensor, num The operands.
/// \return The resultant tensor.
template <typename T, Number N, typename resultant_value_t = decltype(std::fmod(T{}, N{}))>
constexpr auto operator%(const Tensor<T> &tensor, N num) -> Tensor<resultant_value_t> {
  return tensor.template transformed<resultant_value_t>([num](auto x) {
    return std::fmod(x, num);
  });
}

/// \brief Modulus operator.
/// \tparam N Data type of \p num.
/// \tparam T Data type of \p tensor.
/// \tparam resultant_value_t Data type of the resultant tensor.
/// \param[in] num, tensor The operands.
/// \return The resultant tensor.
template <Number N, typename T, typename resultant_value_t = decltype(std::fmod(N{}, T{}))>
constexpr auto operator%(N num, const Tensor<T> &tensor) -> Tensor<resultant_value_t> {
  return tensor.template transformed<resultant_value_t>([num](auto x) {
    return std::fmod(num, x);
  });
}

}

#endif
