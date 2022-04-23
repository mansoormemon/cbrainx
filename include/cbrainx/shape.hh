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

#ifndef CBRAINX__SHAPE_HH_
#define CBRAINX__SHAPE_HH_

#include <iterator>
#include <ranges>
#include <string>
#include <tuple>
#include <vector>

#include "type_aliases.hh"

namespace cbx {

/// \brief The `Shape` class represents the shape of a tensor.
///
/// \details
/// A shape is an ordered container whose length is its rank, and elements represent the dimensions of each
/// axis. Axes are the components that make up the rank, and dimension refers to the number of elements in a
/// given axis. In many instances, axis and dimensions are interchangeable, but there is a subtlety between
/// them. In simple terms, the axis represents the dimensions of data.
///
/// A subtle detail to cognize is that zero is not acceptable as a dimension for any axis. This constraint
/// averts ambiguity during memory allocation because the shape determines how much memory needs to be allocated
/// by the tensor, which would no longer be valid if the product of its dimensions became zero.
///
/// \see Tensor
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

  // /////////////////////////////////////////////
  // Constant(s)
  // /////////////////////////////////////////////

  /// \brief Total number of elements in a scalar.
  static constexpr size_type SCALAR_SIZE = 1;

 private:
  /// \brief Dimensions of shape.
  container data_ = {};

  // /////////////////////////////////////////////
  // Helpers
  // /////////////////////////////////////////////

  /// \brief Performs bounds checking.
  /// \param[in] index The index of the axis.
  ///
  /// \details
  /// This function throws an exception if the \p index is out of bounds.
  ///
  /// \throws IndexOutOfBoundsError
  auto _m_check_bounds(size_type index) const -> void;

  /// \brief Performs rank checking.
  /// \param[in] N The rank to be checked.
  ///
  /// \details
  /// This function throws an exception if the \p N is greater than the rank.
  ///
  /// \throws RankError
  auto _m_check_rank(size_type N) const -> void;

  /// \brief Performs validation check.
  /// \param[in] value The dimension value to be validated.
  ///
  /// \details
  /// This function throws an exception if the \p value is zero.
  ///
  /// \throws ValueError
  static auto _s_validate_dimension(value_type value) -> void;

  /// \brief Performs validation check for a range.
  /// \param[in] first, last The range to be validated.
  ///
  /// \details
  /// This function throws an exception if any value in the range [\p first, \p last) is zero.
  ///
  /// \throws ValueError
  template <std::input_iterator I_It>
  static auto _validate_dimensions(I_It first, I_It last) -> void {
    for (std::input_iterator auto it = first; it != last; ++it) {
      _s_validate_dimension(*it);
    }
  }

 public:
  // /////////////////////////////////////////////
  // Constructors and Destructors
  // /////////////////////////////////////////////

  /// \brief Default constructor.
  Shape() = default;

  /// \brief Copy constructor.
  /// \param[in] other Source shape.
  Shape(const Shape &other) = default;

  /// \brief Move constructor.
  /// \param[in] other Source shape.
  Shape(Shape &&other) noexcept;

  /// \brief Initializer list constructor.
  /// \param[in] ilist List of dimensions of shape.
  ///
  /// \details
  /// This constructor throws an exception if any value in \p ilist is zero.
  ///
  /// \throws ValueError
  Shape(std::initializer_list<value_type> ilist);

  /// \brief Iterator range constructor.
  /// \param[in] first, last The dimensions of shape.
  ///
  /// \details
  /// This constructor throws an exception if any value in the range [\p first, \p last) is zero.
  ///
  /// \note Behaviour is undefined if the range is invalid.
  ///
  /// \throws ValueError
  template <std::input_iterator I_It>
  Shape(I_It first, I_It last) {
    _s_validate_dimension(first, last);
    data_.assign(first, last);
  }

  /// \brief Range constructor.
  /// \param[in] range The dimensions of shape.
  ///
  /// \details
  /// This constructor throws an exception if any value in \p range is zero.
  ///
  /// \throws ValueError
  explicit Shape(const std::ranges::range auto &range) {
    _s_validate_dimension(range.begin(), range.end());
    data_.assign(range.begin(), range.end());
  }

  /// \brief Default destructor.
  ~Shape() = default;

  // /////////////////////////////////////////////
  // Assignment Operators
  // /////////////////////////////////////////////

  /// \brief Default copy assignment operator.
  /// \param[in] other Source shape.
  /// \return A reference to self.
  auto operator=(const Shape &other) -> Shape & = default;

  /// \brief Default move assignment operator.
  /// \param[in] other Source shape.
  /// \return A reference to self.
  auto operator=(Shape &&other) noexcept -> Shape &;

  // /////////////////////////////////////////////
  // Element Access
  // /////////////////////////////////////////////

  /// \brief Accesses the element at the specified index.
  /// \param[in] index The index of the element.
  /// \return An immutable reference to the element at the specified index.
  [[nodiscard]] auto operator[](size_type index) const noexcept -> const_reference;

  /// \brief Accesses the element at the specified index.
  /// \param[in] index The index of the element.
  /// \return An immutable reference to the element at the specified index.
  ///
  /// \note This function performs bounds checking.
  ///
  /// \throws IndexOutOfBoundsError
  [[nodiscard]] auto at(size_type index) const -> const_reference;

  /// \brief Accesses the first dimension.
  /// \return An immutable reference to the first dimension.
  ///
  /// \note Behaviour is undefined for scalars.
  [[nodiscard]] auto front() const -> const_reference;

  /// \brief Accesses the last dimension.
  /// \return An immutable reference to the last dimension.
  ///
  /// \note Behaviour is undefined for scalars.
  [[nodiscard]] auto back() const -> const_reference;

 private:
  /// \brief Helper for unwrap().
  /// \tparam T The type for static casting.
  /// \tparam Indices The axes for constructing the tuple.
  /// \return The dimensions of shape as a tuple.
  template <typename T, size_type... Axes>
  constexpr auto unwrap_helper(std::index_sequence<Axes...>) const {
    return std::make_tuple(T(data_[Axes])...);
  }

 public:
  /// \brief Returns first \p N dimensions as a tuple.
  /// \tparam N The length of tuple.
  /// \tparam T The type of elements.
  /// \return A tuple of \p N elements each of type \p T.
  ///
  /// \details
  /// This function throws an exception if the \p N is greater than the rank.
  ///
  /// \throws RankError
  template <size_type N, typename T = value_type>
  constexpr auto unwrap() const {
    _m_check_rank(N);
    return unwrap_helper<T>(std::make_index_sequence<N>());
  }

  // /////////////////////////////////////////////
  // Accessors and Mutators
  // /////////////////////////////////////////////

  /// \brief Returns the underlying container holding the data.
  /// \return An immutable reference to the underlying container.
  [[nodiscard]] auto underlying_container() const noexcept -> const container &;

  /// \brief Returns the rank of this shape, i.e., the length of the underlying container.
  /// \return The rank of this shape.
  [[nodiscard]] auto rank() const noexcept -> size_type;

  /// \brief Sets the dimension of the specified axis.
  /// \param[in] index The index of the axis.
  /// \param[in] value The new dimension of the axis.
  /// \return A reference to self.
  ///
  /// \details
  /// This function throws an exception if:
  /// * the \p index is out of bounds.
  /// * the \p value is zero.
  ///
  /// \throws IndexOutOfBoundsError
  /// \throws ValueError
  auto set_axis(size_type index, value_type value) -> Shape &;

  // /////////////////////////////////////////////
  // Iterators
  // /////////////////////////////////////////////

  /// \brief Returns an immutable random access iterator pointing to the first element of the underlying
  /// container.
  /// \return An immutable iterator pointing to the beginning of the container.
  [[nodiscard]] auto cbegin() const noexcept -> const_iterator;

  /// \brief Returns a random access iterator pointing to the first element of the underlying container.
  /// \return An immutable iterator pointing to the beginning of the container.
  [[nodiscard]] auto begin() const noexcept -> const_iterator;

  /// \brief Returns an immutable reverse random access iterator pointing to the last element of the underlying
  /// container.
  /// \return An immutable reverse iterator pointing to the reverse beginning of the container.
  [[nodiscard]] auto crbegin() const noexcept -> const_reverse_iterator;

  /// \brief Returns a reverse random access iterator pointing to the last element of the underlying container.
  /// \return An immutable reverse iterator pointing to the reverse beginning of the container.
  [[nodiscard]] auto rbegin() const noexcept -> const_reverse_iterator;

  /// \brief Returns an immutable random access iterator pointing to the last element of the underlying
  /// container.
  /// \return An immutable iterator pointing to the ending of the container.
  [[nodiscard]] auto cend() const noexcept -> const_iterator;

  /// \brief Returns a random access iterator pointing to the last element of the underlying container.
  /// \return An immutable iterator pointing to the ending of the container.
  [[nodiscard]] auto end() const noexcept -> const_iterator;

  /// \brief Returns an immutable reverse random access iterator pointing to the first element of the underlying
  /// container.
  /// \return An immutable reverse iterator pointing to the reverse ending of the container.
  [[nodiscard]] auto crend() const noexcept -> const_reverse_iterator;

  /// \brief Returns a reverse random access iterator pointing to the first element of the underlying
  /// container.
  /// \return An immutable reverse iterator pointing to the reverse ending of the container.
  [[nodiscard]] auto rend() const noexcept -> const_reverse_iterator;

  // /////////////////////////////////////////////
  // Query Functions
  // /////////////////////////////////////////////

  /// \brief Checks if two shapes are equivalent, i.e., they have the same number of total elements.
  /// \param[in] other The shape to be compared for equivalence.
  /// \return True if the shapes are equivalent.
  [[nodiscard]] auto is_equivalent(const Shape &other) const noexcept -> bool;

  /// \brief Returns the total number of elements.
  /// \return Total number of elements.
  ///
  /// \note This function returns `Shape::SCALAR_SIZE` for scalars.
  [[nodiscard]] auto total() const noexcept -> size_type;

  // /////////////////////////////////////////////
  // Informative
  // /////////////////////////////////////////////

  /// \brief Returns meta-information about the shape as a string.
  /// \return A string containing meta-information about the shape.
  [[nodiscard]] auto meta_info() const noexcept -> std::string;

  /// \brief Converts the shape to a string.
  /// \return A string representing the shape in conventional python's tuple notation.
  [[nodiscard]] auto to_string() const noexcept -> std::string;

  // /////////////////////////////////////////////
  // Modifiers
  // /////////////////////////////////////////////

  /// \brief Resizes the shape to the specified rank starting from the rear end.
  /// \param[in] new_rank Target rank.
  /// \param[in] modify_front If true, the front of the shape will be modified to conform with the new rank.
  /// \return A reference to self.
  auto resize(size_type new_rank, bool modify_front = false) -> Shape &;

  // /////////////////////////////////////////////
  // Utility
  // /////////////////////////////////////////////

  /// \brief Swaps the contents of this shape with those of the other.
  /// \param[in] other The shape whither the swap will transpire.
  /// \return A reference to self.
  auto swap(Shape &other) noexcept -> Shape &;

  /// \brief Clones the original shape.
  /// \return A clone of the original shape.
  [[nodiscard]] auto clone() const -> Shape;
};

// /////////////////////////////////////////////////////////////
// External Functions
// /////////////////////////////////////////////////////////////

// /////////////////////////////////////////////
// Equality Operators
// /////////////////////////////////////////////

/// \brief Compares the two shapes for equality.
/// \param[in] a, b The shapes to be compared for equality.
/// \return True if the shapes are equal.
///
/// \details
/// Two shapes are equal if they are lexicographically equal.
///
/// \relates Shape
auto operator==(const Shape &a, const Shape &b) noexcept -> bool;

/// \brief Compares the two shapes for inequality.
/// \param[in] a, b The shapes to be compared for inequality.
/// \return True if the shapes are unequal.
///
/// \details
/// Two shapes are unequal if they are lexicographically unequal.
///
/// \relates Shape
auto operator!=(const Shape &a, const Shape &b) noexcept -> bool;

// @deprecated
using shape_value_t = typename Shape::value_type;

// @deprecated
using shape_size_t = typename Shape::size_type;

}

#endif
