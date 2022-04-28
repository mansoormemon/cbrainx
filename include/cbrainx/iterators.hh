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

#ifndef CBRAINX__ITERATORS_HH_
#define CBRAINX__ITERATORS_HH_

#include <concepts>
#include <iterator>
#include <ranges>

#include "type_aliases.hh"

namespace cbx {

/// \brief The `CyclicIterator` class implements a cyclic random access iterator.
///
/// \details
/// The primary purpose of a cyclic iterator is to iterate over secondary containers. A cyclic iterator allows
/// you to loop through a container indefinitely. It's important to remember that negative indices and seemingly
/// out-of-bounds are also allowed for any valid range. Cyclic iterators are more expensive than simple
/// iterators. Its use should be averted unless critical.
///
/// \note An empty range is invalid, and its behavior is undefined.
template <std::random_access_iterator Iter>
class CyclicIterator {
 public:
  using iterator_type = Iter;

  using value_type = std::iter_value_t<iterator_type>;

  using reference = std::iter_reference_t<iterator_type>;
  using pointer = typename std::iterator_traits<iterator_type>::pointer;

  using difference_type = std::iter_difference_t<iterator_type>;

  using iterator_category = std::random_access_iterator_tag;
  using iterator_concept = std::random_access_iterator_tag;

 private:
  /// \brief Beginning of the range.
  iterator_type begin_ = {};

  /// \brief Length of the range.
  difference_type length_ = {};

  /// \brief Current position of the cursor.
  difference_type cursor_ = {};

  // /////////////////////////////////////////////
  // Helpers
  // /////////////////////////////////////////////

  /// \brief Returns the pointer as it is.
  /// \tparam U Datatype of the pointer.
  /// \param ptr The target pointer.
  /// \return The original pointer.
  template <typename U>
  [[nodiscard]] static constexpr auto _s_to_pointer(U *ptr) -> U * {
    return ptr;
  }

  /// \brief Returns the underlying pointer.
  /// \tparam U Datatype of the iterator.
  /// \param iter The target iterator.
  /// \return The underlying pointer.
  template <typename U>
  [[nodiscard]] static constexpr auto _s_to_pointer(U iter) -> pointer {
    return iter.operator->();
  }

  /// \brief Returns the absolute index for the given position.
  /// \param pos The target position.
  /// \return The absolute index.
  [[nodiscard]] constexpr auto _m_abs_index(difference_type pos) const -> difference_type {
    return pos >= 0 ? pos : pos + length_;
  }

  /// \brief Returns the position of the cursor after taking n steps.
  /// \param n Position relative to the current position of the cursor.
  /// \return The final position of the cursor.
  [[nodiscard]] constexpr auto _m_next_pos(difference_type n) const -> difference_type {
    difference_type pos = (cursor_ + n) % length_;
    return _m_abs_index(pos);
  }

 public:
  // /////////////////////////////////////////////
  // Constructors and Destructors
  // /////////////////////////////////////////////

  /// \brief Default constructor.
  CyclicIterator() = default;

  /// \brief Parameterized constructor.
  /// \param iter_begin Beginning of the range.
  /// \param length Length of the range.
  /// \param n Current position of the cursor.
  constexpr CyclicIterator(iterator_type iter_begin, difference_type length, difference_type n = {})
      : begin_{iter_begin}, length_{length} {
    cursor_ = _m_next_pos(n);
  }

  /// \brief Parameterized constructor.
  /// \param iter_begin Beginning of the range.
  /// \param iter_end Ending of the range.
  /// \param n Current position of the cursor.
  constexpr CyclicIterator(iterator_type iter_begin, iterator_type iter_end, difference_type n = {})
      : CyclicIterator{iter_begin, std::distance(iter_begin, iter_end), n} {}

  /// \brief Default copy constructor.
  /// \param other Source iterator.
  constexpr CyclicIterator(const CyclicIterator &other) = default;

  /// \brief Default destructor.
  ~CyclicIterator() = default;

  // /////////////////////////////////////////////
  // Assignment Operators
  // /////////////////////////////////////////////

  /// \brief Default copy assignment operator.
  /// \param other Source iterator.
  /// \return A reference to self.
  constexpr auto operator=(const CyclicIterator &other) -> CyclicIterator & = default;

  // /////////////////////////////////////////////
  // Accessors (and Mutators)
  // /////////////////////////////////////////////

  /// \brief Returns the current base iterator.
  /// \return The base iterator.
  [[nodiscard]] constexpr auto base() const -> iterator_type { return begin_ + cursor_; }

  /// \brief Returns an iterator pointing to the beginning of the cyclic range.
  /// \return An iterator pointing to the beginning of the cyclic range.
  [[nodiscard]] constexpr auto head() const -> iterator_type { return begin_; }

  /// \brief Returns an iterator pointing to the ending of the cyclic range.
  /// \return An iterator pointing to the ending of the cyclic range.
  [[nodiscard]] constexpr auto tail() const -> iterator_type { return begin_ + length_; }

  // /////////////////////////////////////////////
  // Query Functions
  // /////////////////////////////////////////////

  /// \brief Returns whether or not the range is empty.
  /// \return True if the range is empty.
  ///
  /// \note Empty ranges are invalid, and dereferencing them results in undefined behavior.
  [[nodiscard]] constexpr auto is_empty() const -> bool { return begin_ == tail(); }

  // /////////////////////////////////////////////
  // Modifiers
  // /////////////////////////////////////////////

  /// \brief Dereferences the iterator.
  /// \return A reference to underlying range.
  [[nodiscard]] constexpr auto operator*() const -> reference { return *base(); }

  /// \brief Arrow operator.
  /// \return The underlying pointer.
  [[nodiscard]] constexpr auto operator->() const -> pointer { return _s_to_pointer(base()); }

  /// \brief Add and assign operator.
  /// \param n Position relative to the current position of the cursor.
  /// \return A reference to self.
  constexpr auto operator+=(difference_type n) -> CyclicIterator & {
    cursor_ = _m_next_pos(n);
    return *this;
  }

  /// \brief Prefix increment operator.
  /// \return A reference to self.
  constexpr auto operator++() -> CyclicIterator & { return operator+=(1); }

  /// \brief Postfix increment operator.
  /// \return Returns a cyclic iterator which is advanced by 1.
  [[nodiscard]] constexpr auto operator++(i32) -> CyclicIterator {
    auto tmp = *this;
    operator+=(1);
    return tmp;
  }

  /// \brief Subtract and assign operator.
  /// \param n Position relative to the current position of the cursor.
  /// \return A reference to self.
  constexpr auto operator-=(difference_type n) -> CyclicIterator & {
    cursor_ = _m_next_pos(-n);
    return *this;
  }

  /// \brief Prefix decrement operator.
  /// \return A reference to self.
  constexpr auto operator--() -> CyclicIterator & { return operator-=(1); }

  /// \brief Postfix decrement operator.
  /// \return Returns a cyclic iterator which is backward by 1.
  [[nodiscard]] constexpr auto operator--(i32) -> CyclicIterator {
    auto tmp = *this;
    operator-=(1);
    return tmp;
  }

  /// \brief Subscript operator.
  /// \param n Position relative to the current position of the cursor.
  /// \return Returns a reference to the element at \p n.
  [[nodiscard]] constexpr auto operator[](difference_type n) const -> reference {
    return begin_[_m_next_pos(n)];
  }
};

// /////////////////////////////////////////////////////////////
// External Functions
// /////////////////////////////////////////////////////////////

// /////////////////////////////////////////////
// Factory Functions
// /////////////////////////////////////////////

/// \brief Creates a cyclic iterator from the given range.
/// \param range The target range.
/// \return A cyclic iterator for \p range.
template <std::ranges::range R>
auto make_cyclic_iterator(R &range) {
  return CyclicIterator{std::begin(range), std::end(range)};
}

// /////////////////////////////////////////////
// Relational Operators
// /////////////////////////////////////////////

/// \brief Returns where or not the two iterators are equal.
/// \tparam IterL, IterR Underlying iterator types of the operands.
/// \param x, y The operands.
/// \return True if the iterators are equal.
template <typename IterL, typename IterR>
constexpr auto operator==(const CyclicIterator<IterL> &x, const CyclicIterator<IterR> &y) -> bool {
  return x.base() == y.base();
}

/// \brief Returns where or not the iterators are unequal.
/// \tparam IterL, IterR Underlying iterator types of the operands.
/// \param x, y The operands.
/// \return True if the iterators are unequal.
template <typename IterL, typename IterR>
constexpr auto operator!=(const CyclicIterator<IterL> &x, const CyclicIterator<IterR> &y) -> bool {
  return x.base() != y.base();
}

/// \brief Returns where or not \p x is less than \p y.
/// \tparam IterL, IterR Underlying iterator types of the operands.
/// \param x, y The operands.
/// \return True if \p x is less than \p y.
template <typename IterL, typename IterR>
constexpr auto operator<(const CyclicIterator<IterL> &x, const CyclicIterator<IterR> &y) -> bool {
  return x.base() < y.base();
}

/// \brief Returns where or not \p x is greater than \p y.
/// \tparam IterL, IterR Underlying iterator types of the operands.
/// \param x, y The operands.
/// \return True if \p x is greater than \p y.
template <typename IterL, typename IterR>
constexpr bool operator>(const CyclicIterator<IterL> &x, const CyclicIterator<IterR> &y) {
  return x.base() > y.base();
}

/// \brief Returns where or not \p x is less than or equal to \p y.
/// \tparam IterL, IterR Underlying iterator types of the operands.
/// \param x, y The operands.
/// \return True if \p x is less than or equal to \p y.
template <typename IterL, typename IterR>
constexpr auto operator<=(const CyclicIterator<IterL> &x, const CyclicIterator<IterR> &y) -> bool {
  return x.base() <= y.base();
}

/// \brief Returns where or not \p x is greater than or equal to \p y.
/// \tparam IterL, IterR Underlying iterator types of the operands.
/// \param x, y The operands.
/// \return True if \p x is greater than or equal to \p y.
template <typename IterL, typename IterR>
constexpr auto operator>=(const CyclicIterator<IterL> &x, const CyclicIterator<IterR> &y) -> bool {
  return x.base() >= y.base();
}

/// \brief Spaceship operator.
/// \tparam IterL, IterR Underlying iterator types of the operands.
/// \param x, y The operands.
/// \return A three way comparator.
template <typename IterL, std::three_way_comparable_with<IterL> IterR>
constexpr auto operator<=>(const CyclicIterator<IterL> &x, const CyclicIterator<IterR> &y) {
  return y.base() <=> x.base();
}

// /////////////////////////////////////////////
// Arithmetic Operators
// /////////////////////////////////////////////

/// \brief Addition operator.
/// \tparam Iter Underlying iterator type of \p it.
/// \param it Source iterator.
/// \param n Position relative to the \p it.
/// \return Returns a cyclic iterator which is advanced by n.
template <typename Iter>
[[nodiscard]] constexpr auto operator+(const CyclicIterator<Iter> &it, std::iter_difference_t<Iter> n) {
  auto len = std::distance(it.head(), it.base());
  return CyclicIterator{it.head(), it.tail(), len + n};
}

/// \brief Addition operator.
/// \tparam Iter Underlying iterator type of \p it.
/// \param n Position relative to the \p it.
/// \param it Source iterator.
/// \return Returns a cyclic iterator which is advanced by n.
template <typename Iter>
[[nodiscard]] constexpr auto operator+(std::iter_difference_t<Iter> n, const CyclicIterator<Iter> &it) {
  return it + n;
}

/// \brief Subtraction operator.
/// \tparam Iter Underlying iterator type of \p it.
/// \param it Source iterator.
/// \param n Position relative to the \p it.
/// \return Returns a cyclic iterator which is backward by n.
template <typename Iter>
[[nodiscard]] constexpr auto operator-(const CyclicIterator<Iter> &it, std::iter_difference_t<Iter> n) {
  auto len = std::distance(it.head(), it.base());
  return CyclicIterator{it.head(), it.tail(), len - n};
}

/// \brief Returns the difference between two iterators.
/// \tparam IterL, IterR Underlying iterator types of the operands.
/// \param x, y The operands.
/// \return The difference between two iterators.
template <typename IterL, typename IterR>
[[nodiscard]] constexpr auto operator-(const CyclicIterator<IterL> &x, const CyclicIterator<IterR> &y) {
  return x.base() - y.base();
}

}

#endif
