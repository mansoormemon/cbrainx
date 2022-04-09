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

#ifndef CBRAINX__CUSTOM_ITERATORS_HH_
#define CBRAINX__CUSTOM_ITERATORS_HH_

#include <concepts>
#include <iterator>
#include <ranges>

#include "type_aliases.hh"

namespace cbx {

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
  iterator_type begin_ = {};
  difference_type length_ = {};
  difference_type cursor_ = {};

  // /////////////////////////////////////////////////////////////
  // Helpers
  // /////////////////////////////////////////////////////////////

  template <typename U>
  [[nodiscard]] static constexpr auto _s_to_pointer(U *ptr) -> U * {
    return ptr;
  }

  template <typename U>
  [[nodiscard]] static constexpr auto _s_to_pointer(U iter) -> pointer {
    return iter.operator->();
  }

  [[nodiscard]] constexpr auto _m_abs_index(difference_type pos) const -> difference_type {
    return pos >= 0 ? pos : pos + length_;
  }

  [[nodiscard]] constexpr auto _m_next_pos(difference_type n) const {
    difference_type pos = (cursor_ + n) % length_;
    return _m_abs_index(pos);
  }

 public:
  // /////////////////////////////////////////////////////////////
  // Constructors and Destructors
  // /////////////////////////////////////////////////////////////

  CyclicIterator() = default;

  constexpr CyclicIterator(iterator_type iter_begin, difference_type length, difference_type n = {})
      : begin_{iter_begin}, length_{length} {
    cursor_ = _m_next_pos(n);
  }

  constexpr CyclicIterator(iterator_type iter_begin, iterator_type iter_end, difference_type n = {})
      : CyclicIterator{iter_begin, std::distance(iter_begin, iter_end), n} {}

  constexpr CyclicIterator(const CyclicIterator &other) = default;

  ~CyclicIterator() = default;

  // /////////////////////////////////////////////////////////////
  // Assignment Operator(s)
  // /////////////////////////////////////////////////////////////

  constexpr auto operator=(const CyclicIterator &other) -> CyclicIterator & = default;

  // /////////////////////////////////////////////////////////////
  // Accessors (and Mutators)
  // /////////////////////////////////////////////////////////////

  [[nodiscard]] constexpr auto base() const -> iterator_type { return begin_ + cursor_; }

  [[nodiscard]] constexpr auto head() const -> iterator_type { return begin_; }

  [[nodiscard]] constexpr auto tail() const -> iterator_type { return begin_ + length_; }

  // /////////////////////////////////////////////////////////////
  // Query Functions
  // /////////////////////////////////////////////////////////////

  [[nodiscard]] constexpr auto is_empty() const -> bool { return begin_ == tail(); }

  // /////////////////////////////////////////////////////////////
  // Modifiers
  // /////////////////////////////////////////////////////////////

  [[nodiscard]] constexpr auto operator*() const -> reference { return *base(); }

  [[nodiscard]] constexpr auto operator->() const -> pointer { return _s_to_pointer(base()); }

  constexpr auto operator+=(difference_type n) -> CyclicIterator & {
    cursor_ = _m_next_pos(n);
    return *this;
  }

  constexpr auto operator++() -> CyclicIterator & { return operator+=(1); }

  [[nodiscard]] constexpr auto operator++(int) -> CyclicIterator {
    auto tmp = *this;
    operator+=(1);
    return tmp;
  }

  [[nodiscard]] constexpr auto operator+(difference_type n) const -> CyclicIterator {
    return CyclicIterator{begin_, length_, cursor_ + n};
  }

  constexpr auto operator-=(difference_type n) -> CyclicIterator & {
    cursor_ = _m_next_pos(-n);
    return *this;
  }

  constexpr auto operator--() -> CyclicIterator & { return operator-=(1); }

  [[nodiscard]] constexpr auto operator--(int) -> CyclicIterator {
    auto tmp = *this;
    operator-=(1);
    return tmp;
  }

  [[nodiscard]] constexpr auto operator-(difference_type n) const -> CyclicIterator {
    return CyclicIterator{begin_, length_, cursor_ - n};
  }

  [[nodiscard]] constexpr auto operator[](difference_type n) const -> reference {
    return begin_[_m_next_pos(n)];
  }
};

// /////////////////////////////////////////////////////////////////////////////////////////////
// External Functions
// /////////////////////////////////////////////////////////////////////////////////////////////

// /////////////////////////////////////////////////////////////
// Relational Operators
// /////////////////////////////////////////////////////////////

template <typename IterL, typename IterR>
constexpr auto operator==(const CyclicIterator<IterL> &x, const CyclicIterator<IterR> &y) -> bool {
  return x.base() == y.base();
}

template <typename IterL, typename IterR>
constexpr auto operator!=(const CyclicIterator<IterL> &x, const CyclicIterator<IterR> &y) -> bool {
  return x.base() != y.base();
}

template <typename IterL, typename IterR>
constexpr auto operator<(const CyclicIterator<IterL> &x, const CyclicIterator<IterR> &y) -> bool {
  return x.base() < y.base();
}

template <typename IterL, typename IterR>
constexpr bool operator>(const CyclicIterator<IterL> &x, const CyclicIterator<IterR> &y) {
  return x.base() > y.base();
}

template <typename IterL, typename IterR>
constexpr auto operator<=(const CyclicIterator<IterL> &x, const CyclicIterator<IterR> &y) -> bool {
  return x.base() <= y.base();
}

template <typename IterL, typename IterR>
constexpr auto operator>=(const CyclicIterator<IterL> &x, const CyclicIterator<IterR> &y) -> bool {
  return x.base() >= y.base();
}

template <typename IterL, std::three_way_comparable_with<IterL> IterR>
constexpr auto operator<=>(const CyclicIterator<IterL> &x, const CyclicIterator<IterR> &y)
    -> std::compare_three_way_result_t<IterL, IterR> {
  return y.base() <=> x.base();
}

// /////////////////////////////////////////////////////////////
// Arithmetic Operators
// /////////////////////////////////////////////////////////////

template <typename Iter>
constexpr auto operator+(std::iter_difference_t<Iter> n, const CyclicIterator<Iter> &x)
    -> CyclicIterator<Iter> {
  return CyclicIterator{x.head(), x.tail(), n};
}

template <typename IterL, typename IterR>
constexpr auto operator-(const CyclicIterator<IterL> &x, const CyclicIterator<IterR> &y)
    -> decltype(x.base() - y.base()) {
  return x.base() - y.base();
}

}

#endif
