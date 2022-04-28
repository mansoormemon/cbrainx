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

#ifndef CBRAINX__CUSTOM_VIEWS_HH_
#define CBRAINX__CUSTOM_VIEWS_HH_

#include <concepts>
#include <iterator>
#include <ranges>

#include "iterators.hh"
#include "type_aliases.hh"

namespace cbx {

/// \brief The `CyclicView` class implements a corresponding view for the `CyclicIterator` class.
///
/// \see CyclicIterator
template <std::ranges::view V>
requires std::ranges::bidirectional_range<V>
class CyclicView : public std::ranges::view_interface<CyclicView<V>> {
 public:
  using view_type = V;

  using view_iterator_type = std::ranges::iterator_t<view_type>;

  using view_reference = view_type &;
  using view_const_reference = const view_type &;

  using iterator_type = CyclicIterator<view_iterator_type>;
  using difference_type = std::iter_difference_t<iterator_type>;

 private:
  /// Base view.
  view_type base_ = {};

  /// \brief Cyclic iterator.
  iterator_type iter_ = {std::begin(base_), std::end(base_)};

 public:
  /// \brief Default constructor.
  CyclicView() = default;

  /// \brief Parameterized constructor.
  /// \param base Base view.
  constexpr explicit CyclicView(view_type base)
      : base_{std::move(base)}, iter_{std::begin(base_), std::end(base_)} {}

  /// \brief Returns an immutable lvalue reference to the underlying view.
  /// \return An immutable reference to the underlying view.
  constexpr auto base() -> view_const_reference requires std::copy_constructible<view_type> { return base; }

  /// \brief Returns an rvalue reference to the underlying view.
  /// \return An rvalue reference to the underlying view.
  constexpr auto base() -> view_type && { return std::move(base_); }

  /// \brief Returns a cyclic iterator pointing to the first element of the underlying view.
  /// \return An immutable iterator pointing to the beginning of the view.
  constexpr auto begin() const -> iterator_type requires std::ranges::common_range<const view_type> {
    return iter_;
  }

  /// \brief Returns a cyclic iterator pointing to the first element of the underlying view.
  /// \return An immutable iterator pointing to the beginning of the view.
  constexpr auto begin() -> iterator_type requires std::ranges::common_range<view_type> { return iter_; }

  /// \brief Returns a cyclic iterator pointing to the last element of the underlying view.
  /// \return An immutable iterator pointing to the ending of the view.
  constexpr auto end() const -> iterator_type requires std::ranges::common_range<const view_type> {
    return iterator_type{iter_.tail(), iter_.tail()};
  }

  /// \brief Returns a cyclic iterator pointing to the last element of the underlying view.
  /// \return An immutable iterator pointing to the ending of the view.
  constexpr auto end() -> iterator_type requires std::ranges::common_range<view_type> {
    return iterator_type{iter_.tail(), iter_.tail()};
  }

  /// \brief Returns the size of the view.
  /// \return The size of the view.
  constexpr auto size() requires std::ranges::sized_range<view_type> { return std::ranges::size(base_); }

  /// \brief Returns the size of the view.
  /// \return The size of the view.
  constexpr auto size() const requires std::ranges::sized_range<const view_type> {
    return std::ranges::size(base_);
  }
};

/// \cond helpers

/// \brief Helps in the deduction of template argument.
template <typename V>
CyclicView(V &&base) -> CyclicView<std::ranges::views::all_t<V>>;

/// \endcond

// /////////////////////////////////////////////
// Implementation Detail
// /////////////////////////////////////////////

/// \cond impl_detail

namespace _detail {

/// \brief The `CyclicRangeAdaptor` class implements an adapter for the `CyclicView` class.
struct CyclicRangeAdaptor {
  /// \brief Function call operator.
  /// \param range The target range.
  /// \return A cyclic view for \p range.
  template <std::ranges::viewable_range R>
  constexpr auto operator()(R &&range) const {
    return CyclicView(std::forward<R>(range));
  }
};

/// \brief Overloaded bitwise operator to foster desired syntactic sugar.
/// \param range The target range.
/// \param adaptor A cyclic view adapter.
/// \return A cyclic view for \p range.
template <std::ranges::viewable_range R>
constexpr auto operator|(R &&range, const CyclicRangeAdaptor &adaptor) {
  return adaptor(std::forward<R>(range));
}

}

/// \endcond

// /////////////////////////////////////////////
// Views
// /////////////////////////////////////////////

namespace views {

/// \cond helper

/// \brief Helper type for `cbx::_detail::CyclicRangeAdaptor`.
using CyclicViewClosureAdapter = _detail::CyclicRangeAdaptor;

/// \endcond

/// \brief An adapter closure for cyclic views to foster desired syntactic sugar.
///
/// \note If multiple views are fused, the cyclic view must come last.
///
/// \relates cbx::CyclicView
auto cyclic = CyclicViewClosureAdapter{};

}

}

#endif
