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

#include "custom_iterators.hh"
#include "type_aliases.hh"

namespace cbx {

template <std::ranges::view V>
requires std::ranges::bidirectional_range<V>
class CyclicView : public std::ranges::view_interface<CyclicView<V>> {
 public:
  using view_type = V;
  using view_iterator_type = std::ranges::iterator_t<view_type>;
  using iterator_type = CyclicIterator<view_iterator_type>;
  using difference_type = std::iter_difference_t<iterator_type>;

 private:
  view_type base_ = {};
  iterator_type iter_ = {std::begin(base_), std::end(base_)};

 public:
  CyclicView() = default;

  constexpr explicit CyclicView(view_type base)
      : base_{std::move(base)}, iter_{std::begin(base_), std::end(base_)} {}

  constexpr auto base() -> const view_type &requires std::copy_constructible<view_type> { return base; }

  constexpr auto base() -> view_type && { return std::move(base_); }

  constexpr auto begin() const -> iterator_type requires std::ranges::common_range<const view_type> {
    return iter_;
  }

  constexpr auto begin() -> iterator_type requires std::ranges::common_range<view_type> { return iter_; }

  constexpr auto end() const -> iterator_type requires std::ranges::common_range<const view_type> {
    return iterator_type{iter_.past_the_end(), iter_.past_the_end()};
  }

  constexpr auto end() -> iterator_type requires std::ranges::common_range<view_type> {
    return iterator_type{iter_.past_the_end(), iter_.past_the_end()};
  }

  constexpr auto size() requires std::ranges::sized_range<view_type> { return std::ranges::size(base_); }

  constexpr auto size() const requires std::ranges::sized_range<const view_type> {
    return std::ranges::size(base_);
  }
};

template <typename V>
CyclicView(V &&base) -> CyclicView<std::ranges::views::all_t<V>>;

namespace details {

struct CyclicRangeAdaptor {
  template <std::ranges::viewable_range R>
  constexpr auto operator()(R &&range) const {
    return CyclicView(std::forward<R>(range));
  }
};

template <std::ranges::viewable_range R>
constexpr auto operator|(R &&range, const CyclicRangeAdaptor &adaptor) {
  return adaptor(std::forward<R>(range));
}

}

namespace views {

auto cyclic = details::CyclicRangeAdaptor{};

}

}

#endif
