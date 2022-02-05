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

#include <string>
#include <tuple>
#include <vector>

#include "type_aliases.hh"

namespace cbx {

/**
 * @brief The <b>Shape</b> class is a convenient structure to represent the shape of a tensor.
 */
class Shape {
 public:
  using container = std::vector<u32>;

  using value_type = container::value_type;

  using reference = container::reference;
  using const_reference = container::const_reference;

  using pointer = container::pointer;
  using const_pointer = container::const_pointer;

  using size_type = container::size_type;
  using difference_type = container::difference_type;

  using iterator = container::iterator;
  using const_iterator = container::const_iterator;

  // /////////////////////////////////////////////////////////////

  static constexpr size_type UNIT_DIMENSION_SIZE = 1;

 private:
  container data_ = {};

  // /////////////////////////////////////////////////////////////

  auto range_check(size_type index) const -> void;

  auto arg_count_check(size_type N) const -> void;

  static auto validity_check(value_type value) -> void;

  template <std::input_iterator I_It>
  static auto validity_check(I_It first, I_It last) -> void;

 public:
  Shape() = default;

  Shape(const Shape &other) = default;

  Shape(Shape &&other) noexcept;

  Shape(std::initializer_list<value_type> ilist);

  ~Shape() = default;

  // /////////////////////////////////////////////////////////////

  auto operator=(const Shape &other) -> Shape & = default;

  auto operator=(Shape &&other) noexcept -> Shape &;

  auto operator=(std::initializer_list<value_type> ilist) -> Shape &;

  // /////////////////////////////////////////////////////////////

  [[nodiscard]] auto operator[](size_type index) const noexcept -> const_reference;

  [[nodiscard]] auto at(size_type index) const -> const_reference;

  // /////////////////////////////////////////////////////////////

  [[nodiscard]] auto data() const noexcept -> const_pointer;

  [[nodiscard]] auto dimensions() const noexcept -> size_type;

  auto set_dimension(size_type index, value_type value) -> Shape &;

  // /////////////////////////////////////////////////////////////

  [[nodiscard]] auto begin() const noexcept -> const_iterator;

  [[nodiscard]] auto end() const noexcept -> const_iterator;

  // /////////////////////////////////////////////////////////////

  [[nodiscard]] auto is_scalar() const noexcept -> bool;

  [[nodiscard]] auto is_compatible(const Shape &other) const noexcept -> bool;

  // /////////////////////////////////////////////////////////////

  [[nodiscard]] auto total() const noexcept -> size_type;

  // /////////////////////////////////////////////////////////////

  auto swap(Shape &other) noexcept -> Shape &;

 private:
  template <typename T, size_type... Indices>
  constexpr auto unwrap_helper(std::index_sequence<Indices...>) const {
    return std::make_tuple(static_cast<T>(data_[Indices])...);
  }

 public:
  template <size_type N, typename T = value_type>
  constexpr auto unwrap() const {
    this->arg_count_check(N);
    return unwrap_helper<T>(std::make_index_sequence<N>());
  }

  // /////////////////////////////////////////////////////////////

  [[nodiscard]] auto meta_info() const noexcept -> std::string;

  [[nodiscard]] auto to_string() const noexcept -> std::string;
};

// /////////////////////////////////////////////////////////////////////////////////////////////

[[nodiscard]] auto operator==(const Shape &a, const Shape &b) noexcept -> bool;

[[nodiscard]] auto operator!=(const Shape &a, const Shape &b) noexcept -> bool;

}

#endif
