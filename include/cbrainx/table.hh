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

#ifndef CBRAINX__TABLE_HH_
#define CBRAINX__TABLE_HH_

#include <initializer_list>
#include <list>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "type_aliases.hh"

namespace cbx {

class [[deprecated]] Table {
 public:
  using item_type = std::string;

  using item_reference = item_type &;
  using item_pointer = item_type *;

  using item_const_reference = const item_type &;
  using item_const_pointer = const item_type *;

  using row_type = std::vector<item_type>;
  using container = std::list<row_type>;

  using size_type = container::size_type;
  using difference_type = container::difference_type;

  using iterator = container::iterator;
  using const_iterator = container::const_iterator;

  // /////////////////////////////////////////////////////////////

  enum Width { Small = 12, Medium = 24, Large = 36, XLarge = 48 };

 private:
  item_type caption_ = {};
  row_type header_ = {};
  container data_ = {};

  // /////////////////////////////////////////////////////////////

  auto range_check(size_type index) const -> void;

 public:
  Table() = default;

  Table(const Table &other) = default;

  Table(Table &&other) noexcept;

  Table(std::initializer_list<item_type> header);

  ~Table() = default;

  // /////////////////////////////////////////////////////////////

  auto operator=(const Table &other) -> Table & = default;

  auto operator=(Table &&other) noexcept -> Table &;

  // /////////////////////////////////////////////////////////////

  [[nodiscard]] auto columns() const noexcept -> size_type;

  [[nodiscard]] auto rows() const noexcept -> size_type;

  auto set_caption(item_type caption) -> Table &;

  auto set_header(item_type header_column, size_type index) -> Table &;

  auto override_header(std::initializer_list<item_type> header) -> Table &;

  auto add(std::initializer_list<item_type> row) -> Table &;

  // /////////////////////////////////////////////////////////////

  [[nodiscard]] auto begin() const noexcept -> const_iterator;

  [[nodiscard]] auto end() const noexcept -> const_iterator;

  // /////////////////////////////////////////////////////////////

  [[nodiscard]] auto is_header_set() const noexcept -> bool;

  [[nodiscard]] auto is_empty() const noexcept -> bool;

  // /////////////////////////////////////////////////////////////

  auto clear() -> Table &;

  // /////////////////////////////////////////////////////////////

  [[nodiscard]] auto meta_info() const noexcept -> std::string;

  auto show(bool print_caption = false, Width col_width = Width::Medium) -> void;
};

}

#endif
