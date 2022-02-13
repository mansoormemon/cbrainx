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

#include "cbrainx/table.hh"

#include <fmt/format.h>

namespace cbx {

auto Table::range_check(size_type index) const -> void {
  auto cols = this->columns();
  if (index >= cols) {
    throw std::out_of_range{
        fmt::format("cbx::Table::range_check: `index(={}) >= this->columns()(={})` is true", index, cols)};
  }
}

// /////////////////////////////////////////////////////////////

Table::Table(Table &&other) noexcept
    : caption_{std::move(other.caption_)}, header_{std::move(other.header_)}, data_{std::move(other.data_)} {}

Table::Table(std::initializer_list<item_type> header) : header_{header} {}

// /////////////////////////////////////////////////////////////

auto Table::operator=(Table &&other) noexcept -> Table & {
  caption_ = std::move(other.caption_);
  header_ = std::move(other.header_);
  data_ = std::move(other.data_);
  return *this;
}

// /////////////////////////////////////////////////////////////

auto Table::columns() const noexcept -> size_type { return header_.size(); }

auto Table::rows() const noexcept -> size_type { return data_.size(); }

auto Table::set_caption(item_const_reference caption) -> Table & {
  caption_ = caption;
  return *this;
}

auto Table::set_header(item_const_reference header_column, size_type index) -> Table & {
  this->range_check(index);
  header_[index] = header_column;
  return *this;
}

auto Table::override_header(std::initializer_list<item_type> header) -> Table & {
  header_ = header;
  return *this;
}

auto Table::add(std::initializer_list<item_type> row) -> Table & {
  data_.emplace_back(row);
  return *this;
}

// /////////////////////////////////////////////////////////////

auto Table::begin() const noexcept -> const_iterator { return data_.begin(); }

auto Table::end() const noexcept -> const_iterator { return data_.end(); }

// /////////////////////////////////////////////////////////////

auto Table::is_header_set() const noexcept -> bool { return not header_.empty(); }

auto Table::is_empty() const noexcept -> bool { return not data_.empty(); }

// /////////////////////////////////////////////////////////////

auto Table::clear() -> Table & {
  data_.clear();
  return *this;
}

// /////////////////////////////////////////////////////////////

auto Table::meta_info() const noexcept -> std::string {
  return fmt::format("rows={}, header={{{}}}", this->rows(), fmt::join(header_, ", "));
}

auto Table::show(bool print_caption, Width col_width) -> void {
  if (not this->is_header_set()) {
    return;
  }

  auto table_width = this->columns() * col_width;

  auto print_formatted_row = [this, col_width](const auto &row) {
    for (size_type it = {}; it < this->columns(); ++it) {
      fmt::print("{:{}}", row[it], col_width);
    }
    fmt::print("\n");
  };

  auto add_separator = [table_width](char c = '-') {
    fmt::print("{}\n", std::string(table_width, c));
  };

  if (print_caption) {
    fmt::print("{:^{}}\n", caption_, table_width);
  }
  add_separator('=');
  print_formatted_row(header_);
  add_separator('+');
  for (auto it = data_.begin(), end = data_.end(); it != end; ++it) {
    if (it != data_.begin()) {
      add_separator();
    }
    print_formatted_row(*it);
  }
  add_separator('=');
}

}
