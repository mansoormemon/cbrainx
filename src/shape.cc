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

#include "cbrainx/shape.hh"

#include <numeric>
#include <stdexcept>

#include <fmt/format.h>

namespace cbx {

auto Shape::range_check(size_type index) const -> void {
  auto dims = this->dimensions();
  if (index >= dims) {
    throw std::out_of_range{
        fmt::format("cbx::Shape::range_check: `index(={}) >= this->dimensions()(={})` is true", index, dims)};
  }
}

auto Shape::arg_count_check(size_type N) const -> void {
  auto dims = this->dimensions();
  if (N > dims) {
    throw std::invalid_argument{
        fmt::format("cbx::Shape::arg_count_check: `N(={}) > this->dimensions()(={})` is true", N, dims)};
  }
}

auto Shape::validity_check(value_type value) -> void {
  if (value <= 0) {
    throw std::invalid_argument{"cbx::Shape::validity_check: dimension size must be greater than zero"};
  }
}

template <std::input_iterator I_It>
auto Shape::validity_check(I_It first, I_It last) -> void {
  for (std::input_iterator auto it = first; it != last; ++it) {
    Shape::validity_check(*it);
  }
}

// /////////////////////////////////////////////////////////////

Shape::Shape(Shape &&other) noexcept : data_{std::move(other.data_)} {}

Shape::Shape(std::initializer_list<value_type> ilist) {
  Shape::validity_check(ilist.begin(), ilist.end());
  data_ = ilist;
}

// /////////////////////////////////////////////////////////////

auto Shape::operator=(Shape &&other) noexcept -> Shape & {
  data_ = std::move(other.data_);
  return *this;
}

auto Shape::operator=(std::initializer_list<value_type> ilist) -> Shape & {
  Shape::validity_check(ilist.begin(), ilist.end());
  data_ = ilist;
  return *this;
}

// /////////////////////////////////////////////////////////////

auto Shape::operator[](size_type index) const noexcept -> const_reference { return data_[index]; }

auto Shape::at(size_type index) const -> const_reference {
  this->range_check(index);
  return data_[index];
}

// /////////////////////////////////////////////////////////////

auto Shape::data() const noexcept -> const_pointer { return data_.data(); }

auto Shape::dimensions() const noexcept -> size_type { return data_.size(); }

auto Shape::set_dimension(size_type index, value_type value) -> Shape & {
  this->range_check(index);
  Shape::validity_check(value);
  data_[index] = value;
  return *this;
}

// /////////////////////////////////////////////////////////////

auto Shape::begin() const noexcept -> const_iterator { return data_.begin(); }

auto Shape::end() const noexcept -> const_iterator { return data_.end(); }

// /////////////////////////////////////////////////////////////

auto Shape::is_scalar() const noexcept -> bool { return this->dimensions() == 0; }

auto Shape::is_compatible(const Shape &other) const noexcept -> bool { return this->total() == other.total(); }

// /////////////////////////////////////////////////////////////

auto Shape::total() const noexcept -> size_type {
  return std::accumulate(this->begin(), this->end(), UNIT_DIMENSION_SIZE, std::multiplies());
}

// /////////////////////////////////////////////////////////////

auto Shape::swap(Shape &other) noexcept -> Shape & {
  data_.swap(other.data_);
  return *this;
}

// /////////////////////////////////////////////////////////////

auto Shape::meta_info() const noexcept -> std::string {
  return fmt::format("{{ dimensions={}, total={}, is_scalar={} }}", this->dimensions(), this->total(),
                     this->is_scalar());
}

auto Shape::to_string() const noexcept -> std::string { return fmt::format("({})", fmt::join(data_, ", ")); }

// /////////////////////////////////////////////////////////////////////////////////////////////

auto operator==(const Shape &a, const Shape &b) noexcept -> bool { return a.data_ == b.data_; }

auto operator!=(const Shape &a, const Shape &b) noexcept -> bool { return a.data_ != b.data_; }

}
