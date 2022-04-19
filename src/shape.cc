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

#include "cbrainx/shape.hh"

#include <numeric>

#include <fmt/format.h>

#include "cbrainx/exceptions.hh"

namespace cbx {

// /////////////////////////////////////////////
// Helpers
// /////////////////////////////////////////////

auto Shape::_m_check_bounds(size_type index) const -> void {
  auto cur_rank = rank();
  if (index >= cur_rank) {
    throw IndexOutOfBoundsError{"cbx::Shape::_m_check_bounds: index = {} >= this->rank() = {}", index, cur_rank};
  }
}

auto Shape::_m_check_rank(size_type N) const -> void {
  auto cur_rank = rank();
  if (N > cur_rank) {
    throw RankError{"cbx::Shape::_m_check_rank: N = {} > this->rank() = {}", N, cur_rank};
  }
}

auto Shape::_s_validate_dimension(value_type value) -> void {
  if (value == 0) {
    throw ValueError{"cbx::Shape::_s_validate_dimension: dimension can not be equal than zero", value};
  }
}

// /////////////////////////////////////////////
// Constructors (and Destructors)
// /////////////////////////////////////////////

Shape::Shape(Shape &&other) noexcept : data_{std::move(other.data_)} {}

Shape::Shape(std::initializer_list<value_type> ilist) {
  _validate_dimensions(ilist.begin(), ilist.end());
  data_ = ilist;
}

// /////////////////////////////////////////////
// Assignment Operator(s)
// /////////////////////////////////////////////

auto Shape::operator=(Shape &&other) noexcept -> Shape & {
  data_ = std::move(other.data_);
  return *this;
}

// /////////////////////////////////////////////
// Element Access
// /////////////////////////////////////////////

auto Shape::operator[](size_type index) const noexcept -> const_reference { return data_[index]; }

auto Shape::at(size_type index) const -> const_reference {
  _m_check_bounds(index);
  return data_[index];
}

auto Shape::front() const -> const_reference { return data_.front(); }

auto Shape::back() const -> const_reference { return data_.back(); }

// /////////////////////////////////////////////
// Accessors and Mutators
// /////////////////////////////////////////////

auto Shape::underlying_container() const noexcept -> const container & { return data_; }

auto Shape::rank() const noexcept -> size_type { return data_.size(); }

auto Shape::set_axis(size_type index, value_type value) -> Shape & {
  _m_check_bounds(index);
  _s_validate_dimension(value);
  data_[index] = value;
  return *this;
}

// /////////////////////////////////////////////
// Iterators
// /////////////////////////////////////////////

auto Shape::cbegin() const noexcept -> const_iterator { return data_.cbegin(); }

auto Shape::begin() const noexcept -> const_iterator { return data_.begin(); }

auto Shape::crbegin() const noexcept -> const_reverse_iterator { return data_.crbegin(); }

auto Shape::rbegin() const noexcept -> const_reverse_iterator { return data_.rbegin(); }

auto Shape::cend() const noexcept -> const_iterator { return data_.cend(); }

auto Shape::end() const noexcept -> const_iterator { return data_.end(); }

auto Shape::crend() const noexcept -> const_reverse_iterator { return data_.crend(); }

auto Shape::rend() const noexcept -> const_reverse_iterator { return data_.rend(); }

// /////////////////////////////////////////////
// Query Functions
// /////////////////////////////////////////////

auto Shape::is_equivalent(const Shape &other) const noexcept -> bool { return total() == other.total(); }

auto Shape::total() const noexcept -> size_type {
  return std::accumulate(begin(), end(), SCALAR_SIZE, std::multiplies{});
}

// /////////////////////////////////////////////
// Informative
// /////////////////////////////////////////////

auto Shape::meta_info() const noexcept -> std::string {
  return fmt::format("{{ rank={}, total={} }}", rank(), total());
}

auto Shape::to_string() const noexcept -> std::string { return fmt::format("({})", fmt::join(data_, ", ")); }

// /////////////////////////////////////////////
// Capacity
// /////////////////////////////////////////////

auto Shape::resize(size_type new_rank, bool modify_front) -> Shape & {
  if (modify_front) {
    auto cur_rank = rank();
    auto diff = isize(new_rank - cur_rank);
    if (diff > 0) {
      data_.insert(data_.begin(), diff, SCALAR_SIZE);
    } else {
      data_.erase(data_.begin(), data_.begin() - diff);
    }
  } else {
    data_.resize(new_rank, SCALAR_SIZE);
  }
  return *this;
}

auto Shape::swap(Shape &other) noexcept -> Shape & {
  data_.swap(other.data_);
  return *this;
}

// /////////////////////////////////////////////
// Utility
// /////////////////////////////////////////////

auto Shape::clone() const -> Shape { return *this; }

// /////////////////////////////////////////////////////////////
// External Functions
// /////////////////////////////////////////////////////////////

// /////////////////////////////////////////////
// Equality Operators
// /////////////////////////////////////////////

auto operator==(const Shape &a, const Shape &b) noexcept -> bool {
  return a.underlying_container() == b.underlying_container();
}

auto operator!=(const Shape &a, const Shape &b) noexcept -> bool {
  return a.underlying_container() != b.underlying_container();
}

}
