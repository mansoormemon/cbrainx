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

auto Shape::check_bounds(size_type index) const -> void {
  auto rank = this->rank();
  if (index >= rank) {
    custom_throw<RankError>("cbx::Shape::check_bounds: `index(={}) >= this->rank()(={})` is true", index, rank);
  }
}

auto Shape::check_rank(size_type N) const -> void {
  auto rank = this->rank();
  if (N > rank) {
    custom_throw<RankError>("cbx::Shape::check_rank: `N(={}) > this->rank()(={})` is true", N, rank);
  }
}

auto Shape::validate_input(value_type value) -> void {
  if (value <= 0) {
    custom_throw<ShapeError>("cbx::Shape::validate_input: axis size must be greater than zero");
  }
}

// /////////////////////////////////////////////////////////////

Shape::Shape(Shape &&other) noexcept : data_{std::move(other.data_)} {}

Shape::Shape(std::initializer_list<value_type> ilist) {
  Shape::validate_input(ilist.begin(), ilist.end());
  data_ = ilist;
}

// /////////////////////////////////////////////////////////////

auto Shape::operator=(Shape &&other) noexcept -> Shape & {
  data_ = std::move(other.data_);
  return *this;
}

// /////////////////////////////////////////////////////////////

auto Shape::operator[](size_type index) const noexcept -> const_reference { return data_[index]; }

auto Shape::at(size_type index) const -> const_reference {
  this->check_bounds(index);
  return data_[index];
}

// /////////////////////////////////////////////////////////////

auto Shape::data() const noexcept -> const_pointer { return data_.data(); }

auto Shape::underlying_container() const noexcept -> const container & { return data_; }

auto Shape::rank() const noexcept -> size_type { return data_.size(); }

auto Shape::set_axis(size_type index, value_type value) -> Shape & {
  this->check_bounds(index);
  Shape::validate_input(value);
  data_[index] = value;
  return *this;
}

// /////////////////////////////////////////////////////////////

auto Shape::front() const -> const_reference { return data_.front(); }

auto Shape::back() const -> const_reference { return data_.back(); }

// /////////////////////////////////////////////////////////////

auto Shape::begin() const noexcept -> const_iterator { return data_.begin(); }

auto Shape::rbegin() const noexcept -> const_reverse_iterator { return data_.rbegin(); }

auto Shape::end() const noexcept -> const_iterator { return data_.end(); }

auto Shape::rend() const noexcept -> const_reverse_iterator { return data_.rend(); }

// /////////////////////////////////////////////////////////////

auto Shape::is_scalar() const noexcept -> bool { return this->rank() == SCALAR_RANK; }

auto Shape::is_vector() const noexcept -> bool { return this->rank() == VECTOR_RANK; }

auto Shape::is_matrix() const noexcept -> bool { return this->rank() == MATRIX_RANK; }

auto Shape::is_equivalent(const Shape &other) const noexcept -> bool { return this->total() == other.total(); }

// /////////////////////////////////////////////////////////////

auto Shape::total() const noexcept -> size_type {
  return std::accumulate(this->begin(), this->end(), SCALAR_SIZE, std::multiplies());
}

// /////////////////////////////////////////////////////////////

auto Shape::resize(size_type rank) -> Shape & {
  data_.resize(rank, SCALAR_SIZE);
  return *this;
}

auto Shape::swap(Shape &other) noexcept -> Shape & {
  data_.swap(other.data_);
  return *this;
}

// /////////////////////////////////////////////////////////////

auto Shape::clone() const -> Shape { return *this; }

// /////////////////////////////////////////////////////////////

auto Shape::meta_info() const noexcept -> std::string {
  return fmt::format("{{ rank={}, total={} }}", this->rank(), this->total());
}

auto Shape::to_string() const noexcept -> std::string { return fmt::format("({})", fmt::join(data_, ", ")); }

// /////////////////////////////////////////////////////////////////////////////////////////////

auto operator==(const Shape &a, const Shape &b) noexcept -> bool { return a.data_ == b.data_; }

auto operator!=(const Shape &a, const Shape &b) noexcept -> bool { return a.data_ != b.data_; }

}
