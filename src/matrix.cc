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

#include "cbrainx/matrix.hh"

#include <fmt/format.h>

namespace cbx {

auto Matrix::rank_check(Shape::size_type rank) -> void {
  if (rank != DIMENSIONS) {
    throw std::logic_error{fmt::format("cbx::Matrix::rank_check: incompatible dimensionality(rank={})", rank)};
  }
}

auto Matrix::shape_equality_check(const Shape &a, const Shape &b) -> void {
  if (a != b) {
    throw std::invalid_argument{"cbx::Matrix::shape_equality_check: `a != b` is true"};
  }
}

}
