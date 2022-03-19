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

#include "cbrainx/dataset.hh"

#include <fmt/format.h>

#include <utility>

namespace cbx {

Dataset::Dataset(Dataset &&other) noexcept
    : data_{std::move(other.data_)}, targets_{std::move(other.targets_)} {}

Dataset::Dataset(const data_container &data, const targets_container &targets)
    : data_{data}, targets_{targets} {}

Dataset::Dataset(data_container &&data, targets_container &&targets)
    : data_{std::move(data)}, targets_{std::move(targets)} {}

// /////////////////////////////////////////////////////////////

auto Dataset::operator=(Dataset &&other) noexcept -> Dataset & {
  data_ = std::move(other.data_);
  targets_ = std::move(other.targets_);
  return *this;
}

// /////////////////////////////////////////////////////////////

auto Dataset::data() const noexcept -> data_container_const_reference { return data_; }

auto Dataset::targets() const noexcept -> targets_container_const_reference { return targets_; }

auto Dataset::samples() const noexcept -> size_type {
  auto [sample_count] = targets_.shape().template unwrap<1>();
  return sample_count;
}

// /////////////////////////////////////////////////////////////

auto Dataset::meta_info() const -> std::string {
  return fmt::format("{{ samples={}, data={{ total={}, shape={} }}, targets={{ total={}, shape={} }} }}",
                     this->samples(), data_.total(), data_.shape().to_string(), targets_.total(),
                     targets_.shape().to_string());
}

}
