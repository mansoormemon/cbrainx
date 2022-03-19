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

#ifndef CBRAINX__DATASET_HH_
#define CBRAINX__DATASET_HH_

#include "tensor.hh"
#include "type_aliases.hh"

namespace cbx {

class Dataset {
 public:
  using data_value_type = f32;

  using data_container = Tensor<data_value_type>;

  using data_container_reference = data_container &;
  using data_container_const_reference = const data_container &;

  using data_container_pointer = data_container *;
  using data_container_const_pointer = const data_container *;

  using data_container_iterator = typename data_container::iterator;
  using data_container_const_iterator = typename data_container::const_iterator;

  // /////////////////////////////////////////////////////////////

  using targets_value_type = f32;

  using targets_container = Tensor<targets_value_type>;

  using targets_container_reference = targets_container &;
  using targets_container_const_reference = const targets_container &;

  using targets_container_pointer = targets_container *;
  using targets_container_const_pointer = const targets_container *;

  using targets_container_iterator = typename targets_container::iterator;
  using targets_container_const_iterator = typename targets_container::const_iterator;

  // /////////////////////////////////////////////////////////////

  using size_type = typename data_container::size_type;
  using difference_type = typename data_container::difference_type;

 private:
  data_container data_ = {};
  targets_container targets_ = {};

 public:
  Dataset() = default;

  Dataset(const Dataset &other) = default;

  Dataset(Dataset &&other) noexcept;

  Dataset(const data_container &data, const targets_container &targets);

  Dataset(data_container &&data, targets_container &&targets);

  ~Dataset() = default;

  // /////////////////////////////////////////////////////////////

  auto operator=(const Dataset &other) -> Dataset & = default;

  auto operator=(Dataset &&other) noexcept -> Dataset &;

  // /////////////////////////////////////////////////////////////

  [[nodiscard]] auto data() const noexcept -> data_container_const_reference;

  [[nodiscard]] auto targets() const noexcept -> targets_container_const_reference;

  [[nodiscard]] auto samples() const noexcept -> size_type;

  // /////////////////////////////////////////////////////////////

  [[nodiscard]] auto meta_info() const -> std::string;
};

}

#endif
