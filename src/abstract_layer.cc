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

#include "cbrainx/abstract_layer.hh"

#include <utility>

#include <fmt/format.h>

namespace cbx {

AbstractLayer::AbstractLayer(AbstractLayer &&other) noexcept
    : id_{std::exchange(other.id_, {})}, name_{std::move(other.name_)} {}

AbstractLayer::AbstractLayer(i32 id) : id_{id} {}

AbstractLayer::AbstractLayer(std::string name) : name_{std::move(name)} {}

AbstractLayer::AbstractLayer(i32 id, std::string name) : id_{id}, name_{std::move(name)} {}

// /////////////////////////////////////////////////////////////

auto AbstractLayer::operator=(AbstractLayer &&other) noexcept -> AbstractLayer & {
  id_ = std::exchange(other.id_, {});
  name_ = std::move(other.name_);
  return *this;
}

// /////////////////////////////////////////////////////////////

auto AbstractLayer::id() const -> i32 { return id_; }

auto AbstractLayer::set_id(i32 id) -> AbstractLayer & {
  id_ = id;
  return *this;
}

auto AbstractLayer::name() const -> std::string { return name_; }

auto AbstractLayer::set_name(std::string name) -> AbstractLayer & {
  name_ = std::move(name);
  return *this;
}

// /////////////////////////////////////////////////////////////

auto AbstractLayer::to_string() const -> std::string { return fmt::format("{}{}", name_, id_); }

[[nodiscard]] auto AbstractLayer::type_name() const -> std::string {
  auto layer = this->type();
  switch (layer) {
    case LayerType::Dense: {
      return "Dense";
    }
    case LayerType::Activation: {
      return "Activation";
    }
    case LayerType::SoftMax: {
      return "SoftMax";
    }
  }
  return {};
}

}
