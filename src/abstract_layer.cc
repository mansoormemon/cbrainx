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

#include <fmt/format.h>

namespace cbx {

AbstractLayer::AbstractLayer(i32 id) : id_{id} {}

AbstractLayer::AbstractLayer(const std::string &name) : name_{name} {}

AbstractLayer::AbstractLayer(i32 id, const std::string &name) : id_{id}, name_{name} {}

// /////////////////////////////////////////////////////////////

auto AbstractLayer::id() const -> i32 { return id_; }

auto AbstractLayer::set_id(i32 id) -> AbstractLayer & {
  id_ = id;
  return *this;
}

auto AbstractLayer::name() const -> std::string { return name_; }

auto AbstractLayer::set_name(const std::string &name) -> AbstractLayer & {
  name_ = name;
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
