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

#ifndef CBRAINX__TYPE_CONCEPTS_HH_
#define CBRAINX__TYPE_CONCEPTS_HH_

#include <type_traits>

#include "typeAliases.hh"

namespace cbx {

template <typename T>
concept Bool = std::is_same_v<bool, T>;

template <typename T>
concept Integer = std::is_integral_v<T>;

template <typename T>
concept Float = std::is_floating_point_v<T>;

template <typename T>
concept Number = std::is_arithmetic_v<T>;

template <typename T>
concept Void = std::is_void_v<T>;

template <typename T>
concept NullaryOperation = true;

template <typename T>
concept UnaryOperation = true;

template <typename T>
concept BinaryOperation = true;

}

#endif
