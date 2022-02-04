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

#ifndef CBRAINX__IMAGE_HH_
#define CBRAINX__IMAGE_HH_

#include <type_traits>

#include "tensor.hh"
#include "type_aliases.hh"

namespace cbx {

template <typename T>
concept supported_image_datatype = std::disjunction_v<std::is_same<T, u8>, std::is_same<T, f32>>;

// /////////////////////////////////////////////////////////////////////////////////////////////

class Image {
 private:
  template <supported_image_datatype T>
  struct datatype_after_morph {
    using type = std::conditional_t<std::is_same_v<u8, T>, f32, u8>;
  };

  template <supported_image_datatype T>
  using datatype_after_morph_t = typename datatype_after_morph<T>::type;

  // /////////////////////////////////////////////////////////////

  struct Meta {
    i32 width = {};
    i32 height = {};
    i32 channels = {};

    [[nodiscard]] auto to_shape() const -> Shape;

    [[nodiscard]] auto total() const -> size_t;

    // /////////////////////////////////////////////////////////////////////////////////////////////

    [[nodiscard]] static auto decode_shape(const Shape &shape) -> Meta;
  };

 public:
  static constexpr auto const JPG_QUALITY = 96;

  // /////////////////////////////////////////////////////////////

  enum class Format { BMP, JPG, PNG };
  enum class Model { Mono, Gray, RGB, RGBA };

  // /////////////////////////////////////////////////////////////

  template <supported_image_datatype T>
  [[nodiscard]] static auto morph_datatype(const Tensor<T> &img) -> Tensor<datatype_after_morph_t<T>>;

  // /////////////////////////////////////////////////////////////

  template <supported_image_datatype T = f32>
  [[nodiscard]] static auto read(str img_path) -> Tensor<T>;

  template <supported_image_datatype T>
  static auto write(const Tensor<T> &img, str img_path, Format fmt = Format::PNG) -> void;
};

}

#endif
