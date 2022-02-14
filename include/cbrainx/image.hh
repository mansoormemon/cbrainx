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

#include <tuple>
#include <type_traits>

#include "tensor.hh"
#include "type_aliases.hh"

namespace cbx {

template <typename T>
concept image_datatype = std::disjunction_v<std::is_same<T, u8>, std::is_same<T, f32>>;

// /////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief The <b>Image</b> class contains functionality to interpret a tensor as an image.
 */
class Image {
 private:
  template <image_datatype T>
  struct datatype_after_morph {
    using type = std::conditional_t<std::is_same_v<u8, T>, f32, u8>;
  };

  template <image_datatype T>
  using datatype_after_morph_t = typename datatype_after_morph<T>::type;

 public:
  static constexpr auto const JPG_QUALITY = 72;

  // /////////////////////////////////////////////////////////////

  enum class Format { BMP, JPG, PNG };
  enum Channel { Mono = 0x1, Red = 0x2, Green = 0x4, Blue = 0x8, Alpha = 0x10 };
  enum Model { Gray = 0x1, GrayAlpha = 0x2, RGB = 0x4, RGBA = 0x8 };

  // /////////////////////////////////////////////////////////////

  /**
   * The <b>Meta</b> class is an adapter for switching between tensor metadata and image metadata.
   */
  class Meta {
   private:
    i32 width_ = {};
    i32 height_ = {};
    i32 channels_ = {};

   public:
    Meta() = default;

    Meta(i32 width, i32 height, i32 channels = 1);

    ~Meta() = default;

    // /////////////////////////////////////////////////////////////

    [[nodiscard]] auto width() const -> const i32 &;
    auto width() -> i32 &;

    [[nodiscard]] auto height() const -> const i32 &;
    auto height() -> i32 &;

    [[nodiscard]] auto channels() const -> const i32 &;
    auto channels() -> i32 &;

    // /////////////////////////////////////////////////////////////

    template <typename T>
    [[nodiscard]] auto unwrap() const -> std::tuple<T, T, T> {
      return std::make_tuple(static_cast<T>(width_), static_cast<T>(height_), static_cast<T>(channels_));
    }

    [[nodiscard]] auto model() const -> Model;

    [[nodiscard]] auto total() const -> std::size_t;

    [[nodiscard]] auto bitmask() const -> i32;

    [[nodiscard]] auto is_compatible(Model target) const -> bool;

    [[nodiscard]] auto has_channel(Channel channel) const -> bool;

    [[nodiscard]] auto position_of(Channel target) const -> i32;

    [[nodiscard]] auto to_shape() const -> Shape;

    // /////////////////////////////////////////////////////////////////////////////////////////////

    [[nodiscard]] static auto decode_shape(const Shape &shape) -> Meta;
  };

  // /////////////////////////////////////////////////////////////

  template <image_datatype T>
  [[nodiscard]] static auto make(const Meta &meta) -> Tensor<T> {
    return Tensor<T>{meta.to_shape()};
  }

  template <image_datatype T>
  [[nodiscard]] static auto morph_datatype(const Tensor<T> &img) -> Tensor<datatype_after_morph_t<T>>;

  // /////////////////////////////////////////////////////////////

  template <image_datatype T = u8>
  [[nodiscard]] static auto read(str img_path) -> Tensor<T>;

  template <image_datatype T>
  static auto write(const Tensor<T> &img, str img_path, Format fmt = Format::JPG) -> void;
};

}

#endif
