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

#ifndef CBRAINX__IMG_PROC_HH_
#define CBRAINX__IMG_PROC_HH_

#include <limits>
#include <type_traits>

#include "image.hh"
#include "tensor.hh"
#include "type_aliases.hh"

namespace cbx {

/**
 * @brief The <b>ImgProc</b> class contains functionality for image processing.
 */
class ImgProc {
 private:
  static auto has_channel_check(Image::Meta meta, Image::Channel target) -> void;

  static auto model_compatibility_check(Image::Meta meta, Image::Model target) -> void;

 public:
  template <image_datatype T>
  [[nodiscard]] static auto extract_channel(const Tensor<T> &img, Image::Channel channel) -> Tensor<T> {
    auto meta = Image::Meta::decode_shape(img.shape());
    ImgProc::has_channel_check(meta, channel);
    auto mono_img = Image::make<T>({meta.width(), meta.height()});
    auto src_it = img.begin() + meta.position_of(channel);
    for (auto it = mono_img.begin(), end = mono_img.end(); it != end; ++it) {
      *it = *src_it;
      src_it += meta.channels();
    }
    return mono_img;
  }

  // /////////////////////////////////////////////////////////////

  template <image_datatype T>
  [[nodiscard]] static auto grayscale(const Tensor<T> &img) -> Tensor<T> {
    auto meta = Image::Meta::decode_shape(img.shape());
    if (not meta.is_compatible(Image::Model::RGB)) {
      return img;
    }

    const auto R = 0, G = 1, B = 2;
    auto gray_img = Image::make<T>({meta.width(), meta.height()});
    auto src_it = img.begin();
    for (auto it = gray_img.begin(), end = gray_img.end(); it != end; ++it) {
      *it = (0.3 * src_it[R]) + (0.59 * src_it[G]) + (0.11 * src_it[B]);
      src_it += meta.channels();
    }
    return gray_img;
  }

  // /////////////////////////////////////////////////////////////

  template <image_datatype T>
  static auto invert(Tensor<T> &img) -> Tensor<T> & {
    const auto MAX_CHANNEL_VALUE = std::is_same_v<u8, T> ? std::numeric_limits<u8>::max() : 1.0;
    for (auto &value : img) {
      value = MAX_CHANNEL_VALUE - value;
    }
    return img;
  }

  static auto invert(Tensor<u8> &img) -> Tensor<u8> &;

  // /////////////////////////////////////////////////////////////

  template <image_datatype T>
  static auto binarize(Tensor<T> &img) -> Tensor<T> & {
    const auto MAX_CHANNEL_VALUE = std::is_same_v<u8, T> ? std::numeric_limits<u8>::max() : 1.0;
    auto pivot = MAX_CHANNEL_VALUE / 2;
    for (auto &value : img) {
      value = value > pivot ? MAX_CHANNEL_VALUE : 0;
    }
    return img;
  }

  static auto binarize(Tensor<u8> &img) -> Tensor<u8> &;

  // /////////////////////////////////////////////////////////////

  template <image_datatype T>
  [[nodiscard]] static auto resize(const Tensor<T> &img, const Image::Meta &meta) -> Tensor<T>;

  // /////////////////////////////////////////////////////////////

  template <image_datatype T>
  [[nodiscard]] static auto rescale(const Tensor<T> &img, f32 factor) -> Tensor<T>;
};

}

#endif
