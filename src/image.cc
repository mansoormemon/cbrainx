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

#include "cbrainx/image.hh"

#include <filesystem>
#include <stdexcept>

#define STBI_ONLY_BMP
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

namespace cbx {

auto Image::Meta::to_shape() const -> Shape {
  u32 w = width, h = height, c = channels;
  return (c == Shape::UNIT_DIMENSION_SIZE) ? Shape{h, w} : Shape{h, w, c};
}

auto Image::Meta::total() const -> size_t { return width * height * channels; }

// /////////////////////////////////////////////////////////////////////////////////////////////

auto Image::Meta::decode_shape(const Shape &shape) -> Meta {
  switch (shape.dimensions()) {
    case 2: {
      auto [h, w] = shape.unwrap<2, i32>();
      return Meta{w, h, Shape::UNIT_DIMENSION_SIZE};
    }
    case 3: {
      auto [h, w, c] = shape.unwrap<3, i32>();
      return Meta{w, h, c};
    }
    default: {
      throw std::out_of_range{"cbx::Image::Meta::decode_shape: unsuitable shape"};
    }
  }
}

// /////////////////////////////////////////////////////////////////////////////////////////////

template <supported_image_datatype T>
auto Image::morph_datatype(const Tensor<T> &img) -> Tensor<datatype_after_morph_t<T>> {
  using type = datatype_after_morph_t<T>;
  auto img_iter = img.begin();
  auto morphed_img = Tensor<type>::custom(img.shape(), [&img_iter](const auto &) {
    auto val = *(img_iter++);
    return std::is_same_v<u8, type> ? val * UCHAR_MAX : val / static_cast<f32>(UCHAR_MAX);
  });
  return morphed_img;
}

// /////////////////////////////////////////////////////////////

template auto Image::morph_datatype<u8>(const Tensor<u8> &img) -> Tensor<f32>;

template auto Image::morph_datatype<f32>(const Tensor<f32> &img) -> Tensor<u8>;

// /////////////////////////////////////////////////////////////

template <supported_image_datatype T>
auto Image::read(str img_path) -> Tensor<T> {
  using pointer = typename Tensor<T>::pointer;

  auto abs_path = std::filesystem::absolute(img_path);
  auto meta = Meta{};
  void *temp_buf = {};
  // Choose function appropriately between u8 and f32 based on T.
  if constexpr (std::is_same_v<u8, T>) {
    temp_buf = stbi_load(abs_path.c_str(), &meta.width, &meta.height, &meta.channels, STBI_default);
  } else {
    temp_buf = stbi_loadf(abs_path.c_str(), &meta.width, &meta.height, &meta.channels, STBI_default);
  }
  if (not temp_buf) {
    throw std::invalid_argument{"cbx::Image::read: could not read image"};
  }
  auto img = Tensor<T>::copy(meta.to_shape(), pointer(temp_buf), pointer(temp_buf) + meta.total());
  stbi_image_free(temp_buf);
  return img;
}

// /////////////////////////////////////////////////////////////

template auto Image::read<u8>(str img_path) -> Tensor<u8>;

template auto Image::read<f32>(str img_path) -> Tensor<f32>;

// /////////////////////////////////////////////////////////////

template <>
auto Image::write<u8>(const Tensor<u8> &img, str img_path, Format fmt) -> void {
  auto abs_path = std::filesystem::absolute(img_path);
  auto meta = Meta::decode_shape(img.shape());
  switch (fmt) {
    case Format::BMP: {
      stbi_write_bmp(abs_path.c_str(), meta.width, meta.height, meta.channels, img.data());
      break;
    }
    case Format::JPG: {
      stbi_write_jpg(abs_path.c_str(), meta.width, meta.height, meta.channels, img.data(), JPG_QUALITY);
      break;
    }
    case Format::PNG: {
      stbi_write_png(abs_path.c_str(), meta.width, meta.height, meta.channels, img.data(),
                     meta.width * meta.channels);
      break;
    }
  }
}

template <>
auto Image::write<f32>(const Tensor<f32> &img, str img_path, Format fmt) -> void {
  auto morphed_img = morph_datatype(img);
  Image::write<u8>(morphed_img, img_path, fmt);
}

}
