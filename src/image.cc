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

#define STBI_ONLY_BMP
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <fmt/format.h>

#include "cbrainx/exceptions.hh"

namespace cbx {

Image::Meta::Meta(i32 width, i32 height, i32 channels) : width_{width}, height_{height}, channels_{channels} {}

// /////////////////////////////////////////////////////////////

auto Image::Meta::width() const -> const i32 & { return width_; }

auto Image::Meta::width() -> i32 & { return width_; }

auto Image::Meta::height() const -> const i32 & { return height_; }

auto Image::Meta::height() -> i32 & { return height_; }

auto Image::Meta::channels() const -> const i32 & { return channels_; }

auto Image::Meta::channels() -> i32 & { return channels_; }

// /////////////////////////////////////////////////////////////

auto Image::Meta::model() const -> Model {
  switch (channels_) {
    case 1: {
      return Model::Gray;
    }
    case 2: {
      return Model::GrayAlpha;
    }
    case 3: {
      return Model::RGB;
    }
    case 4: {
      return Model::RGBA;
    }
    default: {
      throw UnrecognizedColorModelError{
          fmt::format("cbx::Image::Meta::model: color model is not recognized(channels={})", channels_)};
    }
  }
}

auto Image::Meta::total() const -> usize { return width_ * height_ * channels_; }

auto Image::Meta::bitmask() const -> i32 {
  auto model = this->model();
  switch (model) {
    case Model::Gray: {
      return Channel::Mono;
    }
    case Model::GrayAlpha: {
      return Channel::Mono | Channel::Alpha;
    }
    case Model::RGB: {
      return Channel::Red | Channel::Green | Channel::Blue;
    }
    case Model::RGBA: {
      return Channel::Red | Channel::Green | Channel::Blue | Channel::Alpha;
    }
  }
  return {};
}

auto Image::Meta::is_compatible(Model target) const -> bool {
  auto model = this->model();
  switch (model) {
    case Model::Gray: {
      return target & Model::Gray;
    }
    case Model::GrayAlpha: {
      return target & (Model::Gray | Model::GrayAlpha);
    }
    case Model::RGB: {
      return target & Model::RGB;
    }
    case Model::RGBA: {
      return target & (Model::RGB | Model::RGBA);
    }
  }
  return false;
}

auto Image::Meta::has_channel(Channel channel) const -> bool { return this->bitmask() & channel; }

auto Image::Meta::position_of(Channel target) const -> i32 {
  if (not this->has_channel(target)) {
    return -1;
  }

  switch (target) {
    case Channel::Mono:
    case Channel::Red: {
      return 0;
    }
    case Channel::Green: {
      return 1;
    }
    case Channel::Blue: {
      return 2;
    }
    default: {
      return channels_ - 1;
    }
  }
}

auto Image::Meta::to_shape() const -> Shape {
  auto [w, h, c] = this->unwrap<Shape::value_type>();
  return (c == Shape::SCALAR_SIZE) ? Shape{h, w} : Shape{h, w, c};
}

// /////////////////////////////////////////////////////////////////////////////////////////////

auto Image::Meta::decode_shape(const Shape &shape) -> Meta {
  switch (shape.rank()) {
    case 2: {
      auto [h, w] = shape.unwrap<2, i32>();
      return Meta{w, h, Shape::SCALAR_SIZE};
    }
    case 3: {
      auto [h, w, c] = shape.unwrap<3, i32>();
      return Meta{w, h, c};
    }
    default: {
      throw ShapeError{"cbx::Image::Meta::decode_shape: unsuitable shape for image"};
    }
  }
}

// /////////////////////////////////////////////////////////////////////////////////////////////

template <image_datatype T>
auto Image::morph_datatype(const Tensor<T> &img) -> Tensor<datatype_after_morph_t<T>> {
  using type = datatype_after_morph_t<T>;

  auto img_iter = img.begin();
  auto morphed_img = Tensor<type>::custom(img.shape(), [&img_iter]() {
    auto val = *(img_iter++);
    return std::is_same_v<u8, type> ? val * UCHAR_MAX : val / static_cast<f32>(UCHAR_MAX);
  });
  return morphed_img;
}

// /////////////////////////////////////////////////////////////

template auto Image::morph_datatype<u8>(const Tensor<u8> &img) -> Tensor<f32>;

template auto Image::morph_datatype<f32>(const Tensor<f32> &img) -> Tensor<u8>;

// /////////////////////////////////////////////////////////////

template <image_datatype T>
auto Image::read(const std::string &img_path) -> Tensor<T> {
  using pointer = typename Tensor<T>::pointer;

  auto abs_path = std::filesystem::absolute(img_path);
  auto meta = Meta{};
  void *temp_buf = {};
  // Choose appropriate function between u8 and f32 based on T.
  if constexpr (std::is_same_v<u8, T>) {
    temp_buf =
        stbi_load(abs_path.string().c_str(), &meta.width(), &meta.height(), &meta.channels(), STBI_default);
  } else {
    temp_buf =
        stbi_loadf(abs_path.string().c_str(), &meta.width(), &meta.height(), &meta.channels(), STBI_default);
  }
  if (not temp_buf) {
    throw ImageIOError{"cbx::Image::read: could not read image"};
  }
  auto img = Tensor<T>::copy(meta.to_shape(), pointer(temp_buf), pointer(temp_buf) + meta.total());
  stbi_image_free(temp_buf);
  return img;
}

// /////////////////////////////////////////////////////////////

template auto Image::read<u8>(const std::string &img_path) -> Tensor<u8>;

template auto Image::read<f32>(const std::string &img_path) -> Tensor<f32>;

// /////////////////////////////////////////////////////////////

template <>
auto Image::write<u8>(const Tensor<u8> &img, const std::string &img_path, Format fmt) -> void {
  auto abs_path = std::filesystem::absolute(img_path);
  auto parent_path = abs_path.parent_path();
  if (not std::filesystem::exists(parent_path)) {
    std::filesystem::create_directories(parent_path);
  }

  auto meta = Meta::decode_shape(img.shape());
  i32 ret_val = {};
  switch (fmt) {
    case Format::BMP: {
      ret_val =
          stbi_write_bmp(abs_path.string().c_str(), meta.width(), meta.height(), meta.channels(), img.data());
      break;
    }
    case Format::JPG: {
      ret_val = stbi_write_jpg(abs_path.string().c_str(), meta.width(), meta.height(), meta.channels(),
                               img.data(), JPG_QUALITY);
      break;
    }
    case Format::PNG: {
      ret_val = stbi_write_png(abs_path.string().c_str(), meta.width(), meta.height(), meta.channels(),
                               img.data(), meta.width() * meta.channels());
      break;
    }
  }
  if (ret_val == 0) {
    throw ImageIOError{"cbx::Image::write: could not write image"};
  }
}

template <>
auto Image::write<f32>(const Tensor<f32> &img, const std::string &img_path, Format fmt) -> void {
  auto morphed_img = morph_datatype(img);
  Image::write<u8>(morphed_img, img_path, fmt);
}

}
