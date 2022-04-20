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

#include "cbrainx/image.hh"

#include <filesystem>

#define STBI_ONLY_BMP
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include "cbrainx/exceptions.hh"

namespace cbx {

// /////////////////////////////////////////////
// Constructors (and Destructors)
// /////////////////////////////////////////////

Image::Meta::Meta(size_type width, size_type height, size_type channels)
    : width_{width}, height_{height}, channels_{channels} {}

// /////////////////////////////////////////////
// Accessors and Mutators
// /////////////////////////////////////////////

auto Image::Meta::width() const noexcept -> size_type { return width_; }

auto Image::Meta::set_width(size_type new_width) -> Meta & {
  width_ = new_width;
  return *this;
}

auto Image::Meta::height() const noexcept -> size_type { return height_; }

auto Image::Meta::set_height(size_type new_height) -> Meta & {
  height_ = new_height;
  return *this;
}

auto Image::Meta::channels() const noexcept -> size_type { return channels_; }

auto Image::Meta::set_channels(size_type new_channels) -> Meta & {
  channels_ = new_channels;
  return *this;
}

// /////////////////////////////////////////////
// Query Functions
// /////////////////////////////////////////////

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
          "cbx::Image::Meta::model: color model is not recognized [channels = {}]", channels_};
    }
  }
}

auto Image::Meta::pixels() const noexcept -> size_type { return width_ * height_; }

auto Image::Meta::total() const noexcept -> size_type { return width_ * height_ * channels_; }

auto Image::Meta::bitmask() const -> u32 {
  auto color_model = model();
  switch (color_model) {
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
    default: {
      return {};
    }
  }
}

auto Image::Meta::is_compatible(Model target) const -> bool {
  auto color_model = model();
  switch (color_model) {
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
    default: {
      return false;
    }
  }
}

auto Image::Meta::has_channel(Channel channel) const -> bool { return bitmask() & channel; }

auto Image::Meta::position_of(Channel channel) const -> i32 {
  if (not has_channel(channel)) {
    return -1;
  }

  switch (channel) {
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
      return i32(channels_ - 1);
    }
  }
}

// /////////////////////////////////////////////
// Informative
// /////////////////////////////////////////////

auto Image::Meta::to_shape() const -> Shape {
  auto [w, h, c] = unwrap();
  return c > Shape::SCALAR_SIZE ? Shape{h, w, c} : Shape{h, w};
}

// /////////////////////////////////////////////////////////////
// Static Functions
// /////////////////////////////////////////////////////////////

auto Image::Meta::decode_shape(const Shape &shape) -> Meta {
  switch (shape.rank()) {
    case 2: {
      auto [h, w] = shape.unwrap<2>();
      return Meta{w, h, Shape::SCALAR_SIZE};
    }
    case 3: {
      auto [h, w, c] = shape.unwrap<3>();
      return Meta{w, h, c};
    }
    default: {
      throw ShapeError{"cbx::Image::Meta::decode_shape: shape = {} is not suitable for an image",
                       shape.to_string()};
    }
  }
}

// /////////////////////////////////////////////
// I/O Functions
// /////////////////////////////////////////////

template <BitDepth B>
auto Image::read(std::string_view img_path) -> Tensor<B> {
  using pointer = typename Tensor<B>::pointer;

  auto abs_path = std::filesystem::absolute(img_path);
  void *temp_buf = nullptr;
  i32 w = {}, h = {}, c = {};

  // Choose the appropriate function based on B.
  if constexpr (std::is_same_v<u8, B>) {
    temp_buf = stbi_load(abs_path.string().c_str(), &w, &h, &c, STBI_default);
  } else {
    temp_buf = stbi_loadf(abs_path.string().c_str(), &w, &h, &c, STBI_default);
  }
  if (not temp_buf) {
    throw ImageIOError{"cbx::Image::read: could not read image [path = {}]", img_path};
  }
  auto meta = Meta(w, h, c);
  auto img = Tensor<B>{meta.to_shape(), pointer(temp_buf)};
  stbi_image_free(temp_buf);
  return img;
}

template auto Image::read<u8>(std::string_view img_path) -> Tensor<u8>;

template auto Image::read<f32>(std::string_view img_path) -> Tensor<f32>;

template <>
auto Image::write<u8>(const Tensor<u8> &img, std::string_view img_path, Format fmt) -> void {
  auto abs_path = std::filesystem::absolute(img_path);
  auto parent_path = abs_path.parent_path();
  if (not std::filesystem::exists(parent_path)) {
    std::filesystem::create_directories(parent_path);
  }

  auto meta = Meta::decode_shape(img.shape());
  i32 ret_val = {};
  auto [w, h, c] = meta.unwrap<i32>();
  switch (fmt) {
    case Format::BMP: {
      ret_val = stbi_write_bmp(abs_path.string().c_str(), w, h, c, img.data());
      break;
    }
    case Format::JPG: {
      ret_val = stbi_write_jpg(abs_path.string().c_str(), w, h, c, img.data(), JPG_QUALITY);
      break;
    }
    case Format::PNG: {
      auto stride = w * c;
      ret_val = stbi_write_png(abs_path.string().c_str(), w, h, c, img.data(), stride);
      break;
    }
  }
  if (ret_val == 0) {
    throw ImageIOError{"cbx::Image::write: could not write image to disk [path = {}]", img_path};
  }
}

template <>
auto Image::write<f32>(const Tensor<f32> &img, std::string_view img_path, Format fmt) -> void {
  Image::write<u8>(morph(img), img_path, fmt);
}

}
