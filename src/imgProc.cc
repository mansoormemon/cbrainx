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

#include "cbrainx/imgProc.hh"

#include <array>

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb_image_resize.h>

#include "cbrainx/exceptions.hh"

namespace cbx {

// /////////////////////////////////////////////
// Implementation Detail
// /////////////////////////////////////////////

namespace _detail {

// /////////////////////
// Bit depth: `u8`
// /////////////////////

template <>
constexpr auto limits<u8>::bits() noexcept -> size_type {
  return sizeof(u8) * CHAR_BIT;
}

template <>
constexpr auto limits<u8>::min() noexcept -> u8 {
  return std::numeric_limits<u8>::min();
};

template <>
constexpr auto limits<u8>::max() noexcept -> u8 {
  return std::numeric_limits<u8>::max();
};

// /////////////////////
// Bit depth: `f32`
// /////////////////////

template <>
constexpr auto limits<f32>::bits() noexcept -> size_type {
  return sizeof(f32) * CHAR_BIT;
}

template <>
constexpr auto limits<f32>::min() noexcept -> f32 {
  return 0.0F;
};

template <>
constexpr auto limits<f32>::max() noexcept -> f32 {
  return 1.0F;
};

}

// /////////////////////////////////////////////
// Helpers
// /////////////////////////////////////////////

auto ImgProc::_s_has_channel_check(Image::Meta meta, Image::Channel channel) -> void {
  if (not meta.has_channel(channel)) {
    throw IncompatibleColorModelError{
        "cbx::ImgProc::_s_has_channel_check: channel = {} is not present in the image", channel};
  }
}

// /////////////////////////////////////////////
// Core Functionality
// /////////////////////////////////////////////

template <>
auto ImgProc::invert(Tensor<u8> &img) noexcept -> Tensor<u8> & {
  const auto MAX_VALUE = _detail::limits<u8>::max();
  const auto CHANNEL_SIZE = MAX_VALUE + 1;

  // Generate an inverted color lookup table.
  auto make_lookup_table = []() {
    auto table = std::array<u8, CHANNEL_SIZE>{};
    std::generate(table.begin(), table.end(), [n = MAX_VALUE]() mutable {
      return n--;
    });
    return table;
  };

  // Assign values from the lookup table in constant time.
  auto lookup_table = make_lookup_table();
  return img |= [&lookup_table](auto value) {
    return lookup_table[value];
  };
}

template <>
auto ImgProc::invert(Tensor<f32> &img) noexcept -> Tensor<f32> & {
  return img |= [](auto value) {
    return _detail::limits<f32>::max() - value;
  };
}

template <>
auto ImgProc::binarize(Tensor<u8> &img) noexcept -> Tensor<u8> & {
  // Algorithm: Otsu's Method
  // Otsu's thresholding method involves iterating through all the possible thresholds and calculating a
  // measure of spread for the pixel intensities in the foreground and background. The aim is to find a
  // threshold value where the sum of the foreground and background spreads is minimum. For this, we can
  // calculate the 'Between-Class Variance' of the image to find the optimal threshold.
  //
  // Formula: 𝜎B² = Wb * Wf (μb - μf)²
  //
  // where:
  //  𝜎B - Between Class Variance (BCV)
  //  Wb - Weight (background)
  //  Wf - Weight (foreground)
  //  μb - Mean (background)
  //  μf - Mean (foreground)
  //
  // The desired threshold corresponds to the maximum 𝜎B².
  //
  // Reference:
  // 1. http://www.labbookpages.co.uk/software/imgProc/otsuThreshold.html

  const auto MAX_VALUE = _detail::limits<u8>::max(), MIN_VALUE = _detail::limits<u8>::min();
  const auto CHANNEL_SIZE = MAX_VALUE + 1;

  auto make_histogram = [](const auto &container) {
    auto hist = std::array<size_type, CHANNEL_SIZE>{};
    for (auto val : container) {
      hist[val] += 1;
    }
    return hist;
  };

  auto hist = make_histogram(img);
  auto total = img.total();

  // Calculate the sum of weights for all possible thresholds.
  // Formula: ⅀ [i = 0, MAX_VALUE] (t * hist[t])
  f32 sumT = {};
  for (auto t = 0; t < CHANNEL_SIZE; ++t) {
    sumT += f32(t * hist[t]);
  }

  f32 sumB = {}, sumF = {};     // Accumulated weights for background and foreground.
  f32 wB = {}, wF = {};         // Weight of threshold `t` for background and foreground.
  f32 mB = {}, mF = {};         // Mean for background and foreground.
  f32 BCV = {}, maxBCV = {};    // Between Class Variance and Maximum Between Class Variance.
  u8 optThresh = {};            // Optimal threshold.

  for (auto t = 0; t < CHANNEL_SIZE; ++t) {
    // Calculate background weight for threshold `t`.
    // Formula: wB = ⅀ [i = 0, t] hist[i]
    wB += f32(hist[t]);
    if (wB == 0) {
      continue;
    }

    // Calculate foreground weight for threshold `t`.
    // Formula: wF = totalPixels - wB.
    wF = f32(total) - wB;
    if (wF == 0) {
      break;
    }

    // Calculate mean (background) for threshold `t`.
    // Formula: mB = ⅀ [i = 0, t] i * hist[i] / wB
    sumB += f32(t * hist[t]);
    mB = sumB / wB;

    // Calculate mean (foreground) for threshold `t`.
    // Formula: mF = (sumT - sumB) / wF
    sumF = sumT - sumB;
    mF = sumF / wF;

    // Calculate 'Between Class Variance'.
    // Formula: 𝜎B² = Wb * Wf (μb - μf)²
    BCV = wB * wF * (mB - mF) * (mB - mF);

    if (BCV > maxBCV) {
      maxBCV = BCV;
      optThresh = t;
    }
  }

  return img |= [MAX_VALUE, MIN_VALUE, optThresh](auto value) {
    return value > optThresh ? MAX_VALUE : MIN_VALUE;
  };
}

template <>
auto ImgProc::binarize(Tensor<f32> &img) noexcept -> Tensor<f32> & {
  const auto MAX_VALUE = _detail::limits<f32>::max(), MIN_VALUE = _detail::limits<f32>::min();
  const auto PIVOT = MAX_VALUE / 2;
  return img |= [MAX_VALUE, MIN_VALUE, PIVOT](auto value) {
    return value > PIVOT ? MAX_VALUE : MIN_VALUE;
  };
}

template <BitDepth B>
auto ImgProc::resize(const Tensor<B> &src, size_type new_width, size_type new_height) -> Tensor<B> {
  auto src_meta = Image::Meta::decode_shape(src.shape());
  auto resized_img = Image::make<B>(new_width, new_height, src_meta.channels());
  constexpr auto TYPE = std::is_same_v<u8, B> ? STBIR_TYPE_UINT8 : STBIR_TYPE_FLOAT;
  stbir_resize(src.data(), src_meta.width(), src_meta.height(), 0, resized_img.data(), new_width, new_height, 0,
               TYPE, src_meta.channels(), STBIR_ALPHA_CHANNEL_NONE, 0, STBIR_EDGE_CLAMP, STBIR_EDGE_CLAMP,
               STBIR_FILTER_BOX, STBIR_FILTER_BOX, STBIR_COLORSPACE_SRGB, nullptr);
  return resized_img;
}

template auto ImgProc::resize<u8>(const Tensor<u8> &src, size_type new_width, size_type new_height)
    -> Tensor<u8>;

template auto ImgProc::resize<f32>(const Tensor<f32> &src, size_type new_width, size_type new_height)
    -> Tensor<f32>;

}
