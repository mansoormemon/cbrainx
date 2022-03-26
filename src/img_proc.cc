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

#include "cbrainx/img_proc.hh"

#include <array>

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb_image_resize.h>

#include <fmt/format.h>

#include "cbrainx/exceptions.hh"

namespace cbx {

auto ImgProc::has_channel_check(Image::Meta meta, Image::Channel target) -> void {
  if (not meta.has_channel(target)) {
    throw IncompatibleColorModelError{
        fmt::format("cbx::ImgProc::has_channel_check: channel(={}) is not available", target)};
  }
}

auto ImgProc::model_compatibility_check(Image::Meta meta, Image::Model target) -> void {
  if (not meta.is_compatible(target)) {
    throw IncompatibleColorModelError{fmt::format(
        "cbx::ImgProc::model_compatibility_check: color models are not compatible(current={}, target={})",
        meta.model(), target)};
  }
}

// /////////////////////////////////////////////////////////////

auto ImgProc::invert(Tensor<u8> &img) -> Tensor<u8> & {
  const auto MAX_CHANNEL_VALUE = std::numeric_limits<u8>::max();
  const auto CHANNEL_SIZE = MAX_CHANNEL_VALUE + 1;

  auto make_lookup_table = []() {
    auto table = std::array<u8, CHANNEL_SIZE>{};
    std::generate(table.begin(), table.end(), [n = MAX_CHANNEL_VALUE]() mutable {
      return n--;
    });
    return table;
  };

  auto lookup_table = make_lookup_table();
  for (auto &val : img) {
    val = lookup_table[val];
  }
  return img;
}

// /////////////////////////////////////////////////////////////

auto ImgProc::binarize(Tensor<u8> &img) -> Tensor<u8> & {
  // Algorithm: Otsu's Method
  // Otsu's thresholding method involves iterating through all the possible thresholds and calculating a
  // measure of spread for the pixel intensities in the foreground and background. The aim is to find a
  // threshold value where the sum of the foreground and background spreads is minimum. For this, we can
  // calculate the 'Between-Class Variance' of the image to find the optimal threshold.
  //
  // Formula: 𝜎B² = Wb * Wf (μb - μf)²
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
  // 1. http://www.labbookpages.co.uk/software/imgProc/otsuThreshold.html#explained

  const auto MAX_CHANNEL_VALUE = std::numeric_limits<u8>::max();
  const auto MIN_CHANNEL_VALUE = std::numeric_limits<u8>::min();
  const auto CHANNEL_SIZE = MAX_CHANNEL_VALUE + 1;

  auto make_histogram = [](const auto &container) {
    auto hist = std::array<usize, CHANNEL_SIZE>{};
    for (const auto &val : container) {
      hist[val] += 1;
    }
    return hist;
  };

  auto hist = make_histogram(img);
  auto total = img.total();

  // Calculate the sum of weights for all possible thresholds.
  // Formula: summation[i = 0, MAX_CHANNEL_VALUE] (t * hist[t])
  f32 sumT = {};
  for (auto t = 0; t < CHANNEL_SIZE; ++t) {
    sumT += static_cast<f32>(t * hist[t]);
  }

  f32 sumB = {}, sumF = {};     // Accumulated weights for background and foreground.
  f32 wB = {}, wF = {};         // Weight of threshold `t` for background and foreground.
  f32 mB = {}, mF = {};         // Mean for background and foreground.
  f32 BCV = {}, maxBCV = {};    // Between Class Variance and Maximum Between Class Variance.
  u8 optThresh = {};            // Optimal threshold.

  for (auto t = 0; t < CHANNEL_SIZE; ++t) {
    // Calculate background weight for threshold `t`.
    // Formula: wB = summation[i = 0, t] hist[i]
    wB += static_cast<f32>(hist[t]);
    if (wB == 0) {
      continue;
    }

    // Calculate foreground weight for threshold `t`.
    // Formula: wF = totalPixels - wB.
    wF = static_cast<f32>(total) - wB;
    if (wF == 0) {
      break;
    }

    // Calculate mean (background) for threshold `t`.
    // Formula: mB = summation[i = 0, t] i * hist[i] / wB
    sumB += static_cast<f32>(t * hist[t]);
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

  for (auto &value : img) {
    value = value > optThresh ? MAX_CHANNEL_VALUE : MIN_CHANNEL_VALUE;
  }
  return img;
}

// /////////////////////////////////////////////////////////////

template <image_datatype T>
auto ImgProc::resize(const Tensor<T> &img, const Image::Meta &meta) -> Tensor<T> {
  auto src_meta = Image::Meta::decode_shape(img.shape());
  auto resized_img = Image::make<T>({meta.width(), meta.height(), src_meta.channels()});
  auto type = std::is_same_v<u8, T> ? STBIR_TYPE_UINT8 : STBIR_TYPE_FLOAT;
  stbir_resize(img.data(), src_meta.width(), src_meta.height(), 0, resized_img.data(), meta.width(),
               meta.height(), 0, type, src_meta.channels(), STBIR_ALPHA_CHANNEL_NONE, 0, STBIR_EDGE_CLAMP,
               STBIR_EDGE_CLAMP, STBIR_FILTER_BOX, STBIR_FILTER_BOX, STBIR_COLORSPACE_SRGB, nullptr);
  return resized_img;
}

// /////////////////////////////////////////////////////////////

template auto ImgProc::resize<u8>(const Tensor<u8> &img, const Image::Meta &meta) -> Tensor<u8>;

template auto ImgProc::resize<f32>(const Tensor<f32> &img, const Image::Meta &meta) -> Tensor<f32>;

// /////////////////////////////////////////////////////////////

template <image_datatype T>
auto ImgProc::rescale(const Tensor<T> &img, f32 factor) -> Tensor<T> {
  auto meta = Image::Meta::decode_shape(img.shape());
  i32 new_width = meta.width() * factor, new_height = meta.height() * factor;
  return ImgProc::resize(img, {new_width, new_height});
}

// /////////////////////////////////////////////////////////////

template auto ImgProc::rescale<u8>(const Tensor<u8> &img, f32 factor) -> Tensor<u8>;

template auto ImgProc::rescale<f32>(const Tensor<f32> &img, f32 factor) -> Tensor<f32>;

}
