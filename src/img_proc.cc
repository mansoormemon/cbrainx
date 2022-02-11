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
#include <stdexcept>

#include <fmt/format.h>

namespace cbx {

auto ImgProc::has_channel_check(Image::Meta meta, Image::Channel target) -> void {
  if (not meta.has_channel(target)) {
    throw std::logic_error{
        fmt::format("cbx::ImgProc::has_channel_check: channel(={}) is not available", target)};
  }
}

auto ImgProc::model_compatibility_check(Image::Meta meta, Image::Model target) -> void {
  if (not meta.is_compatible(target)) {
    throw std::logic_error{fmt::format(
        "cbx::ImgProc::model_compatibility_check: color models are not compatible(current={}, target={})",
        meta.model(), target)};
  }
}

auto ImgProc::invert(Tensor<u8> &img) -> Tensor<u8> & {
  const auto MAX_CHANNEL_VALUE = std::numeric_limits<u8>::max();
  const auto CHANNEL_SIZE = MAX_CHANNEL_VALUE + 1;

  auto make_lookup_table = [CHANNEL_SIZE]() {
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

  auto make_histogram = [CHANNEL_SIZE](const auto &container) {
    auto hist = std::array<std::size_t, CHANNEL_SIZE>{};
    for (const auto &val : container) {
      hist[val] += 1;
    }
    return hist;
  };

  auto hist = make_histogram(img);
  auto total = img.total();

  // Calculate the sum of weights for all possible thresholds.
  // Formula: summation[i = 0, MAX_CHANNEL_VALUE] (t * hist[t])
  auto sumT = f32{};
  for (auto t = 0; t < CHANNEL_SIZE; ++t) {
    sumT += static_cast<f32>(t * hist[t]);
  }

  auto sumB = f32{}, sumF = f32{};     // Accumulated weights for background and foreground.
  auto wB = f32{}, wF = f32{};         // Weight of threshold `t` for background and foreground.
  auto mB = f32{}, mF = f32{};         // Mean for background and foreground.
  auto BCV = f32{}, maxBCV = f32{};    // Between Class Variance and Maximum Between Class Variance.
  auto optThresh = u8{};               // Optimal threshold.

  for (auto t = 0; t < CHANNEL_SIZE; t += 1) {
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

}
