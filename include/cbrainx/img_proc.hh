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

#ifndef CBRAINX__IMG_PROC_HH_
#define CBRAINX__IMG_PROC_HH_

#include "image.hh"
#include "tensor.hh"
#include "type_aliases.hh"

namespace cbx {

// /////////////////////////////////////////////
// Implementation Detail
// /////////////////////////////////////////////

/// \cond impl_detail

namespace _detail {

/// \brief Provides a way to query properties of a given data type in the context of bit depth.
/// \tparam B Bit depth of the image.
template <BitDepth B>
struct limits {
  using size_type = usize;

  // /////////////////////////////////////////////
  // Query Functions
  // /////////////////////////////////////////////

  /// \brief Returns the number of bits for the given bit depth.
  /// \return Number of bits for the given bit depth.
  static constexpr auto bits() noexcept -> size_type;

  /// \brief Returns the smallest finite value of the given bit depth.
  /// \return Smallest finite value for the given bit depth.
  static constexpr auto min() noexcept -> B;

  /// \brief Returns the largest finite value of the given bit depth.
  /// \return Largest finite value for the given bit depth.
  static constexpr auto max() noexcept -> B;
};

}

/// \endcond

/// \brief The `ImgProc` class contains functionality for image processing.
///
/// \details
/// Image processing is a method of performing operations on an image to improve it or extract useful
/// information. This class contains utilities for resizing, channel extraction, thresholding images, etc.
class ImgProc {
 public:
  using size_type = usize;

 private:
  // /////////////////////////////////////////////
  // Helpers
  // /////////////////////////////////////////////

  /// \brief Checks if the image has the channel \p channel.
  /// \param[in] meta Metadata of the image.
  /// \param[in] channel The channel to be looked up.
  ///
  /// \details
  /// This function throws an exception if the image does not have the channel \p channel.
  ///
  /// \throws IncompatibleColorModelError
  static auto _s_has_channel_check(Image::Meta meta, Image::Channel channel) -> void;

 public:
  // /////////////////////////////////////////////
  // Core Functionality
  // /////////////////////////////////////////////

  /// \brief Extracts the specified channel from the image.
  /// \tparam B Bit depth of the image.
  /// \param[in] src Source image.
  /// \param[in] channel The channel to be extracted.
  /// \return A grayscale image formed from the extracted channel.
  ///
  /// \details
  /// This function throws an exception if the image does not have the specified color channel.
  ///
  /// \throws IncompatibleColorModelError
  template <BitDepth B>
  [[nodiscard]] static auto extract_channel(const Tensor<B> &src, Image::Channel channel) -> Tensor<B> {
    auto meta = Image::Meta::decode_shape(src.shape());
    ImgProc::_s_has_channel_check(meta, channel);
    auto mono_img = Image::make<B>(meta.width(), meta.height());
    for (auto src_it = src.begin() + meta.position_of(channel); auto &pix : mono_img) {
      pix = *src_it;
      src_it += meta.channels();
    }
    return mono_img;
  }

  /// \brief Converts the given image to grayscale.
  /// \tparam B Bit depth of the image.
  /// \param[in] src Source image.
  /// \return Grayscale image.
  ///
  /// \note If the image's color model is not compatible with `Image::Model::RGB`, it returns a copy of the
  /// source image.
  template <BitDepth B>
  [[nodiscard]] static auto grayscale(const Tensor<B> &src) -> Tensor<B> {
    auto meta = Image::Meta::decode_shape(src.shape());
    if (not meta.is_compatible(Image::Model::RGB)) {
      return src;
    }

    enum { Red, Green, Blue };
    auto gray_img = Image::make<B>(meta.width(), meta.height());
    for (auto src_it = src.begin(); auto &pix : gray_img) {
      pix = (0.3 * src_it[Red]) + (0.59 * src_it[Green]) + (0.11 * src_it[Blue]);
      src_it += meta.channels();
    }
    return gray_img;
  }

  /// \brief Inverts the given image.
  /// \tparam B Bit depth of the image.
  /// \param[in] img The image to be inverted.
  /// \return A reference to \p img.
  template <BitDepth B>
  static auto invert(Tensor<B> &img) noexcept -> Tensor<B> &;

  /// \brief Binarizes the given image.
  /// \tparam B Bit depth of the image.
  /// \param[in] img The image to be binarized.
  /// \return A reference to \p img.
  template <BitDepth B>
  static auto binarize(Tensor<B> &img) noexcept -> Tensor<B> &;

  /// \brief Resizes the given image.
  /// \tparam B Bit depth of the image.
  /// \param[in] src Source image.
  /// \param[in] new_width, new_height New dimensions of the image.
  /// \return Resized image.
  template <BitDepth B>
  [[nodiscard]] static auto resize(const Tensor<B> &src, size_type new_width, size_type new_height)
      -> Tensor<B>;

  /// \brief Rescales the given image.
  /// \tparam B Bit depth of the image.
  /// \param[in] src Source image.
  /// \param[in] factor The factor by which to rescale the image.
  /// \return Rescaled image.
  template <BitDepth B>
  [[nodiscard]] static auto rescale(const Tensor<B> &src, f32 factor) -> Tensor<B> {
    auto meta = Image::Meta::decode_shape(src.shape());
    size_type new_width = meta.width() * factor, new_height = meta.height() * factor;
    return ImgProc::resize(src, new_width, new_height);
  }
};

}

#endif
