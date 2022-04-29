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

#ifndef CBRAINX__IMAGE_HH_
#define CBRAINX__IMAGE_HH_

#include <string_view>
#include <tuple>
#include <type_traits>

#include "shape.hh"
#include "tensor.hh"
#include "typeAliases.hh"

namespace cbx {

// /////////////////////////////////////////////
// Constraints
// /////////////////////////////////////////////

/// \brief A constraint to filter semantic data types representing an image's bit-depth.
/// \tparam T The data type to which the constraint is to be applied.
///
/// \details
/// Bit depth, also known as color depth, is the number of bits used by each color component of a pixel. This
/// constraint only allows the most commonly used bit depths for images, i.e., `u8` and `f32`.
///
/// The following table lists the available bit depths.
///
/// | Bits | Data type | Range     |
/// | ---- | :-------: | --------- |
/// | 8    | `u8`      | 0 - 255   |
/// | 32   | `f32`     | 0.0 - 1.0 |
///
/// \relates Image
template <typename T>
concept BitDepth = std::disjunction_v<std::is_same<T, u8>, std::is_same<T, f32>>;

// /////////////////////////////////////////////
// Implementation Detail
// /////////////////////////////////////////////

/// \cond impl_detail

namespace _detail {

  /// \brief Specifies the member 'type', which toggles between 'u8' and 'f32' bit depths.
  /// \tparam B The bit depth to be toggled.
  ///
  /// \details
  /// The following is a truth table.
  ///
  /// | Input | Output |
  /// | :---: | :----: |
  /// | `u8`  | `f32`  |
  /// | `f32` | `u8`   |
  template <BitDepth B>
  struct morphed {
    using type = std::conditional_t<std::is_same_v<u8, B>, f32, u8>;
  };

  /// \brief Helper type for `morphed`.
  template <BitDepth B>
  using morphed_value_t = typename morphed<B>::type;

}

/// \endcond

/// \brief The `Image` class contains basic functionality for working with images.
///
/// \details
/// An image is a two-dimensional grid of pixels. A pixel is the smallest addressable element of an image and
/// comprises one or more color channels. The order of these color channels determines the color model of the
/// image, which is then fused with the image's bit depth to define the image's color space.
///
/// In memory, the image is a sequential array in a row-major format. In the case of the colored images, the
/// last dimension represents the channels, which are placed together in memory as interleaving bands.
///
/// \see BitDepth
class Image {
 public:
  using size_type = usize;

  // /////////////////////////////////////////////
  // Constants and Enumerations
  // /////////////////////////////////////////////

  /// \brief Image quality for the JPG format.
  static constexpr auto const JPG_QUALITY = 96;

  /// \brief Supported image formats.
  enum class Format { BMP, JPG, PNG };

  /// \brief Recognized color channels.
  enum Channel { Mono = 0x1, Red = 0x2, Green = 0x4, Blue = 0x8, Alpha = 0x10 };

  /// \brief Supported color models.
  ///
  /// \details
  /// The following table serves as the basis for deducing the image's color model based on the number of
  /// channels.
  ///
  /// | Color Model | Channel Count |  Individual Channels    |
  /// | ----------- | :-----------: | ----------------------- |
  /// | Gray        | 1             | Mono                    |
  /// | GrayAlpha   | 2             | Mono, Alpha             |
  /// | RGB         | 3             | Red, Green, Blue        |
  /// | RGBA        | 4             | Red, Green, Blue, Alpha |
  ///
  /// \see Image::Channel Image::Meta::model()
  enum Model { Gray = 0x1, GrayAlpha = 0x2, RGB = 0x4, RGBA = 0x8 };

  // ////////////////////////////////////////////////////////////////////////////
  // Nested Class(es)
  // ////////////////////////////////////////////////////////////////////////////

  /// \brief The `Meta` class provides an interface for adapting between a tensor and an image.
  ///
  /// \details
  /// This class provides functionality to decode an image's shape to extract information, such as its width,
  /// height, channels, color model, channel order, etc.
  ///
  /// \note The 'Meta' class is more tolerant than the 'Shape' class, for it accepts zero as a dimension.
  /// However, when converting a `Meta` object to a `Shape` object, one must ensure that it conforms to the
  /// constraints implied by the `Shape` class.
  ///
  /// \see Shape
  class Meta {
   public:
    using size_type = usize;

   private:
    /// \brief The width of the image.
    size_type width_ = {};

    /// \brief The height of the image.
    size_type height_ = {};

    /// \brief The number of channels in the image.
    size_type channels_ = {};

   public:
    // /////////////////////////////////////////////
    // Constructors and Destructors
    // /////////////////////////////////////////////

    /// \brief Default constructor.
    Meta() = default;

    /// \brief Constructs a `Meta` object from the given dimensions.
    /// \param[in] width Width of the image.
    /// \param[in] height Height of the image.
    /// \param[in] channels Number of channels in the image.
    Meta(size_type width, size_type height, size_type channels = 1);

    /// \brief Default copy constructor.
    /// \param meta Source `Meta` object.
    Meta(const Meta &meta) = default;

    /// \brief Default destructor.
    ~Meta() = default;

    // /////////////////////////////////////////////
    // Assignment Operators
    // /////////////////////////////////////////////

    /// \brief Default copy assignment operator.
    /// \param meta Source `Meta` object.
    /// \return A reference to self.
    auto operator=(const Meta &meta) -> Meta & = default;

    // /////////////////////////////////////////////
    // Accessors and Mutators
    // /////////////////////////////////////////////

    /// \brief Returns the width of the image.
    /// \return Width of the image.
    [[nodiscard]] auto width() const noexcept -> size_type;

    /// \brief Sets the width of the image.
    /// \param[in] new_width New width of the image.
    /// \return A reference to self.
    ///
    /// \throws ValueError
    auto set_width(size_type new_width) -> Meta &;

    /// \brief Returns the height of the image.
    /// \return Height of the image.
    [[nodiscard]] auto height() const noexcept -> size_type;

    /// \brief Sets the height of the image.
    /// \param[in] new_height New height of the image.
    /// \return A reference to self.
    ///
    /// \throws ValueError
    auto set_height(size_type new_height) -> Meta &;

    /// \brief Returns the number of channels in the image.
    /// \return Number of channels in the image.
    [[nodiscard]] auto channels() const noexcept -> size_type;

    /// \brief Sets the number of channels in the image.
    /// \param[in] new_channels New number of channels in the image.
    /// \return A reference to self.
    ///
    /// \throws ValueError
    auto set_channels(size_type new_channels) -> Meta &;

    // /////////////////////////////////////////////
    // Query Functions
    // /////////////////////////////////////////////

    /// \brief Unwraps the `Meta` object and returns it as a tuple.
    /// \tparam T Data type of tuple elements.
    /// \return A tuple of image's metadata in the order of width, height, and channels.
    template <typename T = size_type>
    [[nodiscard]] auto unwrap() const -> std::tuple<T, T, T> {
      return std::make_tuple(static_cast<T>(width_), static_cast<T>(height_), static_cast<T>(channels_));
    }

    /// \brief Returns the deduced color model of the image.
    /// \return Color model of the image.
    ///
    /// \details
    /// The function uses the number of channels in the image to deduce its color model and throws an exception
    /// on failure.
    ///
    /// \throws UnrecognizedColorModelError
    ///
    /// \see Image::Model
    [[nodiscard]] auto model() const -> Model;

    /// \brief Returns the total number of pixels in the image.
    /// \return Total number of pixels in the image.
    [[nodiscard]] auto pixels() const noexcept -> size_type;

    /// \brief Returns the total number of elements (pixel components) in the image.
    /// \return Total number of elements in the image.
    [[nodiscard]] auto total() const noexcept -> size_type;

    /// \brief Returns a bitmask encoding available color channels represented by the color model.
    /// \return A bitmask encoding available color channels.
    ///
    /// \throws UnrecognizedColorModelError
    ///
    /// \see Image::Channels Image::Model
    [[nodiscard]] auto bitmask() const -> u32;

    /// \brief Returns whether the color model of the image is compatible with the specified color model.
    /// \param[in] target The color model to be tested for compatibility.
    /// \return True if the image is compatible with \p target.
    ///
    /// \details
    /// For any two color models A and B, A is compatible with B if a subset of A. For the scope of this
    /// function: A = `this->model()` and B = \p target.
    ///
    /// \note Compatibility between color models is not commutative.
    ///
    /// \throws UnrecognizedColorModelError
    ///
    /// \see Image::Model
    [[nodiscard]] auto is_compatible(Model target) const -> bool;

    /// \brief Returns whether the image has the channel \p channel.
    /// \param[in] channel The channel to be looked up.
    /// \return True if the image has the channel \p channel.
    ///
    /// \see Image::Model
    [[nodiscard]] auto has_channel(Channel channel) const -> bool;

    /// \brief Returns the position of the channel \p channel.
    /// \param[in] channel The channel whose position is to be looked up.
    /// \return Returns the position of the channel \p channel, or `-1` if the channel is not present.
    ///
    /// \note The color model is used to determine the position of the color channel.
    ///
    /// \see Image::Model
    [[nodiscard]] auto position_of(Channel channel) const -> i32;

    // /////////////////////////////////////////////
    // Informative
    // /////////////////////////////////////////////

    /// \brief Converts the `Meta` object to an equivalent shape.
    /// \return A shape representing the dimensions of the image.
    ///
    /// \note This function will throw an exception if the `Meta` object does not conform to the constraints
    /// implied by the `Shape` class.
    [[nodiscard]] auto to_shape() const -> Shape;

    // /////////////////////////////////////////////////////////////
    // Static Functions
    // /////////////////////////////////////////////////////////////

    /// \brief Creates a `Meta` object by decoding the given shape.
    /// \param[in] shape The shape to be decoded.
    /// \return A `Meta` object equivalent to the shape.
    ///
    /// \details
    /// This function throws an exception if the shape does not portray an image.
    ///
    /// \throws ShapeError
    [[nodiscard]] static auto decode_shape(const Shape &shape) -> Meta;
  };

  // /////////////////////////////////////////////////////////////
  // Static Functions
  // /////////////////////////////////////////////////////////////

  // /////////////////////////////////////////////
  // Factory Functions
  // /////////////////////////////////////////////

  /// \brief Makes an image from the given metadata.
  /// \tparam B Bit depth of the image.
  /// \param[in] meta Metadata about the image.
  /// \return A tensor representing an image.
  template <BitDepth B>
  [[nodiscard]] static auto make(const Meta &meta) -> Tensor<B> {
    return Tensor<B>{meta.to_shape()};
  }

  /// \brief Makes an image from the given dimensions.
  /// \tparam B Bit depth of the image.
  /// \param[in] width The width of the image.
  /// \param[in] height The height of the image.
  /// \param[in] channels The number of channels in the image.
  /// \return A tensor representing an image.
  template <BitDepth B>
  [[nodiscard]] static auto make(size_type width, size_type height, size_type channels = 1) -> Tensor<B> {
    return make<B>({width, height, channels});
  }

  /// \brief Toggles the image's bit depth between `u8` and `f32`.
  /// \tparam B Bit depth of the image.
  /// \param[in] img The image to be morphed.
  /// \return The morphed image.
  template <BitDepth B>
  [[nodiscard]] static auto morph(const Tensor<B> &img) {
    using type = _detail::morphed_value_t<B>;
    return img.template transformed<type>([](auto pix) {
      return pix * (std::is_same_v<u8, type> ? UCHAR_MAX : 1.0F / UCHAR_MAX);
    });
  }

  // /////////////////////////////////////////////
  // I/O Functions
  // /////////////////////////////////////////////

  /// \brief Reads an image from the disk.
  /// \tparam B Bit depth of the image.
  /// \param[in] img_path Path to the image on disk.
  /// \return A tensor representing an image.
  ///
  /// \throws ImageIOError
  template <BitDepth B = u8>
  [[nodiscard]] static auto read(std::string_view img_path) -> Tensor<B>;

  /// \brief Writes the given image to the disk.
  /// \tparam B Bit depth of the image.
  /// \param[in] img The image to be written to the disk.
  /// \param[in] img_path Path of the image on disk.
  /// \param[in] fmt Image format.
  ///
  /// \throws ImageIOError
  template <BitDepth B>
  static auto write(const Tensor<B> &img, std::string_view img_path, Format fmt = Format::JPG) -> void;
};

}

#endif
