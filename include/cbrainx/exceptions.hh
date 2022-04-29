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

#ifndef CBRAINX__EXCEPTIONS_HH_
#define CBRAINX__EXCEPTIONS_HH_

#include <exception>
#include <string>
#include <string_view>

#include <fmt/core.h>

#include "typeAliases.hh"

namespace cbx {

/// \brief An object of `ImageIOError` class will be thrown as an exception to report errors during reading or
/// writing an image to/from a disk.
class ImageIOError : public std::exception {
 private:
  /// \brief Error message.
  std::string msg_ = {};

 public:
  /// \brief Parameterized Constructor.
  /// \tparam Args Data type of the arguments.
  /// \param[in] fmt_str Format string.
  /// \param[in] args Any optional arguments for \p fmt_str.
  template <typename... Args>
  explicit ImageIOError(std::string_view fmt_str, Args... args)
      : msg_{fmt::vformat(fmt_str, fmt::make_format_args(args...))} {}

  /// \brief Returns error description.
  /// \return Error message.
  [[nodiscard]] auto what() const noexcept -> str override;
};

/// \brief An object of `IncompatibleColorModelError` class will be thrown as an exception to report errors due
/// to an incompatible color model.
class IncompatibleColorModelError : public std::exception {
 private:
  /// \brief Error message.
  std::string msg_ = {};

 public:
  /// \brief Parameterized Constructor.
  /// \tparam Args Data type of the arguments.
  /// \param[in] fmt_str Format string.
  /// \param[in] args Any optional arguments for \p fmt_str.
  template <typename... Args>
  explicit IncompatibleColorModelError(std::string_view fmt_str, Args... args)
      : msg_{fmt::vformat(fmt_str, fmt::make_format_args(args...))} {}

  /// \brief Returns error description.
  /// \return Error message.
  [[nodiscard]] auto what() const noexcept -> str override;
};

/// \brief An object of the 'IndexOutOfBoundsError' class will be thrown as an exception to report errors when
/// attempting to access elements outside of a defined range.
class IndexOutOfBoundsError : public std::exception {
 private:
  /// \brief Error message.
  std::string msg_ = {};

 public:
  /// \brief Parameterized Constructor.
  /// \tparam Args Data type of the arguments.
  /// \param[in] fmt_str Format string.
  /// \param[in] args Any optional arguments for \p fmt_str.
  template <typename... Args>
  explicit IndexOutOfBoundsError(std::string_view fmt_str, Args... args)
      : msg_{fmt::vformat(fmt_str, fmt::make_format_args(args...))} {}

  /// \brief Returns error description.
  /// \return Error message.
  [[nodiscard]] auto what() const noexcept -> str override;
};

/// \brief An object of the `RankError` class will be thrown as an exception to report errors due to an
/// invalid interpretation of the tensor's rank.
class RankError : public std::exception {
 private:
  /// \brief Error message.
  std::string msg_ = {};

 public:
  /// \brief Parameterized Constructor.
  /// \tparam Args Data type of the arguments.
  /// \param[in] fmt_str Format string.
  /// \param[in] args Any optional arguments for \p fmt_str.
  template <typename... Args>
  explicit RankError(std::string_view fmt_str, Args... args)
      : msg_{fmt::vformat(fmt_str, fmt::make_format_args(args...))} {}

  /// \brief Returns error description.
  /// \return Error message.
  [[nodiscard]] auto what() const noexcept -> str override;
};

/// \brief An object of the `RankError` class will be thrown as an exception to report errors due to an
/// incompatible or unexpected shape.
class ShapeError : public std::exception {
 private:
  /// \brief Error message.
  std::string msg_ = {};

 public:
  /// \brief Parameterized Constructor.
  /// \tparam Args Data type of the arguments.
  /// \param[in] fmt_str Format string.
  /// \param[in] args Any optional arguments for \p fmt_str.
  template <typename... Args>
  explicit ShapeError(std::string_view fmt_str, Args... args)
      : msg_{fmt::vformat(fmt_str, fmt::make_format_args(args...))} {}

  /// \brief Returns error description.
  /// \return Error message.
  [[nodiscard]] auto what() const noexcept -> str override;
};

/// \brief An object of `UnrecognizedColorModelError` class will be thrown as an exception to report errors when
/// attempting to decode an unsupported color model.
class UnrecognizedColorModelError : public std::exception {
 private:
  /// \brief Error message.
  std::string msg_ = {};

 public:
  /// \brief Parameterized Constructor.
  /// \tparam Args Data type of the arguments.
  /// \param[in] fmt_str Format string.
  /// \param[in] args Any optional arguments for \p fmt_str.
  template <typename... Args>
  explicit UnrecognizedColorModelError(std::string_view fmt_str, Args... args)
      : msg_{fmt::vformat(fmt_str, fmt::make_format_args(args...))} {}

  /// \brief Returns error description.
  /// \return Error message.
  [[nodiscard]] auto what() const noexcept -> str override;
};

/// \brief An object of the`ValueError` class will be thrown as an exception to report errors due to logically
/// incorrect or unexpected input.
class ValueError : public std::exception {
 private:
  /// \brief Error message.
  std::string msg_ = {};

 public:
  /// \brief Parameterized Constructor.
  /// \tparam Args Data type of the arguments.
  /// \param[in] fmt_str Format string.
  /// \param[in] args Any optional arguments for \p fmt_str.
  template <typename... Args>
  explicit ValueError(std::string_view fmt_str, Args... args)
      : msg_{fmt::vformat(fmt_str, fmt::make_format_args(args...))} {}

  /// \brief Returns error description.
  /// \return Error message.
  [[nodiscard]] auto what() const noexcept -> str override;
};

}

#endif
