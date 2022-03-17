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

#ifndef CBRAINX__EXCEPTIONS_HH_
#define CBRAINX__EXCEPTIONS_HH_

#include <exception>
#include <string>

#include "type_aliases.hh"

namespace cbx {

class ImageIOError : public std::exception {
 private:
  std::string msg_ = {};

 public:
  explicit ImageIOError(std::string msg);

  [[nodiscard]] auto what() const noexcept -> str override;
};

// /////////////////////////////////////////////////////////////

class IncompatibleColorModelError : public std::exception {
 private:
  std::string msg_ = {};

 public:
  explicit IncompatibleColorModelError(std::string msg);

  [[nodiscard]] auto what() const noexcept -> str override;
};

// /////////////////////////////////////////////////////////////

class RankError : public std::exception {
 private:
  std::string msg_ = {};

 public:
  explicit RankError(std::string msg);

  [[nodiscard]] auto what() const noexcept -> str override;
};

// /////////////////////////////////////////////////////////////

class ShapeError : public std::exception {
 private:
  std::string msg_ = {};

 public:
  explicit ShapeError(std::string msg);

  [[nodiscard]] auto what() const noexcept -> str override;
};

// /////////////////////////////////////////////////////////////

class UnrecognizedColorModelError : public std::exception {
 private:
  std::string msg_ = {};

 public:
  explicit UnrecognizedColorModelError(std::string msg);

  [[nodiscard]] auto what() const noexcept -> str override;
};

}

#endif
