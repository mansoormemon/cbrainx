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

#include "cbrainx/exceptions.hh"

namespace cbx {

auto ImageIOError::what() const noexcept -> str { return msg_.c_str(); }

auto IncompatibleColorModelError::what() const noexcept -> str { return msg_.c_str(); }

auto IndexOutOfBoundsError::what() const noexcept -> str { return msg_.c_str(); }

auto RankError::what() const noexcept -> str { return msg_.c_str(); }

auto ShapeError::what() const noexcept -> str { return msg_.c_str(); }

auto UnrecognizedColorModelError::what() const noexcept -> str { return msg_.c_str(); }

auto ValueError::what() const noexcept -> str { return msg_.c_str(); }

}
