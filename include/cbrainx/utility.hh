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

#ifndef CBRAINX__UTILITY_HH_
#define CBRAINX__UTILITY_HH_

#include <string_view>

#include <fmt/format.h>

namespace cbx {

enum Verbosity { L0, L1, L2, L3 };

template <typename... Args>
auto verbose(Verbosity demand, Verbosity current, std::string_view fmt_str, Args &&...args) -> void {
  if (demand > current) {
    return;
  }
  fmt::vprint(fmt_str, fmt::make_format_args(std::forward<Args>(args)...));
}

}

#endif
