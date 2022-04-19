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

#include "cbrainx/stopwatch.hh"

namespace cbx {

// /////////////////////////////////////////////
// Controls
// /////////////////////////////////////////////

auto Stopwatch::start(bool force_renew) noexcept -> void {
  if (not is_ticking_ or force_renew) {
    start_ = clock::now();
    is_ticking_ = true;
  }
}

auto Stopwatch::resume() noexcept -> void {
  if (not is_ticking_) {
    auto now = clock::now();
    auto diff = now - end_;
    start_ += diff;
    is_ticking_ = true;
  }
}

auto Stopwatch::stop() noexcept -> void {
  if (is_ticking_) {
    end_ = clock::now();
    is_ticking_ = false;
  }
}

// /////////////////////////////////////////////
// Query Functions
// /////////////////////////////////////////////

auto Stopwatch::is_ticking() const noexcept -> bool { return is_ticking_; }

}
