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

#ifndef CBRAINX__STOPWATCH_HH_
#define CBRAINX__STOPWATCH_HH_

#include <chrono>

#include "type_aliases.hh"

namespace cbx {

class Stopwatch {
 public:
  using clock = std::chrono::high_resolution_clock;
  using duration = clock::duration;

 private:
  duration start_point_ = {};
  duration end_point_ = {};
  bool ticking_ = {};

 public:
  Stopwatch() = default;

  Stopwatch(const Stopwatch &other) = default;

  ~Stopwatch() = default;

  // /////////////////////////////////////////////////////////////

  auto operator=(const Stopwatch &other) -> Stopwatch & = default;

  // /////////////////////////////////////////////////////////////

  auto start(bool force_restart = false) -> void;

  auto stop() -> void;

  // /////////////////////////////////////////////////////////////

  template <typename T>
  auto get_interval() {
    if (ticking_) {
      end_point_ = clock::now().time_since_epoch();
    }
    auto diff = std::chrono::duration_cast<T>(end_point_ - start_point_).count();
    return diff;
  }

  [[nodiscard]] auto is_ticking() const -> bool;
};

}

#endif
