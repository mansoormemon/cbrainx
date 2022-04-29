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

#ifndef CBRAINX__STOPWATCH_HH_
#define CBRAINX__STOPWATCH_HH_

#include <chrono>

#include "typeAliases.hh"

namespace cbx {

/// \brief The `Stopwatch` class represents a simple stopwatch.
///
/// \details
/// It contains simple functionality for naive benchmarking.
class Stopwatch {
 public:
  using clock = std::chrono::high_resolution_clock;
  using time_point = std::chrono::time_point<clock>;

 private:
  /// \brief Starting time.
  time_point start_ = {};

  /// \brief Stopping time.
  time_point end_ = {};

  /// \brief State of the stopwatch.
  bool is_ticking_ = {};

 public:
  // /////////////////////////////////////////////
  // Constructors and Destructors.
  // /////////////////////////////////////////////

  /// \brief Default constructor.
  Stopwatch() = default;

  /// \brief Default copy constructor.
  /// \param[in] other Source stopwatch.
  Stopwatch(const Stopwatch &other) = default;

  /// \brief Default destructor.
  ~Stopwatch() = default;

  // /////////////////////////////////////////////
  // Assignment Operators
  // /////////////////////////////////////////////

  /// \brief Default copy assignment operator.
  /// \param[in] other Source stopwatch.
  /// \return A reference to self.
  auto operator=(const Stopwatch &other) -> Stopwatch & = default;

  // /////////////////////////////////////////////
  // Controls
  // /////////////////////////////////////////////

  /// \brief Starts the stopwatch.
  /// \param[in] force_renew If true, forces the stopwatch to renew if it is already active.
  auto start(bool force_renew = true) noexcept -> void;

  /// \brief Resumes the stopwatch.
  auto resume() noexcept -> void;

  /// \brief Stops the stopwatch.
  auto stop() noexcept -> void;

  /// \brief Returns whether the stopwatch is ticking or not.
  /// \return True if the stopwatch is ticking.
  [[nodiscard]] auto is_ticking() const noexcept -> bool;

  // /////////////////////////////////////////////
  // Query Functions
  // /////////////////////////////////////////////

  /// \brief Returns the duration between two time points depending on the state of the stopwatch.
  /// \tparam D Data type of the duration.
  /// \return The duration between the starting and ending point if the stopwatch is idle. Otherwise, the time
  /// elapsed since the stopwatch was turned on.
  template <typename D = std::chrono::milliseconds>
  [[nodiscard]] auto get_duration() {
    if (is_ticking_) {
      end_ = clock::now();
    }
    return std::chrono::duration_cast<D>(end_ - start_).count();
  }
};

}

#endif
