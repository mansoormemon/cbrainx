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

#ifndef CBRAINX__ABSTRACT_LAYER_HH_
#define CBRAINX__ABSTRACT_LAYER_HH_

#include <string>

#include "optimizers.hh"
#include "shape.hh"
#include "tensor.hh"
#include "typeAliases.hh"

namespace cbx {

// /////////////////////////////////////////////
// Enumerations
// /////////////////////////////////////////////

/// \brief Supported layer types.
enum class LayerType { Dense, Activation, Softmax };

/// \brief The `AbstractLayer` class defines a standard interface for all layers.
///
/// \see LayerType
class AbstractLayer {
 public:
  using value_type = f32;

  using container = Tensor<value_type>;

  using size_type = container::size_type;
  using difference_type = container::difference_type;

 private:
  /// \brief Layer ID.
  i32 id_ = {};

  /// \brief The name of the layer.
  std::string name_ = "LYR";

 protected:
  /// \brief The input layer.
  mutable container input_ = {};

  /// \brief The output layer.
  mutable container output_ = {};

 public:
  // /////////////////////////////////////////////
  // Constructors and Destructors
  // /////////////////////////////////////////////

  /// \brief Default constructor.
  AbstractLayer() = default;

  /// \brief Default copy constructor.
  /// \param[in] other Source layer.
  AbstractLayer(const AbstractLayer &other) = default;

  /// \brief Move constructor.
  /// \param[in] other Source layer.
  AbstractLayer(AbstractLayer &&other) noexcept;

  /// \brief Parameterized constructor.
  /// \param[in] id Layer ID.
  explicit AbstractLayer(i32 id);

  /// \brief Parameterized constructor.
  /// \param[in] name The name of the layer.
  explicit AbstractLayer(std::string_view name);

  /// \brief Parameterized constructor.
  /// \param[in] id Layer ID.
  /// \param[in] name The name of the layer.
  AbstractLayer(i32 id, std::string_view name);

  /// \brief Default destructor.
  virtual ~AbstractLayer() = default;

  // /////////////////////////////////////////////
  // Assignment Operators
  // /////////////////////////////////////////////

  /// \brief Default copy assignment operator.
  /// \param[in] other Source layer.
  /// \return A reference to self.
  auto operator=(const AbstractLayer &other) -> AbstractLayer & = default;

  /// \brief Move assignment operator.
  /// \param[in] other Source layer.
  /// \return A reference to self.
  auto operator=(AbstractLayer &&other) noexcept -> AbstractLayer &;

  // /////////////////////////////////////////////
  // Accessors and Mutators
  // /////////////////////////////////////////////

  /// \brief Returns the ID of the layer.
  /// \return Layer ID.
  [[nodiscard]] auto id() const -> i32;

  /// \brief Sets layer ID.
  /// \param[in] id The new ID of the layer.
  /// \return A reference to self.
  auto set_id(i32 id) -> AbstractLayer &;

  /// \brief Returns the name of the layer.
  /// \return The name of the layer.
  [[nodiscard]] auto name() const -> std::string;

  /// \brief Sets the name of the layer.
  /// \param[in] name The new name of the layer.
  /// \return A reference to self.
  auto set_name(std::string_view name) -> AbstractLayer &;

  // /////////////////////////////////////////////
  // Query Functions
  // /////////////////////////////////////////////

  /// \brief Returns the number of neurons in the layer.
  /// \return The number of neurons in the layer.
  [[nodiscard]] virtual auto neurons() const -> size_type = 0;

  /// \brief Returns the number of modifiable parameters in the layer.
  /// \return The number of modifiable parameters in the layer.
  [[nodiscard]] virtual auto parameters() const -> size_type = 0;

  /// \brief Returns the type of the layer.
  /// \return The type of the layer.
  ///
  /// \see LayerType
  [[nodiscard]] virtual auto type() const -> LayerType = 0;

  // /////////////////////////////////////////////
  // Informative
  // /////////////////////////////////////////////

  /// \brief Returns a string with information about the layer's properties.
  /// \return Information about the layer's properties as a string.
  [[nodiscard]] virtual auto property() const -> std::string = 0;

  /// \brief Returns a string with the layer's name and ID.
  /// \return The layer's name and ID as a string.
  [[nodiscard]] virtual auto to_string() const -> std::string;

  /// \brief Returns the layer's type as a string.
  /// \return The layer's type as a string.
  ///
  /// \see LayerType
  [[nodiscard]] virtual auto type_name() const -> std::string = 0;

  // /////////////////////////////////////////////
  // Core Functionality
  // /////////////////////////////////////////////

  /// \brief Returns the cached input layer.
  /// \return The cached input layer.
  [[nodiscard]] auto input() const -> const container &;

  /// \brief Returns the cached output layer.
  /// \return The cached output layer.
  [[nodiscard]] auto output() const -> const container &;

  /// \brief Drops the cached input and output layers.
  auto drop_caches() const -> void;

  /// \brief Forward pass.
  /// \param[in] input The input layer.
  /// \return The output layer.
  [[nodiscard]] virtual auto forward_pass(const container &input) const -> container = 0;

  /// \brief Backward pass.
  /// \param[in] upstream_gradient The upstream gradient.
  /// \param[in] optimizer The optimizer.
  /// \return The downstream gradient.
  [[nodiscard]] virtual auto backward_pass(const container &upstream_gradient, OptimizerWrapper optimizer)
      -> container = 0;
};

}

#endif
