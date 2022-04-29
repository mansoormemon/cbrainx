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

#ifndef CBRAINX__ACTIVATION_LAYER_HH_
#define CBRAINX__ACTIVATION_LAYER_HH_

#include "abstractLayer.hh"
#include "activationFunctions.hh"
#include "typeAliases.hh"

namespace cbx {

/// \brief The `ActivationLayer` class represents a layer that encompasses the functionality of an activation
/// function.
///
/// \details
/// The activation function is an essential component of neural network design. It determines whether or not a
/// neuron activates. The goal of an activation function is to introduce non-linearity into the output of a
/// neuron. The type of activation function in the hidden layer determines how well the network model will learn
/// during training.
///
/// The forward pass of this layer performs the subsequent operation.
///
/// Formula: Ô = ζ(Î)
///
/// where:
///  ζ - Activation function
///  Î - Input (Matrix)  : Shape => (m, n)
///  Ô - Output (Matrix) : Shape => (m, n)
///
/// \see Activation
class ActivationLayer : public AbstractLayer {
 private:
  /// \brief The number of neurons in the layer.
  size_type neurons_ = {};

  /// \brief The activation function to be applied.
  ActFuncWrapper act_func_ = {};

 public:
  // /////////////////////////////////////////////
  // Constructors and Destructors
  // /////////////////////////////////////////////

  /// \brief Parameterized constructor.
  /// \param[in] inputs The number of neurons in the input layer.
  /// \param[in] activation The activation to be applied.
  ActivationLayer(size_type inputs, Activation activation);

  /// \brief Default copy constructor.
  /// \param[in] other Source layer.
  ActivationLayer(const ActivationLayer &other) = default;

  /// \brief Move constructor.
  /// \param[in] other Source layer.
  ActivationLayer(ActivationLayer &&other) noexcept;

  /// \brief Default destructor.
  ~ActivationLayer() override = default;

  // /////////////////////////////////////////////
  // Assignment Operators
  // /////////////////////////////////////////////

  /// \brief Default copy assignment operator.
  /// \param[in] other Source layer.
  /// \return A reference to self.
  auto operator=(const ActivationLayer &other) -> ActivationLayer & = default;

  /// \brief Move assignment operator.
  /// \param[in] other Source layer.
  /// \return A reference to self.
  auto operator=(ActivationLayer &&other) noexcept -> ActivationLayer &;

  // /////////////////////////////////////////////
  // Query Functions
  // /////////////////////////////////////////////

  /// \brief Returns the number of neurons in the layer.
  /// \return The number of neurons in the layer.
  [[nodiscard]] auto neurons() const -> size_type override;

  /// \brief Returns the number of modifiable parameters in the layer.
  /// \return The number of modifiable parameters in the layer.
  [[nodiscard]] auto parameters() const -> size_type override;

  /// \brief Returns the type of the layer.
  /// \return The type of the layer.
  ///
  /// \see LayerType
  [[nodiscard]] auto type() const -> LayerType override;

  // /////////////////////////////////////////////
  // Informative
  // /////////////////////////////////////////////

  /// \brief Returns a string with information about the layer's properties.
  /// \return Information about the layer's properties as a string.
  [[nodiscard]] auto property() const -> std::string override;

  // /////////////////////////////////////////////
  // Core Functionality
  // /////////////////////////////////////////////

  /// \brief Forward pass.
  /// \param[in] input The input layer.
  /// \return The output layer.
  [[nodiscard]] auto forward_pass(const container &input) const -> container override;
};

}

#endif
