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

#ifndef CBRAINX__DENSE_LAYER_HH_
#define CBRAINX__DENSE_LAYER_HH_

#include "abstract_layer.hh"
#include "tensor.hh"
#include "type_aliases.hh"

namespace cbx {

/// \brief The `DenseLayer` class represents a fully connected dense layer.
///
/// \details
/// A dense layer is a fully connected layer in which each neuron receives input from all the neurons in the
/// previous layer. As a result, each neuron provides one output to the next layer.
///
/// The forward pass of this layer performs the subsequent operation.
///
/// Formula: Ô = Î ⊙ Ŵ + Ƀ
///
/// where:
///  Î - Input (Matrix)   : Shape => (m, n)
///  Ŵ - Weights (Matrix) : Shape => (n, o)
///  Ƀ - Biases (Vector)  : Shape => (o)
///  Ô - Output (Matrix)  : Shape => (m, o)
///
/// and, the symbol `⊙` denotes dot product (typically matrix multiplication).
///
/// \see LayerType AbstractLayer
class DenseLayer : public AbstractLayer {
 private:
  /// \brief A tensor of trainable weights.
  container weights_ = {};

  /// \brief A tensor of trainable biases.
  container biases_ = {};

 public:
  // /////////////////////////////////////////////
  // Constructors and Destructors
  // /////////////////////////////////////////////

  /// \brief Parameterized constructor.
  /// \param[in] input_size The number of neurons in the input layer.
  /// \param[in] neurons The number of neurons in this layer.
  DenseLayer(size_type input_size, size_type neurons);

  /// \brief Default copy constructor.
  /// \param[in] other Source layer.
  DenseLayer(const DenseLayer &other) = default;

  /// \brief Default move constructor.
  /// \param[in] other Source layer.
  DenseLayer(DenseLayer &&other) noexcept;

  /// \brief Default destructor.
  ~DenseLayer() override = default;

  // /////////////////////////////////////////////
  // Assignment Operators
  // /////////////////////////////////////////////

  /// \brief Default copy assignment operator.
  /// \param[in] other Source layer.
  /// \return A reference to self.
  auto operator=(const DenseLayer &other) -> DenseLayer & = default;

  /// \brief Default move assignment operator.
  /// \param[in] other Source layer.
  /// \return A reference to self.
  auto operator=(DenseLayer &&other) noexcept -> DenseLayer &;

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
  /// \return A reference to self.
  [[nodiscard]] auto forward_pass(const container &input) -> container override;
};

}

#endif
