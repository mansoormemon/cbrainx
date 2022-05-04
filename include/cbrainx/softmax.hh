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

#ifndef CBRAINX__SOFT_MAX_HH_
#define CBRAINX__SOFT_MAX_HH_

#include "abstractLayer.hh"
#include "tensor.hh"
#include "typeAliases.hh"

namespace cbx {

/// \brief The `Softmax` class represents the softmax activation function implemented as a custom layer.
///
/// \details
/// The softmax (or softargmax) function is a multidimensional variant of the Sigmoid (or logistic) function.
/// Based on Luce's choice axiom, it is used in multinomial logistic regression as the activation function of
/// the final layer (or, in this case, as the final layer itself) in a neural network to normalize the network's
/// output to a probability distribution over potential output classes.
///
/// The forward pass of this layer performs the subsequent operation.
///
/// Formula: Ō = σ(Ƶ)ὶ [ὶ = 1, ƙ] = ęᶼ / ⅀ [ʝ = 1, ƙ] ęᶽ
///
/// where:
///  σ - Softmax function
///  Ƶ - Input (Vector) : Shape => (k)
///  ʐ - ὶᵗʰ element in the vector
///  ʑ - ʝᵗʰ element in the vector
///  ƙ = Number of classes in the multi-class classifier
///  Ō = Output (Vector) : Shape => (k)
///
/// It should be noted that the formula above only pertains to one sample (along the x-axis). The actual
/// implementation iterates along the y-axis and applies the above formula to each sample individually.
///
/// \note Although softmax is an activation function, it is implemented as a custom layer due to design
/// constraints.
///
/// \see LayerType AbstractLayer
class Softmax : public AbstractLayer {
 private:
  /// \brief The number of neurons in the layer.
  size_type neurons_ = {};

 public:
  // /////////////////////////////////////////////
  // Constructors and Destructors
  // /////////////////////////////////////////////

  /// \brief Parameterized constructor.
  /// \param[in] inputs The number of neurons in the input layer.
  explicit Softmax(size_type inputs);

  /// \brief Default copy constructor.
  /// \param[in] other Source layer.
  Softmax(const Softmax &other) = default;

  /// \brief Move constructor.
  /// \param[in] other Source layer.
  Softmax(Softmax &&other) noexcept;

  /// \brief Default destructor.
  ~Softmax() override = default;

  // /////////////////////////////////////////////
  // Assignment Operators
  // /////////////////////////////////////////////

  /// \brief Default copy assignment operator.
  /// \param[in] other Source layer.
  /// \return A reference to self.
  auto operator=(const Softmax &other) -> Softmax & = default;

  /// \brief Move assignment operator.
  /// \param[in] other Source layer.
  /// \return A reference to self.
  auto operator=(Softmax &&other) noexcept -> Softmax &;

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
  [[nodiscard]] auto forward_pass(const container &input) const -> const AbstractLayer & override;
};

}

#endif
