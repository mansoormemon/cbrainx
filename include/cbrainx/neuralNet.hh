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

#ifndef CBRAINX__NEURAL_NET_HH_
#define CBRAINX__NEURAL_NET_HH_

#include <list>
#include <memory>

#include "abstractLayer.hh"
#include "lossFunctions.hh"
#include "optimizers.hh"
#include "typeAliases.hh"

namespace cbx {

/// \brief A constraint to filter semantic data types representing a layer.
/// \tparam T The data type to which the constraint is to be applied.
///
/// \details
/// This constraint only allows the child classes of `AbstractLayer`.
template <typename T>
concept ConcreteLayer = std::is_base_of_v<AbstractLayer, T> and not std::is_abstract_v<T>;

/// \brief The `NeuralNet` class represents a network of simulated neurons called an artificial neural network.
///
/// \details
/// A neural network (also known as an artificial neural network) is composed of simulated neurons or nodes used
/// to unravel problems that require human-like intelligence. These simulated neurons are devised in multiple
/// layers to shape the network's architecture. A variety of connection patterns are conceivable between any two
/// layers. The neurons in one layer can only communicate with the neurons in the immediately preceding and
/// following layers. The input layer is the one that receives external data, whereas the output layer is the
/// one that yields the final output. In between, there could be zero or more hidden layers. The role of these
/// hidden layers is to extract traits from the input data and use these to correlate between a given input and
/// the correct output to make the final prediction.
///
/// \note The `NeuralNet` object owns all the layers associated with it.
///
/// \see LayerType
class NeuralNet {
 public:
  using value_type = std::shared_ptr<AbstractLayer>;

  using reference = value_type &;
  using const_reference = const value_type &;

  using pointer = value_type *;
  using const_pointer = const value_type *;

  using container = std::list<value_type>;

  using size_type = typename container::size_type;
  using difference_type = typename container::difference_type;

  using iterator = typename container::iterator;
  using const_iterator = typename container::const_iterator;

  using reverse_iterator = typename container::reverse_iterator;
  using const_reverse_iterator = typename container::const_reverse_iterator;

  using tensor_type = Tensor<f32>;

 private:
  /// \brief The shape of the input layer (excluding the samples axis).
  Shape input_shape_ = {};

  /// \brief A doubly-linked list of layers.
  container layers_ = {};

  // /////////////////////////////////////////////
  // Helpers
  // /////////////////////////////////////////////

  /// \brief Validates the input shape.
  /// \param[in] shape The shape to be validated.
  ///
  /// \details
  /// This function throws an exception if the input shape represents a scalar.
  ///
  /// \throws RankError
  static auto _s_validate_input_shape(const Shape &shape) -> void;

  /// \brief Matches the input shape of the network with \p shape.
  /// \param[in] shape The shape to be matched.
  ///
  /// \details
  /// This function throws an exception if the input tensor's shape does not match the input shape of the
  /// network.
  ///
  /// \throws ShapeError
  auto _m_match_input_shape(const Shape &shape) const -> void;

 public:
  // /////////////////////////////////////////////
  // Constructors and Destructors
  // /////////////////////////////////////////////

  /// \brief Parameterized constructor.
  /// \param[in] input_shape The shape of the input layer (excluding the samples axis).
  ///
  /// \details
  /// This function throws an exception if the input shape represents a scalar.
  ///
  /// \throws RankError
  explicit NeuralNet(const Shape &input_shape);

  /// \brief Default copy constructor.
  /// \param[in] other Source network.
  ///
  /// \note This constructor performs a shallow copy.
  NeuralNet(const NeuralNet &other) = default;

  /// \brief Move constructor.
  /// \param[in] other Source network.
  NeuralNet(NeuralNet &&other) noexcept;

  /// \brief Default destructor.
  ~NeuralNet() = default;

  // /////////////////////////////////////////////
  // Assignment Operators
  // /////////////////////////////////////////////

  /// \brief Default copy assignment operator.
  /// \param[in] other Source network.
  /// \return A reference to self.
  ///
  /// \note This function performs a shallow copy.
  auto operator=(const NeuralNet &other) -> NeuralNet & = default;

  /// \brief Move assignment operator.
  /// \param[in] other Source network.
  /// \return A reference to self.
  auto operator=(NeuralNet &&other) noexcept -> NeuralNet &;

  // /////////////////////////////////////////////
  // Element Access
  // /////////////////////////////////////////////

  /// \brief Accesses the first layer.
  /// \return An immutable reference to the first layer.
  ///
  /// \note Behavior is undefined if the neural network has no layers.
  [[nodiscard]] auto front() const -> const_reference;

  /// \brief Accesses the first layer.
  /// \return A mutable reference to the first layer.
  ///
  /// \note Behavior is undefined if the neural network has no layers.
  auto front() -> reference;

  /// \brief Accesses the last layer.
  /// \return A immutable reference to the first layer.
  ///
  /// \note Behavior is undefined if the neural network has no layers.
  [[nodiscard]] auto back() const -> const_reference;

  /// \brief Accesses the last layer.
  /// \return A mutable reference to the last layer.
  ///
  /// \note Behavior is undefined if the neural network has no layers.
  auto back() -> reference;

  // /////////////////////////////////////////////
  // Iterators
  // /////////////////////////////////////////////

  /// \brief Returns an immutable bidirectional iterator pointing to the first layer of the network.
  /// \return An immutable iterator pointing to the beginning of the container.
  [[nodiscard]] auto cbegin() const -> const_iterator;

  /// \brief Returns a bidirectional iterator pointing to the first layer of the network.
  /// \return An immutable iterator pointing to the beginning of the container.
  [[nodiscard]] auto begin() const -> const_iterator;

  /// \brief Returns a bidirectional iterator pointing to the first layer of the network.
  /// \return A mutable iterator pointing to the beginning of the container.
  auto begin() -> iterator;

  /// \brief Returns an immutable bidirectional iterator pointing to the last layer of the network.
  /// \return An immutable reverse iterator pointing to the reverse beginning of the container.
  [[nodiscard]] auto crbegin() const noexcept -> const_reverse_iterator;

  /// \brief Returns a bidirectional iterator pointing to the last layer of the network.
  /// \return An immutable reverse iterator pointing to the reverse beginning of the container.
  [[nodiscard]] auto rbegin() const noexcept -> const_reverse_iterator;

  /// \brief Returns a bidirectional iterator pointing to the last layer of the network.
  /// \return A mutable reverse iterator pointing to the reverse beginning of the container.
  auto rbegin() noexcept -> reverse_iterator;

  /// \brief Returns an immutable bidirectional iterator pointing to the last layer of the network.
  /// \return An immutable iterator pointing to the ending of the container.
  [[nodiscard]] auto cend() const -> const_iterator;

  /// \brief Returns a bidirectional iterator pointing to the last layer of the network.
  /// \return An immutable iterator pointing to the ending of the container.
  [[nodiscard]] auto end() const -> const_iterator;

  /// \brief Returns a bidirectional iterator pointing to the last layer of the network.
  /// \return A mutable iterator pointing to the ending of the container.
  auto end() -> iterator;

  /// \brief Returns an immutable bidirectional iterator pointing to the first layer of the network.
  /// \return An immutable reverse iterator pointing to the reverse ending of the container.
  [[nodiscard]] auto crend() const noexcept -> const_reverse_iterator;

  /// \brief Returns a bidirectional iterator pointing to the first layer of the network.
  /// \return An immutable reverse iterator pointing to the reverse ending of the container.
  [[nodiscard]] auto rend() const noexcept -> const_reverse_iterator;

  /// \brief Returns a bidirectional iterator pointing to the first layer of the network.
  /// \return A mutable reverse iterator pointing to the reverse ending of the container.
  auto rend() noexcept -> reverse_iterator;

  // /////////////////////////////////////////////
  // Query Functions
  // /////////////////////////////////////////////

  /// \brief Returns the number of layers in the network.
  /// \return The number of layers in the network.
  [[nodiscard]] auto size() const -> size_type;

  /// \brief Returns the total number of trainable parameters in the network.
  /// \return The total number of trainable parameters.
  [[nodiscard]] auto total_parameters() const -> size_type;

  // /////////////////////////////////////////////
  // Informative
  // /////////////////////////////////////////////

  /// \brief Prints a summary of the network.
  auto show_summary() const -> void;

  // /////////////////////////////////////////////
  // Modifiers
  // /////////////////////////////////////////////

  /// \brief Adds a new layer to the neural network.
  /// \tparam L Type of the layer.
  /// \tparam Args The data type of arguments.
  /// \param[in] args Parameter list for the constructor of \p L.
  /// \return An immutable reference to the newly created layer, i.e., the last layer.
  template <ConcreteLayer L, typename... Args>
  auto add(Args... args) -> const_reference {
    auto [input_neurons] = input_shape_.unwrap<1>();
    auto previous_layer_size = layers_.empty() ? input_neurons : layers_.back()->neurons();
    layers_.emplace_back(std::make_shared<L>(previous_layer_size, args...));
    layers_.back()->set_id(i32(layers_.size()));
    return layers_.back();
  }

  /// \brief Pops the last layer from the network.
  auto pop() -> void;

  // /////////////////////////////////////////////
  // Core Functionality
  // /////////////////////////////////////////////

  /// \brief Forward pass.
  /// \param[in] input The input layer.
  /// \return The output layer.
  ///
  /// \details
  /// This function throws an exception if the input tensor's shape does not match the input shape of the
  /// network.
  ///
  /// \throws ShapeError
  [[nodiscard]] auto forward_pass(tensor_type input) const -> tensor_type;

  /// \brief Backward pass.
  /// \param[in] x The input data.
  /// \param[in] y The truth values.
  /// \param[in] epochs The number of epochs.
  /// \param[in] batch_size The size of each batch.
  /// \param[in] loss_type The type of the loss function.
  /// \param[in] optimizer The optimizer.
  auto backward_pass(tensor_type x, tensor_type y, usize epochs, usize batch_size, Loss loss_type,
                     OptimizerWrapper optimizer) -> void;
};

}

#endif
