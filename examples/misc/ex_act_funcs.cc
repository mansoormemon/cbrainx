#include <cbrainx/cbrainx.hh>

#include <iostream>

auto print_mat(const auto &mat) -> void {
  auto [rows, cols] = mat.shape().template unwrap<2>();
  std::cout << '[' << std::endl;
  for (auto r = 0UL; r < rows; ++r) {
    for (auto c = 0UL; c < cols; ++c) {
      std::cout << mat(r, c) << std::string(4, ' ');
    }
    std::cout << std::endl;
  }
  std::cout << ']' << std::endl;
};

auto main() -> cbx::i32 {
  std::cout << std::boolalpha;

  auto v0 = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
  auto in = cbx::Tensor<cbx::f32>{{2, 8}, v0.begin()};

  // A cute little neural network.
  auto net = cbx::NeuralNet{{8}};
  net.add<cbx::ActivationLayer>(cbx::Activation::ReLU);
  net.add<cbx::ActivationLayer>(cbx::Activation::Linear);
  net.add<cbx::ActivationLayer>(cbx::Activation::Sigmoid);
  net.add<cbx::ActivationLayer>(cbx::Activation::TanH);
  net.add<cbx::ActivationLayer>(cbx::Activation::GELU);
  net.add<cbx::ActivationLayer>(cbx::Activation::Gaussian);
  net.add<cbx::ActivationLayer>(cbx::Activation::Swish);
  net.add<cbx::ActivationLayer>(cbx::Activation::Softplus);
  net.show_summary();

  auto out = net.forward_pass(in);

  std::cout << "in: " << in.meta_info() << " = ";
  print_mat(in);

  std::cout << "out: " << out.meta_info() << " = ";
  print_mat(out);

  return {};
}
