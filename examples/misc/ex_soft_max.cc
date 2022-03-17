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

  auto v0 = {1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0};
  auto in = cbx::Tensor<cbx::f32>::copy({2, 7}, v0.begin(), v0.end());

  // A cute little neural network.
  auto net = cbx::NeuralNetwork{7};
  auto l1 = net.add<cbx::DenseLayer>(14);
  net.add<cbx::ActivationLayer>(cbx::Activation::Gaussian);
  net.add<cbx::DenseLayer>(7);
  net.add<cbx::ActivationLayer>(cbx::Activation::Swish);
  net.add<cbx::SoftMax>();
  net.show_summary();

  auto stopwatch = cbx::Stopwatch{};
  stopwatch.start();
  auto out = net.forward_pass(in);
  stopwatch.stop();

  std::cout << "in: " << in.meta_info() << " = ";
  print_mat(in);

  std::cout << "out: " << out.meta_info() << " = ";
  print_mat(out);

  std::cout << "l1: " << l1->to_string() << " = ";
  print_mat(l1->output());

  std::cout << "Time taken: " << stopwatch.get_interval<std::chrono::microseconds>() << " microseconds."
            << std::endl;

  return {};
}
