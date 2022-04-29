#include <cbrainx/cbrainx.hh>

#include <iostream>

auto main() -> cbx::i32 {
  std::cout << std::boolalpha;

  auto stopwatch = cbx::Stopwatch{};

  // Extremely large matrices.
  auto mat_a = cbx::Tensor<cbx::i32>::arange({2048, 512}, 1);
  auto mat_b = cbx::Tensor<cbx::f32>::arange({512, 4096}, 2, 2.14);

  std::cout << "mat_a=" << mat_a.meta_info() << std::endl;
  std::cout << "mat_b=" << mat_b.meta_info() << std::endl;
  std::cout << std::endl;

  std::cout << "[ WITH MULTITHREADING ]" << std::endl;
  stopwatch.start();
  auto product_wm = mat_a.matmul(mat_b);
  stopwatch.stop();
  std::cout << "product_wm=" << product_wm.meta_info() << std::endl;
  std::cout << "Time taken: " << stopwatch.get_duration<std::chrono::seconds>() << " seconds." << std::endl;
  std::cout << std::endl;

  std::cout << "[ WITHOUT MULTITHREADING ]" << std::endl;
  stopwatch.start();
  auto product_wom = mat_a.matmul(mat_b, false);
  stopwatch.stop();
  std::cout << "product_wom=" << product_wom.meta_info() << std::endl;
  std::cout << "Time taken: " << stopwatch.get_duration<std::chrono::seconds>() << " seconds." << std::endl;
  std::cout << std::endl;

  return {};
}
