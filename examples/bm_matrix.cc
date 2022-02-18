#include <cbrainx/cbrainx.hh>

#include <iostream>

auto main() -> cbx::i32 {
  std::cout << std::boolalpha;

  auto stopwatch = cbx::Stopwatch{};

  // Extremely large matrices.
  auto mat_a = cbx::Matrix::custom<cbx::i32>(2048, 512, [n = 0]() mutable {
    return n += 1;
  });
  auto mat_b = cbx::Matrix::custom<cbx::i32>(512, 4096, [n = 0]() mutable {
    return n += 2;
  });

  std::cout << "mat_a=" << mat_a.meta_info() << std::endl;
  std::cout << "mat_b=" << mat_b.meta_info() << std::endl;
  std::cout << std::endl;

  std::cout << "[ WITH MULTITHREADING ]" << std::endl;
  stopwatch.start();
  auto product_wm = cbx::Matrix::multiply(mat_a, mat_b);
  stopwatch.stop();
  std::cout << "product_wm=" << product_wm.meta_info() << std::endl;
  std::cout << "Time taken: " << stopwatch.get_interval<std::chrono::seconds>() << " seconds." << std::endl;
  std::cout << std::endl;

  std::cout << "[ WITHOUT MULTITHREADING ]" << std::endl;
  stopwatch.start();
  auto product_wom = cbx::Matrix::multiply(mat_a, mat_b, false);
  stopwatch.stop();
  std::cout << "product_wom=" << product_wom.meta_info() << std::endl;
  std::cout << "Time taken: " << stopwatch.get_interval<std::chrono::seconds>() << " seconds." << std::endl;
  std::cout << std::endl;

  return {};
}