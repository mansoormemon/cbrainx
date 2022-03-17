#include <cbrainx/cbrainx.hh>

#include <iomanip>
#include <iostream>

auto main() -> cbx::i32 {
  std::cout << std::boolalpha;

  auto print_mat = [](const auto &mat) {
    auto [rows, cols] = mat.shape().template unwrap<2>();

    std::cout << '[' << std::endl;
    for (auto r = 0UL; r < rows; ++r) {
      for (auto c = 0UL; c < cols; ++c) {
        std::cout << std::setw(8) << std::left << mat(r, c);
      }
      std::cout << std::endl;
    }
    std::cout << ']' << std::endl;
  };

  auto mat_a = cbx::Matrix::custom<cbx::f32>(8, 4, [n = 0]() mutable {
    return n += 1;
  });
  auto mat_b = cbx::Matrix::custom<cbx::f32>(4, 5, [n = 0.0]() mutable {
    return n += 0.2;
  });

  std::cout << "mat_a=" << mat_a.meta_info() << ":" << std::endl;
  print_mat(mat_a);

  std::cout << "mat_b=" << mat_b.meta_info() << ":" << std::endl;
  print_mat(mat_b);

  auto product = cbx::Matrix::multiply(mat_a, mat_b);
  std::cout << "product=" << product.meta_info() << ":" << std::endl;
  print_mat(product);

  return {};
}
