#include <cbrainx/cbrainx.hh>

#include <iostream>

auto main() -> cbx::i32 {
  std::cout << std::boolalpha;

  auto m0 = cbx::Matrix::make();
  auto m1 = cbx::Matrix::make(3, 4);
  auto random = cbx::Matrix::random(4, 2);
  auto custom = cbx::Matrix::custom(3, 4, [n = 0]() mutable {
    return n++;
  });

  std::cout << "m0=" << m0.meta_info() << std::endl;
  std::cout << "m1=" << m1.meta_info() << std::endl;
  std::cout << "random=" << random.meta_info() << std::endl;
  std::cout << "custom=" << custom.meta_info() << std::endl;

  return {};
}
