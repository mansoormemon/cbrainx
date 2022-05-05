#include <cbrainx/cbrainx.hh>

#include <iostream>

auto main() -> cbx::i32 {
  std::cout << std::boolalpha;

  auto optimizer = cbx::GradientDescent{0.01, 0.025};

  auto epochs = 25;
  for (cbx::i32 e = {}; e < epochs; ++e) {
    ++optimizer;
    std::cout << optimizer.meta_info() << std::endl;
  }

  return {};
}
