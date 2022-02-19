#include <cbrainx/cbrainx.hh>

#include <iostream>

#include <fmt/format.h>

auto main() -> cbx::i32 {
  std::cout << std::boolalpha;

  auto s = cbx::Shape{3, 2};

  auto t0 = cbx::Tensor<cbx::i32>::zeros(s);
  auto t1 = cbx::Tensor<cbx::i32>::ones({3, 2, 3});
  auto t2 = cbx::Tensor<cbx::i32>::fill({3, 2, 3}, 5);
  auto t3 = cbx::Tensor<cbx::f32>::random({3, 2});
  auto t4 = cbx::Tensor<cbx::i32>::custom({2, 4}, [n = 1]() mutable {
    return n *= 2;
  });
  auto t5 = cbx::Tensor<cbx::i32>::random({1, 2, 3, 4, 5, 6});

  fmt::print("zeros={{{}}}\n", fmt::join(t0, ", "));
  fmt::print("ones={{{}}}\n", fmt::join(t1, ", "));
  fmt::print("fill={{{}}}\n", fmt::join(t2, ", "));
  fmt::print("random={{{}}}\n", fmt::join(t3, ", "));
  fmt::print("custom={{{}}}\n", fmt::join(t4, ", "));

  std::cout << "Crampy reshaping t5..." << std::endl;
  std::cout << "Before: " << t5.meta_info() << ", After: " << t5.crampy_reshape(3).meta_info() << std::endl;

  std::cout << "Flattening t5..." << std::endl;
  std::cout << "Before: " << t5.meta_info() << ", After: " << t5.flatten().meta_info() << std::endl;

  return {};
}
