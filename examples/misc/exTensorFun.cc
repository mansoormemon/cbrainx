#include <cbrainx/cbrainx.hh>

#include <iostream>

#include <fmt/format.h>

auto main() -> cbx::i32 {
  std::cout << std::boolalpha;

  auto s = cbx::Shape{3, 2};

  auto t0 = cbx::Tensor<cbx::f32>::random({3, 2});
  auto t1 = cbx::Tensor<cbx::i32>::custom({2, 4}, [n = 1]() mutable {
    return n *= 2;
  });
  auto t3 = cbx::Tensor<cbx::i32>::random({1, 2, 3, 4, 5, 6});

  fmt::print("random={{{}}}\n", fmt::join(t0, ", "));
  fmt::print("custom={{{}}}\n", fmt::join(t1, ", "));

  std::cout << "Reshaping t3..." << std::endl;
  std::cout << "Before: " << t3.meta_info() << ", After: " << t3.reshape(3).meta_info() << std::endl;

  std::cout << "Flattening t3..." << std::endl;
  std::cout << "Before: " << t3.meta_info() << ", After: " << t3.flatten().meta_info() << std::endl;

  std::cout << "Tensor arithmetic..." << std::endl;
  auto t4 = cbx::Tensor<cbx::f32>::arange({2, 5}, 32, 2.5);
  auto t5 = cbx::Tensor<cbx::i32>::arange({5}, 1);
  fmt::print("t4 = {{ {} }}\n", fmt::join(t4, ", "));
  fmt::print("t5 = {{ {} }}\n", fmt::join(t5, ", "));

  t4 += t5;
  fmt::print("After t4 += t5, t4 = {{ {} }}\n", fmt::join(t4, ", "));

  auto t6 = t4 * t5;
  auto t7 = t5 / t4;
  fmt::print("t6 = t4 * t5 = {{ {} }}\n", fmt::join(t6, ", "));
  fmt::print("t7 = t5 / t4 = {{ {} }}\n", fmt::join(t7, ", "));

  auto t8 = t5 * 2.4;
  fmt::print("t8 = t5 * 2.4 = {{ {} }}\n", fmt::join(t8, ", "));

  auto t9 = cbx::Tensor<cbx::f32>::arange({10}, 1, 0.5);
  auto t10 = cbx::Tensor<cbx::f32>::arange({10}, 0.15, 0.3);
  auto t11 = t10 % t9;
  auto t12 = 2.3 % t10;
  fmt::print("t9 = {{ {} }}\n", fmt::join(t9, ", "));
  fmt::print("t10 = {{ {} }}\n", fmt::join(t10, ", "));
  fmt::print("t11 = t10 % t9 = {{ {} }}\n", fmt::join(t11, ", "));
  fmt::print("t12 = 2.3 % t10 = {{ {} }}\n", fmt::join(t12, ", "));

  std::cout << "Clamping t12 to [0, 0.09]..." << std::endl;
  t12.clamp(0.01, 0.09);
  fmt::print("t12 = {{ {} }}\n", fmt::join(t12, ", "));

  std::cout << "Rounding t12 to 2 decimal places..." << std::endl;
  t12 |= [](auto x) {
    x *= 100;
    x += 0.5;
    x = cbx::i32(x);
    return x / 100;
  };

  fmt::print("t12 = {{ {} }}\n", fmt::join(t12, ", "));

  auto m0 = cbx::Tensor{{1, 5}, 2};
  auto m1 = cbx::Tensor<double>::arange({5, 1}, 1, 2);
  std::cout << m0.meta_info() << std::endl;
  std::cout << m1.meta_info() << std::endl;
  fmt::print("{}\n", fmt::join(m0, ", "));
  fmt::print("{}\n", fmt::join(m1, ", "));

  auto p = m0.matmul(m1);
  std::cout << p.meta_info() << std::endl;
  fmt::print("{}\n", fmt::join(p, ", "));

  return {};
}
