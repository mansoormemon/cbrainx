#include <cbrainx/cbrainx.hh>

#include <iostream>

auto main() -> cbx::i32 {
  std::cout << std::boolalpha;

  auto t0 = cbx::Tensor{};
  std::cout << "t0=" << t0.meta_info() << std::endl;

  auto t1 = cbx::Tensor{cbx::Shape{6, 7, 3}};
  std::cout << "t1=" << t1.meta_info() << std::endl;

  auto t2 = cbx::Tensor{{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}};
  std::cout << "t2=" << t2.meta_info() << std::endl;

  std::cout << "Transforming t0 to rank-3 tensor..." << std::endl;
  std::cout << "Before: " << t0.meta_info() << ", After: " << t0.reshape({1, 1, 1}).meta_info() << std::endl;

  std::cout << "Reshaping t1..." << std::endl;
  std::cout << "Before: " << t1.meta_info() << ", After: " << t1.reshape({2, 3, 3, 7}).meta_info() << std::endl;

  std::cout << "Reducing t2 to scalar..." << std::endl;
  std::cout << "Before: " << t2.meta_info() << ", After: " << t2.reshape({}).meta_info() << std::endl;

  // /////////////////////////////////////////////////////////////

  auto t3 = cbx::Tensor{{3, 4}};
  std::cout << "t3=" << t3.meta_info() << std::endl;
  std::cout << "Demonstrating Taylor expansion using t3 for: f(x) = e" << std::endl;

  using t3_value_t = decltype(t3)::value_type;

  auto factorial = [](auto x) {
    auto result = 1;
    for (auto i = x; i > 1; i -= 1) {
      result *= i;
    }
    return result;
  };

  std::generate(t3.begin(), t3.end(), [&factorial, Tn = t3_value_t{}, n = 0]() mutable {
    auto y = std::pow(1.0, n) / factorial(n);
    return n += 1, Tn += y;
  });
  std::cout << "t3 => { " << fmt::format("{}", fmt::join(t3, ", ")) << " }" << std::endl;

  return {};
}
