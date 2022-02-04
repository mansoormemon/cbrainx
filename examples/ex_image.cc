#include <cbrainx/cbrainx.hh>

#include <iostream>

auto main() -> cbx::i32 {
  std::cout << std::boolalpha;

  auto shape = cbx::Shape{100, 100};
  auto img = cbx::Tensor<cbx::f32>::custom(shape, [n = 0.0](const auto &) mutable { return n += 0.0001; });
  std::cout << "img=" << img.meta_info() << std::endl;
  cbx::Image::write(img, "out.png");
  return {};
}
