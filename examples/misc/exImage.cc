#include <cbrainx/cbrainx.hh>

#include <iostream>

auto main() -> cbx::i32 {
  std::cout << std::boolalpha;

  auto img = cbx::Tensor<cbx::f32>::arange({100, 100}, 0, 0.0001);
  std::cout << "img=" << img.meta_info() << std::endl;
  cbx::Image::write(img, "out.jpg");

  return {};
}
