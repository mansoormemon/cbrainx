#include <cbrainx/cbrainx.hh>

#include <iostream>

auto main() -> cbx::i32 {
  std::cout << std::boolalpha;

  auto img = cbx::Tensor<cbx::f32>::custom({100, 100}, [n = 0.0]() mutable {
    return n += 0.0001;
  });
  cbx::ImgProc::binarize(img);
  cbx::ImgProc::invert(img);
  img = cbx::ImgProc::rescale(img, 3);
  cbx::Image::write(img, "binarized.png");

  return {};
}
