#include <cbrainx/cbrainx.hh>

#include <iostream>

auto main() -> cbx::i32 {
  std::cout << std::boolalpha;

  auto img = cbx::Tensor<cbx::f32>::arange({100, 100, 3}, 0, 0.00003333);
  cbx::ImgProc::binarize(img);
  cbx::ImgProc::invert(img);
  img = cbx::ImgProc::rescale(img, 3);
  cbx::Image::write(img, "binarized.jpg");

  try {
    std::cout << "Attempting to read image..." << std::endl;
    auto sample_img = cbx::Image::read("s0.png");
    std::cout << "Image read successfully!" << std::endl;
    std::cout << "=> " << sample_img.meta_info() << std::endl;

    auto meta = cbx::Image::Meta::decode_shape(sample_img.shape());
    std::cout << "Apply filter..." << std::endl;
    auto filter = cbx::Tensor<cbx::f32>{{meta.channels()}, std::initializer_list<cbx::f32>{0.6, 0.8, 1.1, 0.6}};
    fmt::print("filter => {}\n", fmt::join(filter, ", "));
    sample_img *= filter;

    std::cout << "Binarizing image..." << std::endl;
    cbx::ImgProc::binarize(sample_img);

    std::cout << "Writing image..." << std::endl;
    cbx::Image::write(sample_img, "filtered.png", cbx::Image::Format::PNG);
  } catch (cbx::ImageIOError &e) {
    std::cout << e.what() << std::endl;
    std::cout << "Terminating..." << std::endl;
  }
  return {};
}
