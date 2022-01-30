#include <cbrainx/cbrainx.hh>

#include <iostream>

auto main() -> cbx::i32 {
  std::cout << std::boolalpha;

  auto s0 = cbx::Shape{704, 640, 3};
  std::cout << s0.meta_info() << ", s0=" << s0.to_string() << std::endl;

  auto s1 = cbx::Shape{};
  std::cout << s1.meta_info() << ", s1=" << s1.to_string() << std::endl;

  auto s2 = cbx::Shape{168960, 8};
  std::cout << s2.meta_info() << ", s2=" << s2.to_string() << std::endl;

  auto s3 = cbx::Shape{1, 1, 1, 1, 1, 1, 1, 1};
  std::cout << s3.meta_info() << ", s3=" << s3.to_string() << std::endl;

  std::cout << "s0 is compatible with s2: " << s0.is_compatible(s2) << std::endl;
  std::cout << "s1 is compatible with s3: " << s1.is_compatible(s3) << std::endl;

  auto [rows, cols, channels] = cbx::Shape{4, 3, 3}.unwrap<3>();
  std::cout << "Rows: " << rows << ", Columns: " << cols << ", Channels: " << channels << std::endl;

  return {};
}
