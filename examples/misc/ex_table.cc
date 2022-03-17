#include <cbrainx/cbrainx.hh>

#include <iostream>

auto main() -> cbx::i32 {
  std::cout << std::boolalpha;

  auto table = cbx::Table{"ID", "Name", "Department", "Batch"};

  table.add({"CS1", "John", "CS", "2021"});
  table.add({"CS2", "Mary", "CS", "2021"});
  table.add({"CS3", "Alex", "CS", "2021"});
  table.add({"CS4", "Sean", "CS", "2021"});

  std::cout << table.meta_info() << std::endl;

  table.show();

  return {};
}
