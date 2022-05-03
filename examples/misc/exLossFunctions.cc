#include <cbrainx/cbrainx.hh>

#include <iostream>

auto main() -> cbx::i32 {
  std::cout << std::boolalpha;

  {
    auto _y_true = {12.12, 1.2, 23.2, 32.4, 32.7, 36.4, 23.2, 32.3};
    auto _y_pred = {11.12, 1.3, 22.3, 31.2, 34.3, 34.4, 23.1, 23.4};

    auto y_true = cbx::Tensor{{2, 4}, _y_true};
    auto y_pred = cbx::Tensor{{2, 4}, _y_pred};

    auto MSE = cbx::MeanSquaredError{};
    std::cout << "M.S.E. = " << MSE(y_true, y_pred) << std::endl;
    std::cout << "Derivative = " << MSE.derivative(y_true, y_pred) << std::endl;
  }

  {
    auto _y_true = {1, 1, 0, 0, 1, 0, 1, 0};
    auto _y_pred = {0.6, 0.9, 0.3, 0.1, 0.8, 0.2, 0.7, 0.1};

    auto y_true = cbx::Tensor{{2, 4}, _y_true};
    auto y_pred = cbx::Tensor{{2, 4}, _y_pred};

    auto BCE = cbx::BinaryCrossEntropy{};
    std::cout << "B.C.E. = " << BCE(y_true, y_pred) << std::endl;
    std::cout << "Derivative = " << BCE.derivative(y_true, y_pred) << std::endl;
  }

  {
    auto _y_true = {1, 0, 0, 0, 0, 0, 1, 0};
    auto _y_pred = {0.6, 0.1, 0.2, 0.1, 0.1, 0.1, 0.7, 0.1};

    auto y_true = cbx::Tensor{{2, 4}, _y_true};
    auto y_pred = cbx::Tensor{{2, 4}, _y_pred};

    auto CCE = cbx::CategoricalCrossEntropy{};
    std::cout << "C.C.E. = " << CCE(y_true, y_pred) << std::endl;
    std::cout << "Derivative = " << CCE.derivative(y_true, y_pred) << std::endl;
  }

  {
    auto _y_true = {0, 2};
    auto _y_pred = {0.6, 0.1, 0.2, 0.1, 0.1, 0.1, 0.7, 0.1};

    auto y_true = cbx::Tensor{{2}, _y_true};
    auto y_pred = cbx::Tensor{{2, 4}, _y_pred};

    auto SCE = cbx::SparseCrossEntropy{};
    std::cout << "S.C.E. = " << SCE(y_true, y_pred) << std::endl;
    std::cout << "Derivative = " << SCE.derivative(y_true, y_pred) << std::endl;
  }

  return {};
}
