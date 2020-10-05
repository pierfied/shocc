//
// Created by pierfied on 10/5/20.
//

#include <iostream>
#include <torch/extension.h>

void hello() {
    std::cout << "Hello, World!" << std::endl;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hello", &hello);
}