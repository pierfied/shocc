cmake_minimum_required(VERSION 3.17)
project(pytorch_test LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)

find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_library(shocc SHARED src/shocc.cpp)
target_link_libraries(shocc "${TORCH_LIBRARIES}")