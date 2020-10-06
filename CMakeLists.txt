cmake_minimum_required(VERSION 3.17)
project(shocc LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)

find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_library(shocc SHARED src/shocc.cu src/wigner.cu src/wigner.cuh src/standard_transforms.cu src/standard_transforms.cuh)
target_link_libraries(shocc "${TORCH_LIBRARIES}")
target_compile_options(shocc PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        --relocatable-device-code=true
        >)