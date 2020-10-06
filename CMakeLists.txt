cmake_minimum_required(VERSION 3.17)
project(shocc LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)

find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")

add_library(driver SHARED src/shocc.cu src/wigner.cu src/wigner.cuh src/standard_transforms.cu src/standard_transforms.cuh)
target_link_libraries(driver "${TORCH_LIBRARIES}" ${TORCH_PYTHON_LIBRARY} healpix_cxx)
set_target_properties(driver PROPERTIES CUDA_SEPARABLE_COMPILATION ON PREFIX "")