cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(curadon LANGUAGES CXX CUDA)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# add our cmake modules under cmake/
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

include(CPM)

if(CMAKE_BUILD_TYPE MATCHES Debug)
  set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G")
endif()

set(CMAKE_CUDA_ARCHITECTURES "native")

find_package(CUDAToolkit REQUIRED)

add_library(curadon src/curadon.cu)
target_include_directories(curadon PUBLIC include)
target_link_libraries(curadon PUBLIC CUDA::cudart)

target_compile_options(
  curadon PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda
                 --expt-relaxed-constexpr>)

cpmaddpackage("gh:onqtam/doctest@2.4.11")

find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)

add_executable(
  tests
  test/test_main.cpp
  test/test_device_buffer.cpp
  test/test_device_uvector.cpp
  test/test_kernel.cpp
  test/test_intersection.cu
  test/test_traversal.cu
  test/test_parallel.cu
  test/test_fanbeam.cu
  test/test_backprojection.cu
  test/test_forward_projection.cu)
target_link_libraries(tests curadon doctest OpenGL::GL GLEW::glew glfw)

add_subdirectory(benchmark)
