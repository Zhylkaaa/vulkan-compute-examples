cmake_minimum_required(VERSION 3.15)
project(vulkan_compute_examples)

set(CMAKE_CXX_STANDARD 17)

find_package(Vulkan REQUIRED)

add_subdirectory(vector_add)
add_subdirectory(matmult)