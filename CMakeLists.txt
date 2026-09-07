cmake_minimum_required(VERSION 3.21)

project(mi300x_cu_power_sweep LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_STANDARD_REQUIRED ON)
set(CMAKE_HIP_ARCHITECTURES gfx950 CACHE STRING "AMD GPU targets")

add_executable(cu_power_bench src/cu_power_bench.hip)
target_compile_options(cu_power_bench PRIVATE
  $<$<COMPILE_LANGUAGE:HIP>:-O3>
)

