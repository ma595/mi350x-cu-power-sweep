cmake_minimum_required(VERSION 3.21)

project(mi300x_cu_power_sweep LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_STANDARD_REQUIRED ON)
set(CMAKE_HIP_ARCHITECTURES gfx950 CACHE STRING "AMD GPU targets")

# ROCm install prefix, for the runtime library search path below.
if(NOT DEFINED ROCM_PATH)
  if(DEFINED ENV{ROCM_PATH})
    set(ROCM_PATH $ENV{ROCM_PATH})
  else()
    set(ROCM_PATH /opt/rocm)
  endif()
endif()

add_executable(cu_power_bench src/cu_power_bench.hip)
target_compile_options(cu_power_bench PRIVATE
  $<$<COMPILE_LANGUAGE:HIP>:-O3>
)

# libamdhip64 is linked implicitly by the HIP language, so CMake derives no
# RPATH for it and /opt/rocm/lib is not on the default loader path. Record it
# explicitly. Use ${ROCM_PATH}/lib rather than its resolved target so the
# binary follows the /etc/alternatives symlink across ROCm version switches.
set_target_properties(cu_power_bench PROPERTIES
  BUILD_RPATH "${ROCM_PATH}/lib"
)
