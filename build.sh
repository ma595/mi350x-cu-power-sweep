#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${root_dir}/build"
rocm_path="${ROCM_PATH:-/opt/rocm}"
hipcc="${rocm_path}/bin/hipcc"
hip_compiler="${rocm_path}/llvm/bin/clang++"
gcc_install_dir="${GCC_INSTALL_DIR:-$(dirname -- "$(gcc -print-libgcc-file-name)")}"
cmake_bin="${CMAKE_BIN:-/usr/bin/cmake}"

if [[ ! -x "${hipcc}" || ! -x "${hip_compiler}" ]]; then
  echo "ROCm compiler not found under ${rocm_path}" >&2
  exit 1
fi

if [[ ! -x "${cmake_bin}" ]]; then
  cmake_bin="$(command -v cmake)"
fi

"${cmake_bin}" -S "${root_dir}" -B "${build_dir}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_COMPILER="${hip_compiler}" \
  -DCMAKE_PREFIX_PATH="${rocm_path}" \
  -DCMAKE_HIP_FLAGS="--gcc-install-dir=${gcc_install_dir}" \
  -DCMAKE_HIP_ARCHITECTURES=gfx950
"${cmake_bin}" --build "${build_dir}" --parallel "$(nproc)"

"${build_dir}/cu_power_bench" --list
