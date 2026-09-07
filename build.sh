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

hipcc_version="$("${hipcc}" --version 2>&1)"
if [[ "${hipcc_version}" != *"HIP version: 6.1."* || \
      "${hipcc_version}" != *"roc-6.1.0"* ]]; then
  echo "ROCm 6.1.0 is required. Found:" >&2
  echo "${hipcc_version}" >&2
  exit 1
fi

"${cmake_bin}" -S "${root_dir}" -B "${build_dir}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_COMPILER="${hip_compiler}" \
  -DCMAKE_PREFIX_PATH="${rocm_path}" \
  -DCMAKE_HIP_FLAGS="--gcc-install-dir=${gcc_install_dir}" \
  -DCMAKE_HIP_ARCHITECTURES=gfx942
"${cmake_bin}" --build "${build_dir}" --parallel "$(nproc)"

"${build_dir}/cu_power_bench" --list
