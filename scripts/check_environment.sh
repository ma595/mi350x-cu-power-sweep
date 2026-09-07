#!/usr/bin/env bash
set -euo pipefail

hipcc_version="$(hipcc --version 2>&1)"
echo "${hipcc_version}"
if [[ "${hipcc_version}" != *"HIP version: 6.1."* || \
      "${hipcc_version}" != *"roc-6.1.0"* ]]; then
  echo "ROCm 6.1.0 is required." >&2
  exit 1
fi

echo
echo "ROCm-SMI:"
rocm-smi --version

echo
echo "GPU and driver:"
rocm-smi \
  --showproductname \
  --showdriverversion \
  --showclocks \
  --showpower \
  --showtemp \
  --showuse

echo
echo "GPU device files:"
ls -l /dev/kfd /dev/dri/renderD*
