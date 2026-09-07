#!/usr/bin/env bash
set -euo pipefail

hipcc_version="$(hipcc --version 2>&1)"
echo "${hipcc_version}"
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
