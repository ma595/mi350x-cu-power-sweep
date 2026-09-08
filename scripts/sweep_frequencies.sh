#!/usr/bin/env bash
# Sweep SCLK across the frequencies in the config, one collector run each.
#
# collect.py only observes clocks - it never sets them - so the frequency axis
# has to be driven from outside. This caps SCLK before each run, records what
# was requested, and reports what was actually achieved.
#
# Usage: scripts/sweep_frequencies.sh [config] [output-base]

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG="${1:-config.json}"
OUTBASE="${2:-results/fsweep}"
SUDO="${SUDO-sudo}"

export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
export PATH="$ROCM_PATH/bin:$PATH"

# ---------------------------------------------------------------- preflight
test -f "$CONFIG" || { echo "no such config: $CONFIG" >&2; exit 1; }
test -x build/cu_power_bench || { echo "build the benchmark first: ./build.sh" >&2; exit 1; }

# A branch switch rewrites source mtimes, so a stale binary is easy to acquire
# and produces results that do not match the checked-out code.
if [ src/cu_power_bench.hip -nt build/cu_power_bench ]; then
  echo "build/cu_power_bench is older than src/cu_power_bench.hip - run ./build.sh" >&2
  exit 1
fi
command -v amd-smi >/dev/null || { echo "amd-smi not found on PATH" >&2; exit 1; }

if ! { test -r /dev/kfd && test -w /dev/kfd; }; then
  echo "no read/write access to /dev/kfd - are you in the render group?" >&2
  echo "a shell opened before 'usermod -aG render' will not have it; log in again" >&2
  exit 1
fi

# amd-smi indexes GPUs by PCI address; ROCr (and so the config's `device`)
# indexes by KFD node. Resolve to a PCI address so we cannot cap the wrong card.
eval "$(python3 - "$CONFIG" <<'PY'
import json, sys
sys.path.insert(0, "scripts")
from collect import kfd_gpu_addresses

config = json.load(open(sys.argv[1]))
device = int(config["device"])
addresses = kfd_gpu_addresses()
if device >= len(addresses):
    sys.exit(f"device {device} out of range; KFD reports {len(addresses)} GPUs")
frequencies = [int(f) for f in config["sclk_reference_values_mhz"]]
print(f"DEVICE={device}")
print(f"BDF={addresses[device]}")
print(f'FREQUENCIES="{" ".join(str(f) for f in frequencies)}"')
PY
)"

# ------------------------------------------------------------------- cleanup
reset_clocks() {
  # Must be safe to call before BDF is assigned: under `set -u` an unbound
  # dereference here kills the trap and leaves a cap in place.
  [ -n "${BDF:-}" ] || return 0
  $SUDO amd-smi reset -g "$BDF" -c >/dev/null 2>&1 || true
  $SUDO amd-smi reset -g "$BDF" -d >/dev/null 2>&1 || true
}
trap 'status=$?; echo; echo "resetting clocks on ${BDF:-<unset>}"; reset_clocks; exit $status' EXIT
trap 'exit 130' INT TERM

echo "device $DEVICE -> $BDF"
echo "frequencies: $FREQUENCIES"
echo

# ---------------------------------------------------------------- the sweep
for frequency in $FREQUENCIES; do
  out="$OUTBASE/sclk_$(printf '%05d' "$frequency")"

  if [ -f "$out/state.json" ] && grep -q '"status": *"completed"' "$out/state.json"; then
    echo "== ${frequency} MHz: already completed, skipping"
    continue
  fi

  echo "== ${frequency} MHz -> $out"
  mkdir -p "$out"

  # amd-smi exits 0 even when it refuses the request, so the output has to be
  # inspected. `-L sclk max` cannot be set equal to the DPM minimum, so the
  # bottom of the range needs performance determinism instead.
  method="clk-limit"
  reply="$($SUDO amd-smi set -g "$BDF" -L sclk max "$frequency" 2>&1 || true)"
  if grep -qiE 'not_supported|unable to set' <<<"$reply"; then
    echo "   -L sclk max refused (${reply//$'\n'/ }); falling back to determinism"
    method="perf-determinism"
    reply="$($SUDO amd-smi set -g "$BDF" -d "$frequency" 2>&1 || true)"
    if grep -qiE 'not_supported|unable to' <<<"$reply"; then
      echo "   could not set ${frequency} MHz by either method; skipping"
      echo "   ${reply//$'\n'/ }"
      continue
    fi
  fi

  # Make each output directory self-describing: the request lives with the data,
  # not only in the directory name.
  cat > "$out/clock_request.json" <<JSON
{
  "requested_sclk_mhz": $frequency,
  "device": $DEVICE,
  "bdf": "$BDF",
  "method": "$method"
}
JSON

  if ! python3 -u scripts/collect.py --config "$CONFIG" --output-dir "$out" \
        > "$out/collector.log" 2>&1; then
    echo "   collector exited non-zero; see $out/collector.log"
  fi

  # Did the cap hold? An unhonoured cap yields a full dataset at max clock with
  # no error, so this check is the difference between data and noise.
  python3 - "$out" "$frequency" <<'PY'
import csv, pathlib, sys

out, requested = pathlib.Path(sys.argv[1]), float(sys.argv[2])
summary = out / "summary.csv"
if not summary.exists():
    print("   no summary.csv"); raise SystemExit
rows = [r for r in csv.DictReader(summary.open()) if r["status"] == "ok"]
if not rows:
    print("   no successful measurements"); raise SystemExit
achieved = [float(r["actual_sclk_mhz_mean"]) for r in rows]
lo, hi = min(achieved), max(achieved)
power = [float(r["power_w_mean"]) for r in rows]
note = ""
deviation = (sum(achieved) / len(achieved) - requested) / requested
if deviation > 0.05:
    note = f"  <-- CAP NOT HONOURED (+{deviation * 100:.0f}% over {requested:.0f})"
elif deviation < -0.05:
    note = f"  <-- well under request ({deviation * 100:.0f}%), likely a nearby DPM point"
total = sum(1 for _ in csv.DictReader(summary.open()))
print(f"   {len(rows)}/{total} ok  achieved {lo:.0f}-{hi:.0f} MHz  "
      f"power {min(power):.0f}-{max(power):.0f} W{note}")
PY
  echo
done

echo "sweep complete: $OUTBASE"
