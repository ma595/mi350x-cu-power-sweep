# MI350X CU power sweep

Per-CU power characterisation for AMD Instinct MI350X (gfx950). Five
microkernels of increasing power envelope are run one block per CU, while
power, temperature and clocks are sampled from sysfs.

Measured values for a particular machine — device mapping, idle temperatures,
DPM range — belong in `system/`, not here. See [`system/`](system/) for the
nodes this has been run on.

## Requirements

- A ROCm release with `gfx950` support, including its HIP runtime and LLVM
- CMake >= 3.21, a C++17 host compiler, Python 3 (standard library only)
- Read/write access to `/dev/kfd`, normally via the `render` group

`hipcc --version` reports HIP's own version, which is numbered independently of
the ROCm release — do not use it to identify the ROCm version. Read the release
directly:

```bash
cat /opt/rocm/.info/version 2>/dev/null || cat /opt/rocm/core-*/.info/version
```

If `rocminfo` reports a permission error on `/dev/kfd`:

```bash
sudo usermod -aG render,video "$USER"   # then log out and back in
```

## Workloads

 `SCALAR ADD`
 `SFU TRANSCENDENTAL`
 `MATRIX MFMA FP32`
 `VECTOR FP64 FMA`
 `VECTOR FP32 FMA`

All workloads default to 256 blocks and 1024 threads per block — one block per
CU, MI350X having 256 CUs.

The block count must match the CU count of the part under test. MI300X has 304;
using that count on MI350X oversubscribes 48 CUs, and the run completes and
reports plausible figures rather than failing.

## Build

```bash
export ROCM_PATH=/opt/rocm
export PATH="$ROCM_PATH/bin:$PATH"
./build.sh
```

The output binary is `build/cu_power_bench`, built for `gfx950`. It records an
RPATH of `$ROCM_PATH/lib`, so it needs no `LD_LIBRARY_PATH` at runtime.

## Single GPU only

The benchmark measures one GPU and refuses to start if more than one is
visible, since a second active device would confound the power telemetry.
`collect.py` sets `ROCR_VISIBLE_DEVICES` and `HIP_VISIBLE_DEVICES` from the
`device` field for you. To run the binary by hand, mask it yourself:

```bash
HIP_VISIBLE_DEVICES=0 ./build/cu_power_bench --exact-filter "SCALAR ADD" --duration-s 5
```

To cover several GPUs, run the collector once per device into separate output
directories, sequentially — concurrent runs heat each other and invalidate the
sweep.

## Configure

Edit a copy of `config.example.json`. The two fields that must be set per
machine are `device` and `cool_to_c`.

### `device`

An index into ROCr's enumeration order, which is KFD node order — **not** the
order `rocm-smi` lists GPUs. `rocm-smi` sorts by PCI address; KFD does not, so
on multi-GPU nodes the same index can name different cards in the two spaces.

The collector resolves the index through KFD topology, and the benchmark
reports the PCI address it actually ran on. A mismatch aborts the run rather
than recording another GPU's power. To print the mapping for a node:

```bash
python3 -c "import sys; sys.path.insert(0, 'scripts')
from collect import kfd_gpu_addresses
print(*enumerate(kfd_gpu_addresses()), sep='\n')"
```

### `cool_to_c`

Must sit above the GPU's idle junction temperature, or every measurement fails
on a cooling timeout after `cool_timeout_s`. Idle floors vary substantially
with cooling design and rack density, so measure before a long run:

```bash
rocm-smi --showtemp
```

Pick a value a few degrees above the highest idle reading of any device you
intend to sweep. It still represents a real thermal reset, since these parts
run far hotter under load.

### SCLK reference values

Ten reference values for binning observed samples — not requested clock locks.
`clock_mode` is `observe`; the collector never writes clocks. They should span
the node's DPM range:

```bash
cat /sys/class/drm/card*/device/pp_dpm_sclk
```

### Environment check

```bash
./scripts/check_environment.sh
```

## Test

Validate the schedule without running anything:

```bash
python3 scripts/collect.py --config config.json --output-dir results/run1 --dry-run
```

Run one measurement:

```bash
python3 -u scripts/collect.py --config config.json --output-dir results/run1 --max-runs 1
```

Confirm it captured load rather than an idle GPU. In `summary.csv`,
`gpu_use_pct_mean` should be ~100 and `actual_sclk_mhz_mean` near the top of
the DPM range. **A `status` of `ok` alone is not sufficient** — a measurement
taken against an idle card yields ~0 % utilisation, minimum SCLK and flat idle
power, and still reports `ok`.

## Run

```bash
mkdir -p results/run1
nohup python3 -u scripts/collect.py \
  --config config.json \
  --output-dir results/run1 \
  > results/run1/collector.log 2>&1 &
pgrep -f 'python3 -u scripts/collect.py' > results/run1/collector.pid
```

Monitor:

```bash
tail -f results/run1/collector.log
```

Stop cleanly:

```bash
kill -TERM "$(cat results/run1/collector.pid)"
```

Run the same command again to resume. Completed measurements recorded in
`state.json` are skipped.

## Results

- `summary.csv`: one row per completed measurement
- `runs/*/measurement_telemetry.csv`: raw power, temperature, SCLK, MCLK and
  utilization samples
- `runs/*/precondition_telemetry.csv`: preconditioning samples
- `metadata.json`: compiler, driver, GPU and configuration
- `state.json`: resumable state
