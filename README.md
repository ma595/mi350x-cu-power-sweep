# MI300X CU power sweep — ROCm 6.1

## Requirements

- Ubuntu 22.04
- ROCm 6.1.0: HIP 6.1.40091 and AMD Clang 17.0.0
- GCC 11, CMake >= 3.21 , Python 3

Install ROCm 6.1.0 using AMD's instructions:

https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.1.0/tutorial/quick-start.html

Check the installed version:

```bash
/opt/rocm/bin/hipcc --version
```

## Workloads

 `SCALAR ADD` 
 `SFU TRANSCENDENTAL` 
 `MATRIX MFMA FP32` 
 `VECTOR FP64 FMA`
 `VECTOR FP32 FMA` 

All workloads default to 304 blocks and 1024 threads per block.

## Build

```bash
cd mi300x-cu-power-sweep
export ROCM_PATH=/opt/rocm
export PATH="$ROCM_PATH/bin:$PATH"
./build.sh
```

The output binary is `build/cu_power_bench`.

## Configure

Edit `config.example.json`. Set:

- `device`

The config contains 10 evenly spaced SCLK reference values covering 500–2100
MHz. They are reference values for observed samples, not requested clock locks.

Check the GPU and available clocks:

```bash
./scripts/check_environment.sh
```

`clock_mode` is `observe`: the script never writes clocks. Each workload run
samples the SCLK values selected by the pod.

## Test

Validate the schedule without changing clocks:

```bash
python3 scripts/collect.py \
  --config config.example.json \
  --output-dir results/mi300x-cu \
  --dry-run
```

Run one measurement:

```bash
python3 -u scripts/collect.py \
  --config config.example.json \
  --output-dir results/mi300x-cu \
  --max-runs 1
```

## Run

```bash
mkdir -p results/mi300x-cu
nohup python3 -u scripts/collect.py \
  --config config.example.json \
  --output-dir results/mi300x-cu \
  > results/mi300x-cu/collector.log 2>&1 &
echo $! > results/mi300x-cu/collector.pid
```

Monitor:

```bash
tail -f results/mi300x-cu/collector.log
```

Stop cleanly:

```bash
kill -TERM "$(cat results/mi300x-cu/collector.pid)"
```

Run the same command again to resume. Completed measurements recorded in
`state.json` are skipped.

## Results

- `results/mi300x-cu/summary.csv`: one row per completed measurement
- `results/mi300x-cu/runs/*/measurement_telemetry.csv`: raw power, temperature,
  SCLK, MCLK, and utilization samples
- `results/mi300x-cu/runs/*/precondition_telemetry.csv`: preconditioning samples
- `results/mi300x-cu/metadata.json`: compiler, driver, GPU, and configuration
- `results/mi300x-cu/state.json`: resumable state
