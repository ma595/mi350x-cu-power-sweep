# hwn-z2-gpu02

8 x AMD Instinct MI350X (gfx950). Values measured 2026-09-07.

## Toolchain

| | |
|---|---|
| OS | Ubuntu 26.04 LTS |
| Kernel | 7.0.0-31-generic |
| ROCm release | 10.0.0 |
| HIP | 7.15.26333 |
| AMD clang | 23.0.0git |
| amdgpu-dkms | 1:7.1.3.31500000 (driver 31.50.0) |
| CMake | 4.2.3 |
| GCC | 15.2.0 |
| Python | 3.14.4 |

Ubuntu 26.04 with kernel 7.0 GA is the pairing AMD validates for MI350X on
ROCm 10.

ROCm is installed from `stable.repo.amd.com` as `amdrocm-*10.0` packages with
`-gfx950` arch variants, under `/opt/rocm/core-10.0` and selected through
`/etc/alternatives`. The ROCm 6.x package names (`rocm-hip-sdk`, `rocm-libs`,
`rocm-dev`, `miopen-hip`) do not exist in this scheme — MIOpen is
`amdrocm-dnn*`, and `rocprof` is `rocprofv3`. Checking for the old names
reports "not installed" against a complete stack.

`/opt/rocm/bin` is not on the default `PATH` and `/opt/rocm/lib` is not in the
loader cache, so `hipcc` and `rocm-smi` need `PATH` exported per shell. The
benchmark binary itself carries an RPATH and does not.

## Device mapping

`rocm-smi` sorts by PCI address; KFD does not. `device` in the config indexes
the KFD column.

| `device` (KFD) | PCI address | `rocm-smi` index | idle junction °C |
|---|---|---|---|
| 0 | `0000:75:00.0` | 3 | 62 |
| 1 | `0000:05:00.0` | 0 | 57 |
| 2 | `0000:65:00.0` | 2 | 57 |
| 3 | `0000:15:00.0` | 1 | 62 |
| 4 | `0000:f5:00.0` | 7 | 64 |
| 5 | `0000:85:00.0` | 4 | 58 |
| 6 | `0000:e5:00.0` | 6 | 58 |
| 7 | `0000:95:00.0` | 5 | 61 |

Only indices 2 and 6 coincide between the two orderings.

## Thermal and power

- Idle junction temperature: **57–64 °C** across the eight cards, fully idle.
- `cool_to_c` is set to **67**, clearing the hottest idle floor by 3 °C. The
  shipped default of 50 is unreachable here: every measurement would wait the
  full `cool_timeout_s` and then fail.
- Power cap: 1000 W per card.
- Idle board power: ~265 W.

## Clocks

```
pp_dpm_sclk:  0: 500Mhz   1: 2200Mhz
mclk:         2000 MHz
```

The config's SCLK reference values originally stopped at 2100 MHz — MI300X's
peak engine clock — so the top of this part's range was never requested. They
have been re-spaced evenly across 500–2200 MHz.

Note that `collect.py` never reads that list; it only validates it. The
frequencies are consumed by `scripts/sweep_frequencies.sh`, which sets each one
before invoking the collector.

## Access

Users need the `render` group for `/dev/kfd`; without it `rocminfo` fails with
a permission error and no GPU work can run. Group membership here is
directory-backed, so a local `usermod` may drift — add it centrally.

## SCLK sweep

Device 0, 256 blocks x 1024 threads, 5 workloads x 3 repetitions per frequency
(15 measurements each, 150 total, all successful). Clocks set externally by
`scripts/sweep_frequencies.sh`; `clock_mode` remains `observe`.

| requested MHz | achieved MHz | deviation | power W | n |
|---|---|---|---|---|
| 500 | 496 | -0.7% | 278–337 | 15 |
| 689 | 673 | -2.3% | 285–360 | 15 |
| 878 | 856 | -2.5% | 288–383 | 15 |
| 1067 | 1035 | -3.0% | 293–411 | 15 |
| 1256 | 1214 | -3.3% | 300–439 | 15 |
| 1444 | 1381 | -4.4% | 307–469 | 15 |
| 1633 | 1554 | -4.8% | 323–513 | 15 |
| 1822 | 1807 | -0.8% | 341–628 | 15 |
| 2011 | 1808 | -10.1% | 341–627 | 15 |
| 2200 | 2194 | -0.3% | 372–789 | 15 |

**Requested SCLK does not map linearly onto achieved.** The undershoot widens
with frequency, from −0.7% at 500 MHz to −4.8% at 1633 MHz, then 1822 and 2011
MHz both resolve to the same hardware clock point near 1807 MHz. Ten requested
frequencies therefore yield **nine distinct measured clocks**, with a ~390 MHz
gap between 1807 and 2194 MHz. Quote the achieved column, not the requested one.

The 1822/2011 collapse is reproducible across independent runs, including with
the clock state reset immediately beforehand, so it is hardware behaviour rather
than an artefact of the sweep script.

`-L sclk max` cannot be set to 500 MHz on this part — it equals the DPM minimum
and returns `AMDSMI_STATUS_NOT_SUPPORTED`. The bottom of the range is set with
`amd-smi set -d` (performance determinism) instead, which also tracks the
request more closely than capping does. Note that `amd-smi` exits 0 even when it
refuses a request, so a rejected cap yields a complete dataset at the wrong
frequency with no error.

## Observed power envelopes

Device 0 (`0000:75:00.0`), 256 blocks x 1024 threads, `clock_mode: observe`,
3 repetitions, mean of 300 samples over 30 s each.

| workload | config `envelope` | power W | SCLK MHz | temp °C |
|---|---|---|---|---|
| SCALAR ADD | very-low | 370.9 | 2199 | 66.7 |
| VECTOR FP64 FMA | high | 599.9 | 2191 | 69.6 |
| VECTOR FP32 FMA | very-high | 688.3 | 2192 | 71.1 |
| MATRIX MFMA FP32 | medium | 755.2 | 2194 | 71.6 |
| SFU TRANSCENDENTAL | low | 786.7 | 2195 | 72.4 |

Repeatability is tight: spread across the three repetitions is under 1 % for
every workload (largest is SFU at 0.6 %).

**The `envelope` labels in the config do not match measured ordering on this
part.** Sorted by measured power the order is very-low, high, very-high,
medium, low - only `very-low` lands where its label predicts. The labels carry
MI300X/CDNA3 expectations; CDNA4 evidently distributes power differently across
these units. Treat the labels as inherited annotation, not as a result.

Of note, `VECTOR FP32 FMA` ("very-high") draws less than both MFMA FP32 and the
transcendental workload. The kernel emits plain `v_fma_f32` rather than packed
`v_pk_fma_f32`, so it does not exercise CDNA4's dual-issue FP32 path and is
unlikely to represent a true FP32 power ceiling.

All eight cards remain to be swept; only device 0 has been measured.
