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

The config's SCLK reference values stop at 2100, so the top bin never fills.
Harmless under `clock_mode: observe`, but worth widening to 2200 for this node.

## Access

Users need the `render` group for `/dev/kfd`; without it `rocminfo` fails with
a permission error and no GPU work can run. Group membership here is
directory-backed, so a local `usermod` may drift — add it centrally.

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
