# Provenance

The five HIP microkernels were extracted and minimally adapted from the local
GPUsage checkout at commit:

```text
393ba1b8a9db5ea70a17c6efa0ce124bcf79365b
```

Original source location:

```text
GPUsage/amd/CDNA-benchmarks/benchmarks/power/main.hip
```

Only these workloads are included:

- `SCALAR ADD`
- `SFU TRANSCENDENTAL`
- `MATRIX MFMA FP32`
- `VECTOR FP64 FMA`
- `VECTOR FP32 FMA`

The standalone runner, clock-control collector, configuration, and documentation
were added for this archive.
