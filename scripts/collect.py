#!/usr/bin/env python3
"""Precondition, run five CU workloads, and collect read-only GPU telemetry."""

from __future__ import annotations

import argparse
import atexit
import csv
import hashlib
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BENCHMARKS = {
    "SCALAR ADD",
    "SFU TRANSCENDENTAL",
    "MATRIX MFMA FP32",
    "VECTOR FP64 FMA",
    "VECTOR FP32 FMA",
}
SUMMARY_FIELDS = [
    "run_id",
    "status",
    "benchmark",
    "envelope",
    "blocks",
    "threads",
    "repetition",
    "clock_mode",
    "actual_sclk_mhz_mean",
    "actual_sclk_mhz_min",
    "actual_sclk_mhz_max",
    "power_w_mean",
    "power_w_median",
    "power_w_stddev",
    "power_w_min",
    "power_w_max",
    "temperature_c_mean",
    "temperature_c_min",
    "temperature_c_max",
    "gpu_use_pct_mean",
    "mclk_mhz_mean",
    "samples",
    "precondition_benchmark",
    "precondition_duration_s",
    "target_warmup_s",
    "target_duration_s",
    "transition_gap_s",
    "run_dir",
    "error",
]


class Cancelled(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.example.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_runs is not None and args.max_runs < 1:
        parser.error("--max-runs must be positive")
    return args


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text())
    required = {
        "device",
        "binary",
        "clock_mode",
        "sclk_reference_values_mhz",
        "repetitions",
        "samples_per_second",
        "cool_to_c",
        "cool_timeout_s",
        "precondition",
        "measurement",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError("config is missing: " + ", ".join(sorted(missing)))
    if config["clock_mode"] != "observe":
        raise ValueError("this package supports only clock_mode='observe'")
    reference_values = config["sclk_reference_values_mhz"]
    if (
        len(reference_values) != 10
        or len(set(reference_values)) != 10
        or reference_values != sorted(reference_values)
        or any(float(value) <= 0 for value in reference_values)
    ):
        raise ValueError("sclk_reference_values_mhz must contain 10 increasing values")
    if int(config["repetitions"]) <= 0:
        raise ValueError("repetitions must be positive")
    benchmarks = config["measurement"].get("benchmarks", [])
    names = {str(item.get("name", "")) for item in benchmarks}
    if len(benchmarks) != 5 or names != EXPECTED_BENCHMARKS:
        raise ValueError("measurement.benchmarks must contain exactly the five packaged workloads")
    if config["precondition"].get("benchmark") not in EXPECTED_BENCHMARKS:
        raise ValueError("precondition benchmark must be one of the five packaged workloads")
    for item in [config["precondition"], *benchmarks]:
        if int(item.get("blocks", 0)) <= 0 or not 0 < int(item.get("threads", 0)) <= 1024:
            raise ValueError(f"invalid geometry for {item.get('name', item.get('benchmark'))}")
    for value in (
        config["samples_per_second"],
        config["cool_timeout_s"],
        config["precondition"]["duration_s"],
        config["measurement"]["duration_s"],
    ):
        if float(value) <= 0:
            raise ValueError("sampling, timeout, and durations must be positive")
    if float(config["measurement"]["warmup_s"]) < 0:
        raise ValueError("measurement.warmup_s cannot be negative")
    return config


def binary_path(config_path: Path, config: dict[str, Any]) -> Path:
    path = Path(str(config["binary"])).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def make_schedule(config: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    benchmarks = config["measurement"]["benchmarks"]
    for repetition in range(1, int(config["repetitions"]) + 1):
        offset = (repetition - 1) % len(benchmarks)
        ordered_benchmarks = benchmarks[offset:] + benchmarks[:offset]
        for benchmark in ordered_benchmarks:
            result.append(
                {
                    "run_id": f"r{repetition:02d}_observed_{slug(benchmark['name'])}",
                    "benchmark": benchmark["name"],
                    "envelope": benchmark["envelope"],
                    "blocks": int(benchmark["blocks"]),
                    "threads": int(benchmark["threads"]),
                    "repetition": repetition,
                    "clock_mode": "observe",
                }
            )
    return result


def run_command(command: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, check=check)


def device_sysfs(device: int) -> tuple[Path, Path, str]:
    listing = run_command(["rocm-smi", "--showbus"])
    match = re.search(
        rf"GPU\[{device}\].*?PCI Bus:\s*([0-9a-fA-F:.]+)",
        listing.stdout,
    )
    if match is None:
        raise RuntimeError(f"could not find GPU {device} in rocm-smi --showbus")
    bdf = match.group(1).lower()
    for card in Path("/sys/class/drm").glob("card[0-9]*"):
        uevent = card / "device/uevent"
        if not uevent.exists() or f"pci_slot_name={bdf}" not in uevent.read_text().lower():
            continue
        hwmons = list((card / "device/hwmon").glob("hwmon*"))
        if hwmons:
            return card / "device", hwmons[0], bdf
    raise RuntimeError(f"could not map GPU {device} ({bdf}) to amdgpu sysfs")


def read_number(path: Path, divisor: float = 1.0) -> float | None:
    try:
        return float(path.read_text().strip()) / divisor
    except (OSError, ValueError):
        return None


def telemetry(device_path: Path, hwmon: Path, elapsed_s: float) -> dict[str, float | None]:
    return {
        "timestamp_s": elapsed_s,
        "gpu_use_pct": read_number(device_path / "gpu_busy_percent"),
        "temperature_c": read_number(hwmon / "temp2_input", 1000.0),
        "power_w": read_number(hwmon / "power1_input", 1_000_000.0),
        "sclk_mhz": read_number(hwmon / "freq1_input", 1_000_000.0),
        "mclk_mhz": read_number(hwmon / "freq2_input", 1_000_000.0),
    }


def wait_idle(device_path: Path, timeout_s: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_s
    consecutive = 0
    while time.monotonic() < deadline:
        busy = read_number(device_path / "gpu_busy_percent")
        consecutive = consecutive + 1 if busy is not None and busy <= 1 else 0
        if consecutive >= 3:
            return
        time.sleep(0.2)
    raise TimeoutError("GPU did not become idle; do not run on a shared/busy GPU")


def wait_cool(hwmon: Path, maximum_c: float, timeout_s: float, cancelled: threading.Event) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if cancelled.is_set():
            raise Cancelled("cancelled during cooldown")
        temperature = read_number(hwmon / "temp2_input", 1000.0)
        if temperature is not None and temperature <= maximum_c:
            return
        time.sleep(2.0)
    raise TimeoutError(f"GPU did not cool to {maximum_c:g} C within {timeout_s:g} s")


def stop_process(process: subprocess.Popen[str] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def sample_for(
    duration_s: float,
    samples_per_second: float,
    device_path: Path,
    hwmon: Path,
    cancelled: threading.Event,
    process: subprocess.Popen[str],
) -> list[dict[str, float | None]]:
    count = max(1, int(round(duration_s * samples_per_second)))
    interval_s = 1.0 / samples_per_second
    start = time.monotonic()
    rows = []
    for index in range(count):
        delay = start + index * interval_s - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        if cancelled.is_set():
            raise Cancelled("measurement cancelled")
        if process.poll() is not None:
            raise RuntimeError(f"benchmark exited early with {process.returncode}")
        rows.append(telemetry(device_path, hwmon, time.monotonic() - start))
    return rows


def run_window(
    binary: Path,
    name: str,
    blocks: int,
    threads: int,
    warmup_s: float,
    measurement_s: float,
    samples_per_second: float,
    environment: dict[str, str],
    device_path: Path,
    hwmon: Path,
    cancelled: threading.Event,
    active: dict[str, subprocess.Popen[str] | None],
) -> tuple[list[dict[str, float | None]], str, str, list[str], float]:
    command = [
        str(binary),
        "--exact-filter",
        name,
        "--blocks",
        str(blocks),
        "--threads",
        str(threads),
        "--duration-s",
        str(warmup_s + measurement_s + 2.0),
    ]
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    active["process"] = process
    assert process.stdout is not None
    ready = process.stdout.readline()
    if ready.strip() != f"BENCHMARK_READY {name}":
        stop_process(process)
        stdout, stderr = process.communicate()
        active["process"] = None
        raise RuntimeError(f"benchmark did not become ready: {ready!r}; {stderr.strip()}")
    ready_time = time.monotonic()
    stdout_parts = [ready]
    stderr_parts: list[str] = []
    assert process.stderr is not None
    stdout_thread = threading.Thread(
        target=lambda: stdout_parts.append(process.stdout.read()), daemon=True
    )
    stderr_thread = threading.Thread(
        target=lambda: stderr_parts.append(process.stderr.read()), daemon=True
    )
    stdout_thread.start()
    stderr_thread.start()
    try:
        if warmup_s:
            sample_for(
                warmup_s,
                min(samples_per_second, 2.0),
                device_path,
                hwmon,
                cancelled,
                process,
            )
        rows = sample_for(
            measurement_s,
            samples_per_second,
            device_path,
            hwmon,
            cancelled,
            process,
        )
    finally:
        stop_process(process)
        active["process"] = None
        stdout_thread.join(timeout=5)
        stderr_thread.join(timeout=5)
    return rows, "".join(stdout_parts), "".join(stderr_parts), command, ready_time


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no data for {path}")
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def numbers(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if row.get(key) is not None]


def mean(rows: list[dict[str, Any]], key: str) -> float | str:
    values = numbers(rows, key)
    return statistics.fmean(values) if values else ""


def summarize(
    point: dict[str, Any],
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    transition_gap_s: float,
    run_dir: Path,
) -> dict[str, Any]:
    powers = numbers(rows, "power_w")
    clocks = numbers(rows, "sclk_mhz")
    temperatures = numbers(rows, "temperature_c")
    if not powers or not clocks or not temperatures:
        raise RuntimeError("power, SCLK, or temperature telemetry is unavailable")
    actual_sclk = statistics.fmean(clocks)
    row = {field: "" for field in SUMMARY_FIELDS}
    row.update(point)
    row.update(
        {
            "status": "ok",
            "actual_sclk_mhz_mean": actual_sclk,
            "actual_sclk_mhz_min": min(clocks),
            "actual_sclk_mhz_max": max(clocks),
            "power_w_mean": statistics.fmean(powers),
            "power_w_median": statistics.median(powers),
            "power_w_stddev": statistics.pstdev(powers),
            "power_w_min": min(powers),
            "power_w_max": max(powers),
            "temperature_c_mean": statistics.fmean(temperatures),
            "temperature_c_min": min(temperatures),
            "temperature_c_max": max(temperatures),
            "gpu_use_pct_mean": mean(rows, "gpu_use_pct"),
            "mclk_mhz_mean": mean(rows, "mclk_mhz"),
            "samples": len(rows),
            "precondition_benchmark": config["precondition"]["benchmark"],
            "precondition_duration_s": config["precondition"]["duration_s"],
            "target_warmup_s": config["measurement"]["warmup_s"],
            "target_duration_s": config["measurement"]["duration_s"],
            "transition_gap_s": transition_gap_s,
            "run_dir": str(run_dir),
            "error": "",
        }
    )
    return row


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def write_summary(path: Path, completed: dict[str, dict[str, Any]]) -> None:
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for run_id in sorted(completed):
            writer.writerow(completed[run_id])
    os.replace(temporary, path)


def version_output(command: list[str]) -> str:
    try:
        result = run_command(command, check=False)
        return (result.stdout + result.stderr).strip()
    except FileNotFoundError:
        return "not found"


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    schedule = make_schedule(config)
    if args.dry_run:
        print(json.dumps(schedule, indent=2))
        print(f"{len(schedule)} scheduled measurements")
        return 0

    binary = binary_path(config_path, config)
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise FileNotFoundError(f"build the benchmark first: {binary}")
    available = set(run_command([str(binary), "--list"]).stdout.splitlines())
    if available != EXPECTED_BENCHMARKS:
        raise RuntimeError(f"unexpected workload set in binary: {sorted(available)}")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    runs_dir = output / "runs"
    runs_dir.mkdir(exist_ok=True)
    state_path = output / "state.json"
    summary_path = output / "summary.csv"
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    completed: dict[str, dict[str, Any]] = {}
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get("config_sha256") != config_hash:
            raise RuntimeError("config changed; use the original config or a new output directory")
        completed = dict(state.get("runs", {}))

    device = int(config["device"])
    clock_mode = "observe"
    device_path, hwmon, bdf = device_sysfs(device)
    environment = os.environ.copy()
    environment["ROCR_VISIBLE_DEVICES"] = str(device)
    environment["HIP_VISIBLE_DEVICES"] = str(device)
    cancelled = threading.Event()
    active: dict[str, subprocess.Popen[str] | None] = {"process": None}

    def cleanup() -> None:
        stop_process(active["process"])

    atexit.register(cleanup)

    def request_stop(signum: int, _frame: Any) -> None:
        cancelled.set()
        stop_process(active["process"])
        print(f"received signal {signum}; saving state and resetting clocks", file=sys.stderr)

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    metadata = {
        "status": "running",
        "config": config,
        "config_path": str(config_path),
        "config_sha256": config_hash,
        "binary": str(binary),
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "device": device,
        "clock_mode": clock_mode,
        "bdf": bdf,
        "hwmon": str(hwmon),
        "hipcc_version": version_output(["hipcc", "--version"]),
        "rocm_smi_version": version_output(["rocm-smi", "--version"]),
        "driver_version": version_output(["rocm-smi", "--showdriverversion"]),
        "scheduled_measurements": len(schedule),
        "completed_ok": sum(row.get("status") == "ok" for row in completed.values()),
    }
    atomic_json(output / "metadata.json", metadata)

    unfinished = [
        point
        for point in schedule
        if completed.get(point["run_id"], {}).get("status") != "ok"
    ]
    if args.max_runs is not None:
        unfinished = unfinished[: args.max_runs]

    for index, point in enumerate(unfinished, 1):
        if cancelled.is_set():
            break
        run_id = point["run_id"]
        run_dir = runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[{index}/{len(unfinished)}] {point['benchmark']} with observed SCLK",
            flush=True,
        )
        row = {field: "" for field in SUMMARY_FIELDS}
        row.update(point)
        row.update({"status": "failed", "run_dir": str(run_dir), "error": ""})
        try:
            wait_idle(device_path)
            wait_cool(
                hwmon,
                float(config["cool_to_c"]),
                float(config["cool_timeout_s"]),
                cancelled,
            )
            precondition = config["precondition"]
            pre_rows, pre_stdout, pre_stderr, pre_command, _ = run_window(
                binary,
                str(precondition["benchmark"]),
                int(precondition["blocks"]),
                int(precondition["threads"]),
                0.0,
                float(precondition["duration_s"]),
                min(float(config["samples_per_second"]), 2.0),
                environment,
                device_path,
                hwmon,
                cancelled,
                active,
            )
            precondition_end = time.monotonic()
            write_csv(run_dir / "precondition_telemetry.csv", pre_rows)
            (run_dir / "precondition.stdout.log").write_text(pre_stdout)
            (run_dir / "precondition.stderr.log").write_text(pre_stderr)
            atomic_json(run_dir / "precondition_command.json", pre_command)

            rows, stdout, stderr, command, target_ready = run_window(
                binary,
                str(point["benchmark"]),
                int(point["blocks"]),
                int(point["threads"]),
                float(config["measurement"]["warmup_s"]),
                float(config["measurement"]["duration_s"]),
                float(config["samples_per_second"]),
                environment,
                device_path,
                hwmon,
                cancelled,
                active,
            )
            transition_gap_s = target_ready - precondition_end
            write_csv(run_dir / "measurement_telemetry.csv", rows)
            (run_dir / "measurement.stdout.log").write_text(stdout)
            (run_dir / "measurement.stderr.log").write_text(stderr)
            atomic_json(run_dir / "measurement_command.json", command)
            row = summarize(point, rows, config, transition_gap_s, run_dir)
        except Cancelled as error:
            row["status"] = "cancelled"
            row["error"] = str(error)
        except Exception as error:
            row["error"] = f"{type(error).__name__}: {error}"
            (run_dir / "error.log").write_text(row["error"] + "\n")
            print(f"FAILED: {row['error']}", file=sys.stderr, flush=True)

        completed[run_id] = row
        atomic_json(
            state_path,
            {
                "status": "cancelled" if cancelled.is_set() else "running",
                "config_sha256": config_hash,
                "runs": completed,
            },
        )
        write_summary(summary_path, completed)
        metadata["completed_ok"] = sum(
            item.get("status") == "ok" for item in completed.values()
        )
        metadata["failed_or_invalid"] = sum(
            item.get("status") not in ("ok", "cancelled") for item in completed.values()
        )
        metadata["last_run_id"] = run_id
        atomic_json(output / "metadata.json", metadata)

    successful = {
        point["run_id"]
        for point in schedule
        if completed.get(point["run_id"], {}).get("status") == "ok"
    }
    status = (
        "completed"
        if len(successful) == len(schedule)
        else ("cancelled" if cancelled.is_set() else "partial")
    )
    atomic_json(
        state_path,
        {"status": status, "config_sha256": config_hash, "runs": completed},
    )
    metadata["status"] = status
    metadata["completed_ok"] = len(successful)
    atomic_json(output / "metadata.json", metadata)
    print(f"state={status} summary={summary_path}")
    if cancelled.is_set():
        return 130
    return 0 if status == "completed" or args.max_runs is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
