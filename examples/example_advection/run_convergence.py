#!/usr/bin/env python3
"""Report short-time L2 convergence for the boundary-driven advection example."""

import math
import re
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BINARY = ROOT / "example_advection_cpu.out"
RESOLUTIONS = (4, 8, 12, 16, 24)
THETA = math.pi / 4.0
END_TIME = 0.1
PULSE_DURATION = 0.25


def main():
    if not BINARY.is_file():
        raise SystemExit("Build first: make example_advection EIGEN_INC_DIR=tpl/BoBa/tpl/eigen")

    print("n       L2 error       observed rate")
    previous_n = None
    previous_error = None
    first_error = None
    with tempfile.TemporaryDirectory(prefix="example_advection_") as output_dir:
        for n in RESOLUTIONS:
            run = subprocess.run(
                [str(BINARY), "--resolution", str(n), "--order", "1",
                 "--theta", str(THETA), "--end_time", str(END_TIME),
                 "--pulse_duration", str(PULSE_DURATION)],
                cwd=output_dir,
                capture_output=True,
                text=True,
                check=True,
            )
            match = re.search(r"\bl2_error=([0-9.eE+-]+)", run.stdout)
            if match is None:
                raise RuntimeError(f"No L2 error in output for n={n}:\n{run.stdout}")
            error = float(match.group(1))
            if not math.isfinite(error) or error <= 0:
                raise RuntimeError(f"Invalid L2 error for n={n}: {error}")
            rate = "—" if previous_error is None else (
                f"{math.log(previous_error / error) / math.log(n / previous_n):.3f}"
            )
            print(f"{n:<3}  {error:>12.6e}  {rate:>13}")
            if first_error is None:
                first_error = error
            previous_n, previous_error = n, error

    overall_rate = math.log(first_error / previous_error) / math.log(
        RESOLUTIONS[-1] / RESOLUTIONS[0]
    )
    print(f"Overall observed rate: {overall_rate:.3f}")
    print(f"Smooth pulse duration: {PULSE_DURATION}; final time: {END_TIME}")
    print("For order 1 and smooth data, the asymptotic L2 target is rate 2.")


if __name__ == "__main__":
    main()
