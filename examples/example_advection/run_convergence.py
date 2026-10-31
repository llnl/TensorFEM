#!/usr/bin/env python3
"""Plot L2 convergence and solver runtime for both advection schemes."""

import math
import os
import re
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BINARY = ROOT / "example_advection_cpu.out"
OUTPUT_DIR = Path(__file__).resolve().parent
RESOLUTIONS = (4, 8, 12, 16, 24)
THETA = math.pi / 4.0
END_TIME = 0.1
PULSE_DURATION = 0.25


def metric(output, name, resolution):
    match = re.search(rf"\b{name}=([0-9.eE+-]+)", output)
    if match is None:
        raise RuntimeError(f"Missing {name} for n={resolution}:\n{output}")
    value = float(match.group(1))
    if not math.isfinite(value) or value <= 0.0:
        raise RuntimeError(f"Invalid {name} for n={resolution}: {value}")
    return value


def observed_rate(errors):
    return math.log(errors[0] / errors[-1]) / math.log(
        RESOLUTIONS[-1] / RESOLUTIONS[0]
    )


def main():
    if not BINARY.is_file():
        raise SystemExit("Build first: make example_advection EIGEN_INC_DIR=tpl/BoBa/tpl/eigen")

    # Keep Matplotlib's cache writable on headless systems.
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "tensorfem_matplotlib")
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter

    results = {
        "TT": {"error": [], "runtime": [], "color": "blue"},
        "Full": {"error": [], "runtime": [], "color": "black"},
    }
    compression_times = []
    print("n     TT L2 error    Full L2 error    TT step (s)   Full step (s)   TT compression (s)")
    with tempfile.TemporaryDirectory(prefix="example_advection_") as output_dir:
        for n in RESOLUTIONS:
            run = subprocess.run(
                [str(BINARY), "--scheme", "0", "--resolution", str(n),
                 "--order", "1", "--theta", str(THETA),
                 "--end_time", str(END_TIME),
                 "--pulse_duration", str(PULSE_DURATION)],
                cwd=output_dir,
                capture_output=True,
                text=True,
                check=True,
            )
            for method, prefix in (("TT", "tt"), ("Full", "full")):
                results[method]["error"].append(
                    metric(run.stdout, f"{prefix}_l2_error", n)
                )
                results[method]["runtime"].append(
                    metric(run.stdout, f"{prefix}_runtime_seconds", n)
                )
            compression_times.append(metric(run.stdout, "tt_compression_seconds", n))
            print(
                f"{n:<3}  {results['TT']['error'][-1]:>12.6e}  "
                f"{results['Full']['error'][-1]:>14.6e}  "
                f"{results['TT']['runtime'][-1]:>12.6e}  "
                f"{results['Full']['runtime'][-1]:>13.6e}  "
                f"{compression_times[-1]:>18.6e}"
            )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), sharex=True)
    for method in ("Full", "TT"):
        data = results[method]
        rate = observed_rate(data["error"])
        ax1.loglog(
            RESOLUTIONS, data["error"], color=data["color"],
            marker="x" if method == "TT" else "o",
            linestyle="--" if method == "TT" else "-",
            label=f"{method} (rate {rate:.2f})",
        )
        print(f"{method} overall observed L2 rate: {rate:.3f}")
    reference = [
        results["Full"]["error"][0] * (RESOLUTIONS[0] / n) ** 2
        for n in RESOLUTIONS
    ]
    ax1.loglog(RESOLUTIONS, reference, color="gray", linestyle=":",
               label="Second order reference")
    ax1.set(title="L2 convergence", xlabel="Elements per direction",
            ylabel="L2 error at T=0.1")
    ax1.set_xticks(RESOLUTIONS, labels=[str(n) for n in RESOLUTIONS])
    ax1.xaxis.set_minor_formatter(NullFormatter())
    ax1.grid(True, which="both", alpha=0.25)
    ax1.legend()
    for method in ("Full", "TT"):
        data = results[method]
        ax2.loglog(
            RESOLUTIONS, data["runtime"], color=data["color"],
            marker="x" if method == "TT" else "o",
            linestyle="--" if method == "TT" else "-", label=method,
        )
    ax2.set(title="Time stepping runtime", xlabel="Elements per direction",
            ylabel="Time stepping (seconds)")
    ax2.set_xticks(RESOLUTIONS, labels=[str(n) for n in RESOLUTIONS])
    ax2.xaxis.set_minor_formatter(NullFormatter())
    ax2.grid(True, which="both", alpha=0.25)
    ax2.legend()
    fig.suptitle("Advection convergence and runtime")
    fig.text(0.5, 0.01, "Runtime excludes compression, shared assembly, and verification.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    plot_path = OUTPUT_DIR / "convergence_comparison.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    print("Runtime includes time stepping only; TT compression is reported separately.")
    print(f"Saved {plot_path}")


if __name__ == "__main__":
    main()
