#!/usr/bin/env python3
"""
Runtime comparison A* vs. IBF: plots per noise level (x: trace length, y: runtime).

Each point is one trace variant; its runtime is averaged over all repetitions.
For every noise level, one plot with a linear and one with a logarithmic y-axis
is created.

"""

from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

# ==================================== Settings =====================================
LOG_NAME = "BPI_Challenge_2012"
NOISE_LEVELS = [0.8, 0.2]  # plots are created in this order

# Directory containing the CSVs. Default: <project>/output, assuming this script
# lives in a subdirectory of the project (like the script creating the IBF file).
DATA_DIR = Path(__file__).resolve().parent.parent / "output"
OUT_DIR = DATA_DIR / "plots"  # target directory for the plots
FILE_FORMAT = "png"           # "png", "pdf" or "svg"
SHOW_MEDIAN = False           # additionally draw a median line per trace length
FONT_SIZE = 28                # base font size; labels and ticks scale with it

FIG_SIZE = (9, 5.5)           # figure size in inches
MARKER_SIZE = 8               # area of one point; raise it for larger figures

# Text of the y-axis. With a large font it can become taller than the figure;
# in that case its font size is reduced automatically (see fit_ylabel).
Y_LABEL = "Runtime per variant [s]"
X_TICK_STEP = 100             # unlabelled ticks on the x-axis every ... events
#X_LABEL_STEP = 200             # labelled ticks every ... events; None -> automatic
#X_LABEL_STEP = None             # labelled ticks every ... events; None -> automatic
#X_LABEL_STEP = 5             # labelled ticks every ... events; None -> automatic
X_LABEL_STEP = 50             # labelled ticks every ... events; None -> automatic
                              # (e.g. 50 gives labels at 0, 50, 100, 150 + longest)

# Settings per method. {log} is replaced by LOG_NAME, paths are relative to DATA_DIR.
#   sep                 CSV delimiter
#   noise/variant/time  column names
#   length              column with the trace length; None -> computed from the variant
#   time_to_s           factor to convert the time column to seconds
#   computed_col        only keep rows where this column is filled; None -> keep all
METHODS = {
    "A*": {
        "file": "{log}_all_noise_conformance_variant_runtimes.csv",
        "sep": ",",
        "noise": "noise_threshold",
        "variant": "trace",
        "length": "trace_length",
        "time": "time",               # seconds
        "time_to_s": 1.0,
        "computed_col": None,
    },
    "IBF": {
        "file": "IBF_alignment_{log}_all_noise_all_variants.csv",
        "sep": ";",
        "noise": "noise",
        "variant": "given trace",
        "length": None,
        "time": "calculation time [ms]",  # milliseconds
        "time_to_s": 1e-3,
        # Repeated variants have time 0 and an empty "searched buckets"
        # (result reused) -> keep only rows that were actually computed
        "computed_col": "searched buckets",
    },
}
# ===================================================================================

TRACE_SEPARATOR = " - "  # separator of the events within a trace

# Unified column names after loading
NOISE, LENGTH, TIME, METHOD = "noise", "trace_length", "time_s", "method"

COLORS = {"A*": "#0072B2", "IBF": "#E69F00"}
MIN_VARIANTS = 5  # median line only for trace lengths with at least this many variants

# Padding below 0 on the linear y-axis, as a share of the largest runtime
Y_BOTTOM_MARGIN = 0.03

# y-axis variants: (file suffix, logarithmic?)
Y_SCALES = [("linear", False), ("log", True)]

plt.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.labelsize": FONT_SIZE + 2,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE,
})


def find_files(log: str, data_dir: Path) -> dict:
    """Returns {method: path} for a log and aborts with a hint if something is missing."""
    paths = {m: data_dir / cfg["file"].format(log=log) for m, cfg in METHODS.items()}
    missing = [p for p in paths.values() if not p.is_file()]
    if missing:
        # show files containing the log name, otherwise all CSVs in the directory
        found = sorted(data_dir.rglob(f"*{log}*.csv")) or sorted(data_dir.glob("*.csv"))
        raise SystemExit(
            "Not found:\n" + "".join(f"  {p}\n" for p in missing)
            + f"CSV files found in {data_dir}:\n"
            + ("".join(f"  {p.relative_to(data_dir)}\n" for p in found) or "  (none)\n")
            + "-> Adjust LOG_NAME, DATA_DIR or METHODS at the top of the script."
        )
    return paths


def load(path: Path, method: str) -> pd.DataFrame:
    """
    Reads a result file and returns one row per (noise, variant)
    with the runtime in seconds, averaged over all repetitions.
    """
    cfg = METHODS[method]
    header = pd.read_csv(path, sep=cfg["sep"], nrows=0).columns

    needed = [cfg["noise"], cfg["variant"], cfg["time"], cfg["length"], cfg["computed_col"]]
    needed = [c for c in needed if c is not None]
    missing = [c for c in needed if c not in header]
    if missing:
        raise SystemExit(
            f"{path.name}: column(s) {missing} missing. Available: {list(header)}\n"
            f"-> Check the delimiter (currently {cfg['sep']!r}) and column names in METHODS."
        )

    # Only load the required columns (the files are large due to the trace strings)
    df = pd.read_csv(path, sep=cfg["sep"], usecols=needed)
    rows_total = len(df)

    if cfg["computed_col"] is not None:
        df = df[df[cfg["computed_col"]].notna()]
        print(f"{method}: {len(df)} of {rows_total} rows actually computed "
              f"(remaining rows = reused variants, ignored)")

    out = pd.DataFrame({
        NOISE: df[cfg["noise"]].astype(float),
        "variant": df[cfg["variant"]],
        TIME: df[cfg["time"]].astype(float) * cfg["time_to_s"],
    })
    if cfg["length"] is not None:
        out[LENGTH] = df[cfg["length"]].astype(int)
    else:
        out[LENGTH] = df[cfg["variant"]].str.count(TRACE_SEPARATOR) + 1

    # Average per variant over the repetitions
    out = (out.groupby([NOISE, "variant", LENGTH], as_index=False)[TIME]
              .mean()
              .drop(columns="variant"))
    out[METHOD] = method
    return out


def fit_ylabel(fig, ax) -> None:
    """
    Shrinks the y-axis label until it fits into the height of the figure.

    A rotated label is drawn along the whole height, so with a large font it can
    become taller than the figure itself and its ends are cut off when saving.
    """
    label = ax.yaxis.label

    for _ in range(30):
        renderer = fig.canvas.get_renderer()
        # the label is centred on the axes, so the axes height is the limit
        available = ax.get_window_extent(renderer).height
        if label.get_window_extent(renderer).height <= available:
            break
        label.set_fontsize(label.get_fontsize() * 0.95)
        fig.canvas.draw()


def plot_noise(data: pd.DataFrame, out: Path, log_y: bool, median: bool) -> None:
    """Creates one plot (A* and IBF) for one noise level."""
    if log_y:
        # A runtime of 0 cannot be shown on a logarithmic axis
        zero = data[TIME] <= 0
        if zero.any():
            print(f"Note: {zero.sum()} variants with runtime 0 not shown in the log plot: "
                  f"{data[zero].groupby(METHOD).size().to_dict()}")
            data = data[~zero]

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    groups = {m: g for m, g in data.groupby(METHOD, sort=False)}

    # First all points, then the median lines, so no line is hidden
    for m, g in groups.items():
        ax.scatter(g[LENGTH], g[TIME], s=MARKER_SIZE, alpha=0.3, linewidths=0,
                   color=COLORS[m], rasterized=True)
    handles = [Line2D([], [], color=COLORS[m], marker="o", ls="none", label=m) for m in groups]

    if median:
        for m, g in groups.items():
            stats = g.groupby(LENGTH)[TIME].agg(["median", "size"])
            stats = stats[stats["size"] >= MIN_VARIANTS]
            ax.plot(stats.index, stats["median"], color=COLORS[m], lw=2,
                    path_effects=[pe.Stroke(linewidth=4, foreground="white"), pe.Normal()])
        handles.append(Line2D([], [], color="grey", lw=2,
                              label=f"Median per length (≥ {MIN_VARIANTS} variants)"))

    ax.set_xlabel("Trace length")

    # x-axis from 0, with regular labelled ticks plus one on the longest variant
    longest = int(data[LENGTH].max())
    ax.set_xlim(0, longest * 1.03)
    if X_LABEL_STEP:
        labelled = list(range(0, longest, X_LABEL_STEP))
    else:
        labelled = list(ax.get_xticks())
    # drop ticks that would collide with the label on the longest variant
    labelled = [t for t in labelled if 0 <= t < longest * 0.93]
    ax.set_xticks(sorted(set(labelled + [0, longest])))
    ax.set_xticks([t for t in range(0, longest, X_TICK_STEP)], minor=True)
    ax.tick_params(axis="x", which="minor", length=4)
    if log_y:
        ax.set_yscale("log")
        ax.set_ylabel(Y_LABEL)
    else:
        # small gap below 0 so the points do not sit on the axis line
        top = data[TIME].max()
        ax.set_ylim(bottom=-Y_BOTTOM_MARGIN * top, top=top * (1 + Y_BOTTOM_MARGIN))
        ax.set_ylabel(Y_LABEL)
    ax.grid(alpha=0.3)

    # Legend below the plot so it does not hide any points
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.13),
              ncol=len(handles), frameon=False)
    fig.canvas.draw()
    fit_ylabel(fig, ax)

    # a little padding, so nothing touches the edge of the image
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def main() -> None:
    paths = find_files(LOG_NAME, DATA_DIR)
    data = pd.concat([load(p, m) for m, p in paths.items()], ignore_index=True)

    available = ", ".join(f"{v:g}" for v in sorted(data[NOISE].unique()))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for noise in NOISE_LEVELS:
        sub = data[(data[NOISE] - noise).abs() < 1e-9]  # float comparison with tolerance
        if sub.empty:
            print(f"Noise {noise:g}: not in the data (available: {available}), skipped\n")
            continue

        absent = set(paths) - set(sub[METHOD])
        if absent:
            print(f"Note: no data from {', '.join(sorted(absent))} for noise {noise:g}")

        print(f"Noise {noise:g} (runtime per variant in s):")
        print(sub.groupby(METHOD)[TIME].agg(["count", "median", "mean", "max"]).round(6))

        for suffix, log_y in Y_SCALES:
            out = OUT_DIR / f"{LOG_NAME}_noise_{noise:g}_{suffix}.{FILE_FORMAT}"
            plot_noise(sub, out, log_y, SHOW_MEDIAN)
            print(f"Saved: {out}")
        print()


if __name__ == "__main__":
    main()
