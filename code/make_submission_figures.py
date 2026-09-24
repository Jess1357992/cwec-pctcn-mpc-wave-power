"""Generate the four core submission items directly from verified outputs.

No experimental value is typed into this script. All table entries, curves,
confidence intervals, and p-values are read from the saved CSV results.

VS Code use: open this file and press "Run Python File".
"""
from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"
VALIDATION_DIR = OUTPUTS_DIR / "model_validation"
MPC_DIR = OUTPUTS_DIR / "mpc"
FIGURE_DIR = OUTPUTS_DIR / "submission_figures"

MANIFEST_PATH = DATA_DIR / "manifest_demo.csv"
MODEL_METRICS_PATH = VALIDATION_DIR / "all_run_metrics.csv"
PAIRED_TESTS_PATH = VALIDATION_DIR / "pctcn_paired_tests.csv"
MPC_METRICS_PATH = MPC_DIR / "mpc_metrics.csv"
TRACE_PATHS = {
    "zoh": MPC_DIR / "trace_zoh.csv",
    "forecast": MPC_DIR / "trace_forecast.csv",
    "perfect": MPC_DIR / "trace_perfect.csv",
}
MPC_SCRIPT_PATH = ROOT / "run_hard_mpc.py"

SPLIT_ORDER = ["train", "validation", "test"]
EXPECTED_MODELS = {
    "pctcn",
    "tcn",
    "rnn",
    "gru",
    "cnn_rnn",
    "ridge_AR156",
    "climatology",
    "persistence",
}
EXPECTED_CONTROLLERS = {"no_storage", "zoh", "forecast", "perfect"}
MODEL_LABELS = {
    "pctcn": "PC-TCN",
    "tcn": "TCN",
    "rnn": "RNN",
    "gru": "GRU",
    "cnn_rnn": "CNN-RNN",
    "ridge_AR156": "Ridge/AR(156)",
    "climatology": "Climatology",
    "persistence": "Persistence",
}
CONTROLLER_LABELS = {
    "no_storage": "No storage",
    "zoh": "ZOH-MPC",
    "forecast": "PC-TCN-MPC",
    "perfect": "Perfect-MPC",
}
COLORS = {
    "pctcn": "#0072B2",
    "ridge_AR156": "#E69F00",
    "zoh": "#D55E00",
    "forecast": "#0072B2",
    "perfect": "#009E73",
    "no_storage": "#777777",
}


def require_files(paths):
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required result files:\n" + "\n".join(missing))


def configure_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_figure(fig, stem):
    png_path = FIGURE_DIR / f"{stem}.png"
    pdf_path = FIGURE_DIR / f"{stem}.pdf"
    fig.savefig(png_path, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path, pdf_path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_provenance(source_paths):
    rows = []
    for path in source_paths:
        stat = path.stat()
        rows.append(
            {
                "source_file": str(path.resolve()),
                "sha256": sha256(path),
                "size_bytes": stat.st_size,
                "modified_local": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "figure_generation_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
    pd.DataFrame(rows).to_csv(
        FIGURE_DIR / "source_provenance_sha256.csv",
        index=False,
        encoding="utf-8-sig",
    )


def validate_and_summarize_dataset(manifest):
    required = {
        "split",
        "realization_id",
        "samples",
        "seed",
        "Hs_m",
        "Te_s",
        "omega0_over_omegap",
        "zeta",
    }
    if not required.issubset(manifest.columns):
        raise ValueError(f"Manifest lacks columns: {sorted(required - set(manifest.columns))}")
    if set(manifest["split"]) != set(SPLIT_ORDER):
        raise ValueError("Manifest must contain train, validation, and test splits")

    scenario_columns = ["Hs_m", "Te_s", "omega0_over_omegap", "zeta"]
    overlap = manifest.groupby(scenario_columns)["split"].nunique()
    if int(overlap.max()) != 1:
        raise ValueError("A physical scenario appears in more than one data split")

    scenarios = manifest.drop_duplicates(scenario_columns)
    rows = []
    for split in SPLIT_ORDER:
        split_rows = manifest[manifest["split"] == split]
        split_scenarios = scenarios[scenarios["split"] == split]
        phase_counts = split_rows.groupby(scenario_columns).size()
        rows.append(
            {
                "Split": split.capitalize(),
                "Physical scenarios": len(split_scenarios),
                "Realizations": len(split_rows),
                "Phases per scenario": (
                    str(int(phase_counts.min()))
                    if phase_counts.min() == phase_counts.max()
                    else f"{int(phase_counts.min())}-{int(phase_counts.max())}"
                ),
                "Samples per realization": int(split_rows["samples"].iloc[0]),
                "Duration per realization (s)": int(split_rows["samples"].iloc[0] / 10),
            }
        )
    summary = pd.DataFrame(rows)
    total = {
        "Split": "Total",
        "Physical scenarios": int(summary["Physical scenarios"].sum()),
        "Realizations": int(summary["Realizations"].sum()),
        "Phases per scenario": "4",
        "Samples per realization": int(manifest["samples"].iloc[0]),
        "Duration per realization (s)": int(manifest["samples"].iloc[0] / 10),
    }
    return pd.concat([summary, pd.DataFrame([total])], ignore_index=True)


def make_dataset_table(dataset_summary):
    dataset_summary.to_csv(
        FIGURE_DIR / "01_dataset_split_table.csv",
        index=False,
        encoding="utf-8-sig",
    )
    fig, ax = plt.subplots(figsize=(11.2, 3.1))
    ax.axis("off")
    cell_text = dataset_summary.astype(str).values.tolist()
    table = ax.table(
        cellText=cell_text,
        colLabels=dataset_summary.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.13, 0.17, 0.14, 0.17, 0.20, 0.22],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.55)
    for (row, column), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#D9EAF7")
            cell.set_text_props(weight="bold")
        elif row == len(dataset_summary):
            cell.set_facecolor("#EEEEEE")
            cell.set_text_props(weight="bold")
        cell.set_edgecolor("#666666")
        cell.set_linewidth(0.6)
    ax.set_title(
        "Dataset partition by mutually exclusive physical scenarios",
        fontweight="bold",
        pad=12,
    )
    fig.text(
        0.5,
        0.025,
        "Scenario = (Hs, Te, omega0/omegap, zeta). All random-phase realizations "
        "of a scenario remain in one split.",
        ha="center",
        fontsize=8.5,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    return save_figure(fig, "01_dataset_split_table")


def validate_model_metrics(metrics):
    if set(metrics["model"]) != EXPECTED_MODELS:
        missing = EXPECTED_MODELS - set(metrics["model"])
        extra = set(metrics["model"]) - EXPECTED_MODELS
        raise ValueError(f"Model set mismatch. Missing={missing}, extra={extra}")
    if metrics["model"].duplicated().any():
        raise ValueError("Expected one completed run per model in the minimum draft")
    required_numeric = [
        "MAE_W",
        "RMSE_W",
        "R2_NSE",
        "amplitude_ratio",
        "peak_precision",
        "peak_recall",
        "ramp_MAE_W",
    ]
    if metrics[required_numeric].isna().any().any():
        raise ValueError("Missing values found in common model-comparison metrics")


def make_forecast_comparison(metrics):
    columns = [
        "model",
        "MAE_W",
        "RMSE_W",
        "R2_NSE",
        "amplitude_ratio",
        "peak_precision",
        "peak_recall",
        "ramp_MAE_W",
        "P10_P90_coverage",
        "mean_interval_width_W",
        "params",
    ]
    table = metrics[columns].sort_values("RMSE_W").copy()
    table.insert(0, "RMSE_rank", np.arange(1, len(table) + 1))
    table["model"] = table["model"].map(MODEL_LABELS)
    table.to_csv(
        FIGURE_DIR / "02_forecast_model_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plot = metrics.sort_values("RMSE_W", ascending=True).copy()
    labels = plot["model"].map(MODEL_LABELS).tolist()
    colors = [COLORS.get(name, "#999999") for name in plot["model"]]
    y = np.arange(len(plot))
    panels = [
        ("RMSE_W", "RMSE (W) - lower is better", "{:.2f}"),
        ("MAE_W", "MAE (W) - lower is better", "{:.2f}"),
        ("R2_NSE", "R2/NSE - higher is better", "{:.3f}"),
        ("peak_recall", "Peak recall - higher is better", "{:.3f}"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2))
    for ax, (column, xlabel, formatter) in zip(axes.ravel(), panels):
        values = plot[column].to_numpy(dtype=float)
        ax.barh(y, values, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        ax.axvline(0, color="black", lw=0.7)
        ax.grid(axis="x", alpha=0.22)
        span = max(values.max() - min(values.min(), 0), 1e-6)
        for index, value in enumerate(values):
            offset = 0.012 * span
            ax.text(
                value + offset,
                index,
                formatter.format(value),
                va="center",
                ha="left",
                fontsize=7.7,
            )
    fig.suptitle(
        "Forecasting performance on identical unseen test windows (seed 42)",
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.012,
        "Blue: proposed PC-TCN; orange: strongest classical RMSE baseline. "
        "All results use the same 24 test realizations and 14,400 windows.",
        ha="center",
        fontsize=8.5,
    )
    fig.tight_layout(rect=[0, 0.035, 1, 0.97])
    return save_figure(fig, "02_forecast_model_comparison")


def validate_paired_tests(paired):
    expected_competitors = EXPECTED_MODELS - {"pctcn"}
    if set(paired["competitor"]) != expected_competitors:
        raise ValueError("Paired-test competitor set is incomplete")
    if (paired["bootstrap_CI_low_W"] > paired["bootstrap_CI_high_W"]).any():
        raise ValueError("Invalid bootstrap confidence interval ordering")


def make_paired_forest(paired):
    table = paired.copy()
    table["competitor"] = table["competitor"].map(MODEL_LABELS)
    table.to_csv(
        FIGURE_DIR / "03_pctcn_paired_statistics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plot = paired.sort_values("relative_RMSE_improvement_pct", ascending=True).copy()
    # Existing delta is PC-TCN minus competitor. Flip the sign so positive
    # values and intervals indicate an improvement in favour of PC-TCN.
    improvement = -plot["delta_RMSE_W"].to_numpy(dtype=float)
    ci_low = -plot["bootstrap_CI_high_W"].to_numpy(dtype=float)
    ci_high = -plot["bootstrap_CI_low_W"].to_numpy(dtype=float)
    xerr = np.vstack([improvement - ci_low, ci_high - improvement])
    significant = plot["significant_after_Holm_0.05"].astype(bool).to_numpy()
    colors = np.where(significant, "#0072B2", "#E69F00")
    y = np.arange(len(plot))

    fig, ax = plt.subplots(figsize=(10.2, 5.4))
    for index in range(len(plot)):
        ax.errorbar(
            improvement[index],
            y[index],
            xerr=xerr[:, index : index + 1],
            fmt="o",
            color=colors[index],
            ecolor=colors[index],
            capsize=4,
            markersize=6,
            lw=1.5,
        )
    ax.axvline(0.0, color="black", ls="--", lw=1.0)
    ax.set_yticks(y, plot["competitor"].map(MODEL_LABELS))
    ax.set_xlabel("RMSE reduction by PC-TCN (W); positive values favour PC-TCN")
    ax.set_title(
        "Paired realization-block bootstrap estimates with 95% confidence intervals",
        fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.25)
    x_span = max(ci_high.max() - min(ci_low.min(), 0), 1.0)
    for index, (_, row) in enumerate(plot.iterrows()):
        ax.text(
            ci_high[index] + 0.02 * x_span,
            index,
            f"Holm p={row['holm_adjusted_p']:.3g}",
            va="center",
            fontsize=8,
        )
    ax.scatter([], [], color="#0072B2", label="Significant after Holm correction")
    ax.scatter([], [], color="#E69F00", label="Not significant")
    ax.legend(loc="lower right", frameon=False)
    fig.text(
        0.5,
        0.015,
        "Inference is across 24 held-out random-phase realizations for one training seed. "
        "The Ridge/AR interval crosses zero.",
        ha="center",
        fontsize=8.5,
    )
    fig.tight_layout(rect=[0, 0.045, 1, 1])
    return save_figure(fig, "03_pctcn_paired_statistical_comparison")


def validate_mpc(mpc_metrics, traces):
    if set(mpc_metrics["controller"]) != EXPECTED_CONTROLLERS:
        raise ValueError("MPC metrics do not contain the expected four controllers")
    if mpc_metrics["realization_id"].nunique() != 1:
        raise ValueError("The minimum submission figure expects one representative case")
    controlled = mpc_metrics[mpc_metrics["controller"] != "no_storage"]
    if int(controlled["solver_failure_count"].sum()) != 0:
        raise ValueError("MPC solver failures are present; refusing to generate submission figure")
    if int(controlled["hard_SOC_violation_count"].sum()) != 0:
        raise ValueError("Hard SOC violations are present; refusing to generate submission figure")
    if int(controlled["grid_import_count"].sum()) != 0:
        raise ValueError("Grid import events are present; refusing to generate submission figure")
    if float(controlled["energy_balance_error_J"].abs().max()) > 1e-6:
        raise ValueError("Energy-balance error exceeds 1e-6 J")

    lengths = {name: len(frame) for name, frame in traces.items()}
    if set(lengths.values()) != {600}:
        raise ValueError(f"Expected 600-step (60 s) traces, found {lengths}")
    reference_time = traces["forecast"]["time_s"].to_numpy()
    for name, frame in traces.items():
        if not np.allclose(frame["time_s"].to_numpy(), reference_time):
            raise ValueError(f"Time axis mismatch in {name} trace")


def make_mpc_case_figure(mpc_metrics, traces):
    table_columns = [
        "controller",
        "mean_grid_W",
        "sigma_grid_W",
        "sigma_reduction_pct",
        "ramp_max_Wps",
        "ramp_mean_Wps",
        "LPSP_pct",
        "curtailment_pct",
        "SOC_min",
        "SOC_max",
        "terminal_delta_SOC",
        "safety_filter_count",
        "solver_failure_count",
        "hard_SOC_violation_count",
        "energy_balance_error_J",
    ]
    table = mpc_metrics[table_columns].copy()
    table["controller"] = table["controller"].map(CONTROLLER_LABELS)
    table.to_csv(
        FIGURE_DIR / "04_mpc_case_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    time = traces["forecast"]["time_s"].to_numpy(dtype=float)
    raw = traces["forecast"]["wave_power_W"].to_numpy(dtype=float)
    # Use a deterministic zoom (the first 20 s), rather than selecting a
    # visually favourable interval. All metrics and SOC traces remain 60 s.
    power_zoom_s = 20.0
    power_mask = time < (time.min() + power_zoom_s)
    power_time = time[power_mask]
    fig = plt.figure(figsize=(12.0, 9.2))
    grid = fig.add_gridspec(3, 1, height_ratios=[1.45, 1.0, 1.0], hspace=0.35)
    ax_power = fig.add_subplot(grid[0])
    ax_soc = fig.add_subplot(grid[1])
    ax_bar = fig.add_subplot(grid[2])

    ax_power.plot(
        power_time,
        raw[power_mask],
        color="#A6A6A6",
        lw=0.75,
        alpha=0.38,
        label="Raw wave power",
        zorder=1,
    )
    for name in ("zoh", "forecast"):
        ax_power.plot(
            power_time,
            traces[name]["grid_power_W"].to_numpy(dtype=float)[power_mask],
            color=COLORS[name],
            lw=0.9 if name == "zoh" else 1.55,
            alpha=0.78 if name == "zoh" else 1.0,
            label=CONTROLLER_LABELS[name],
            zorder=2 if name == "zoh" else 3,
        )
    ax_power.axhline(12.0, color="black", ls="--", lw=0.8, label="12 W reference")
    ax_power.set_ylabel("Power (W)")
    ax_power.set_xlabel("Time (s; first 20 s shown)")
    ax_power.set_xlim(power_time.min(), power_time.max())
    ax_power.grid(alpha=0.20)
    ax_power.legend(
        ncol=1,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.005, 1.0),
        borderaxespad=0.0,
    )
    ax_power.text(
        0.01,
        0.95,
        "Fixed first-20-s zoom; 60-s metrics below",
        transform=ax_power.transAxes,
        va="top",
        fontsize=8,
        color="#555555",
    )

    for name in ("zoh", "forecast", "perfect"):
        ax_soc.plot(
            time,
            traces[name]["SOC"],
            color=COLORS[name],
            lw=1.2,
            label=CONTROLLER_LABELS[name],
        )
    ax_soc.axhline(0.50, color="black", ls="--", lw=0.8, label="Initial SOC")
    all_soc = np.concatenate([traces[name]["SOC"].to_numpy() for name in TRACE_PATHS])
    margin = max(0.006, 0.12 * (all_soc.max() - all_soc.min()))
    ax_soc.set_ylim(all_soc.min() - margin, all_soc.max() + margin)
    ax_soc.set_ylabel("Flywheel SOC")
    ax_soc.set_xlim(time.min(), time.max())
    ax_soc.set_xlabel("Time (s; full 60 s case)")
    ax_soc.grid(alpha=0.20)
    ax_soc.text(
        0.01,
        0.06,
        "Hard bounds: 0.10-0.90",
        transform=ax_soc.transAxes,
        fontsize=8,
    )

    controlled = (
        mpc_metrics[mpc_metrics["controller"].isin(["zoh", "forecast", "perfect"])]
        .set_index("controller")
        .loc[["zoh", "forecast", "perfect"]]
    )
    metric_columns = ["sigma_reduction_pct", "LPSP_pct", "curtailment_pct"]
    metric_labels = ["Sigma reduction", "LPSP", "Curtailment"]
    x = np.arange(len(metric_columns))
    width = 0.24
    for offset, name in enumerate(("zoh", "forecast", "perfect")):
        values = controlled.loc[name, metric_columns].to_numpy(dtype=float)
        bars = ax_bar.bar(
            x + (offset - 1) * width,
            values,
            width,
            label=CONTROLLER_LABELS[name],
            color=COLORS[name],
        )
        ax_bar.bar_label(bars, fmt="%.1f", fontsize=7.5, padding=2)
    ax_bar.set_xticks(x, metric_labels)
    ax_bar.set_ylabel("Percentage (%)")
    ax_bar.grid(axis="y", alpha=0.20)
    ax_bar.legend(ncol=3, frameon=False)

    realization = str(mpc_metrics["realization_id"].iloc[0])
    match = re.fullmatch(
        r"Hs(?P<hs>[0-9.]+)_Te(?P<te>[0-9.]+)_r(?P<ratio>[0-9.]+)_seed(?P<seed>[0-9]+)",
        realization,
    )
    if match:
        case_title = (
            f"Hs={match.group('hs')} m, Te={match.group('te')} s, "
            f"omega0/omegap={match.group('ratio')}, seed={match.group('seed')}"
        )
    else:
        case_title = realization
    fig.suptitle(
        f"Representative unseen 60 s MPC case: {case_title}",
        fontweight="bold",
        y=0.995,
    )
    forecast_safety_count = int(controlled.loc["forecast", "safety_filter_count"])
    fig.text(
        0.5,
        0.008,
        "PC-TCN-MPC is a coordinated FESS-and-curtailment controller with a "
        f"measurement-based safety layer ({forecast_safety_count} interventions "
        f"in {len(time)} steps).",
        ha="center",
        fontsize=8.5,
    )
    fig.subplots_adjust(bottom=0.055, top=0.955)
    return save_figure(fig, "04_mpc_representative_case")


def write_captions(dataset_summary, metrics, paired, mpc_metrics):
    pc = metrics.set_index("model").loc["pctcn"]
    forecast = mpc_metrics.set_index("controller").loc["forecast"]
    ridge_test = paired.set_index("competitor").loc["ridge_AR156"]
    captions = f"""01_dataset_split_table
Dataset partition used for leakage-safe model development. A physical scenario is
defined by (Hs, Te, omega0/omegap, zeta), and all random-phase realizations from a
scenario are assigned exclusively to train, validation, or test. The dataset contains
{int(dataset_summary.iloc[-1]['Physical scenarios'])} scenarios and
{int(dataset_summary.iloc[-1]['Realizations'])} realizations.

02_forecast_model_comparison
Point-forecast performance on 14,400 windows from 24 unseen test realizations for
training seed 42. PC-TCN obtained RMSE={pc['RMSE_W']:.3f} W, MAE={pc['MAE_W']:.3f} W,
and R2/NSE={pc['R2_NSE']:.3f}. Ordered interval outputs are reported in the CSV but
are not claimed to be calibrated probability intervals.

03_pctcn_paired_statistical_comparison
RMSE reduction of PC-TCN relative to each comparator. Points show paired differences
and bars show 95% realization-block bootstrap confidence intervals. Wilcoxon p-values
were adjusted by Holm's method. The PC-TCN versus Ridge/AR difference was not
significant (Holm p={ridge_test['holm_adjusted_p']:.3f}). Inference covers one model-training seed.

04_mpc_representative_case
Representative 60 s case from an unseen test realization. PC-TCN-assisted MPC reduced
grid-power standard deviation by {forecast['sigma_reduction_pct']:.2f}% while maintaining
zero solver failures, grid-import events, and hard SOC violations. The controller used
coordinated FESS action and {forecast['curtailment_pct']:.2f}% curtailment; its measurement-based
safety layer intervened {int(forecast['safety_filter_count'])} times. This is an illustrative
case and is not a cross-sea-state aggregate result.
"""
    (FIGURE_DIR / "figure_captions.txt").write_text(captions, encoding="utf-8")


def main():
    source_paths = [
        MANIFEST_PATH,
        MODEL_METRICS_PATH,
        PAIRED_TESTS_PATH,
        MPC_METRICS_PATH,
        MPC_SCRIPT_PATH,
        *TRACE_PATHS.values(),
    ]
    require_files(source_paths)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    configure_style()

    manifest = pd.read_csv(MANIFEST_PATH)
    metrics = pd.read_csv(MODEL_METRICS_PATH)
    paired = pd.read_csv(PAIRED_TESTS_PATH)
    mpc_metrics = pd.read_csv(MPC_METRICS_PATH)
    traces = {name: pd.read_csv(path) for name, path in TRACE_PATHS.items()}

    dataset_summary = validate_and_summarize_dataset(manifest)
    validate_model_metrics(metrics)
    validate_paired_tests(paired)
    validate_mpc(mpc_metrics, traces)

    generated = []
    generated.extend(make_dataset_table(dataset_summary))
    generated.extend(make_forecast_comparison(metrics))
    generated.extend(make_paired_forest(paired))
    generated.extend(make_mpc_case_figure(mpc_metrics, traces))
    write_captions(dataset_summary, metrics, paired, mpc_metrics)
    write_provenance(source_paths)

    print("All source checks passed.")
    print("Generated submission items:")
    for path in generated:
        print(f"  {path}")
    print(f"Tables, captions, and provenance: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
