"""One-click, leakage-safe comparison of PC-TCN and forecasting baselines.

The script trains missing neural models on the same realization-level split,
evaluates every model on exactly the same test windows, performs paired tests
at the realization level, and writes paper-ready CSV tables and a figure.

Default use in VS Code: open this file and press Run Python File.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


NEURAL_MODELS = ("pctcn", "tcn", "rnn", "gru", "cnn_rnn")
CLASSICAL_MODELS = ("persistence", "climatology", "ridge_AR156")
COMMON_METRICS = (
    "MAE_W",
    "RMSE_W",
    "R2_NSE",
    "amplitude_ratio",
    "peak_precision",
    "peak_recall",
    "ramp_MAE_W",
)


def run_command(command, cwd):
    print("\n" + "=" * 78)
    print("Running:", " ".join(str(item) for item in command))
    print("=" * 78, flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def ensure_outputs(args, root, comparison_dir):
    training_script = root / "train_pctcn.py"
    for seed in args.seeds:
        for model in args.models:
            run_dir = args.training_output_dir / f"{model}_seed{seed}"
            metrics_path = run_dir / "test_metrics.csv"
            predictions_path = run_dir / "sampled_test_predictions.npz"
            complete = metrics_path.exists() and predictions_path.exists()
            if complete and not args.force:
                print(f"Reusing completed run: {model}, seed={seed}")
                continue
            command = [
                sys.executable,
                str(training_script),
                "--dataset",
                str(args.dataset),
                "--model",
                model,
                "--seed",
                str(seed),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--train-windows-per-realization",
                str(args.train_windows_per_realization),
                "--eval-windows-per-realization",
                str(args.test_windows_per_realization),
                "--control-realizations",
                "0",
                "--output-dir",
                str(args.training_output_dir),
            ]
            run_command(command, root)

    classical_metrics = comparison_dir / "classical_baselines.csv"
    classical_predictions = comparison_dir / "classical_test_predictions.npz"
    if args.force or not (classical_metrics.exists() and classical_predictions.exists()):
        command = [
            sys.executable,
            str(root / "run_classical_baselines.py"),
            "--dataset",
            str(args.dataset),
            "--train-windows-per-realization",
            str(args.train_windows_per_realization),
            "--test-windows-per-realization",
            str(args.test_windows_per_realization),
            "--seed",
            str(args.seeds[0]),
            "--output",
            str(classical_metrics),
            "--predictions-output",
            str(classical_predictions),
        ]
        run_command(command, root)
    else:
        print("Reusing completed classical baselines")
    return classical_metrics, classical_predictions


def load_results(args, classical_metrics_path, classical_predictions_path):
    metric_frames = []
    prediction_records = []

    for seed in args.seeds:
        for model in args.models:
            run_dir = args.training_output_dir / f"{model}_seed{seed}"
            frame = pd.read_csv(run_dir / "test_metrics.csv")
            frame["model"] = model
            frame["seed"] = seed
            frame["family"] = "neural"
            metric_frames.append(frame)

            saved = np.load(run_dir / "sampled_test_predictions.npz")
            prediction_records.append(
                dict(
                    model=model,
                    seed=seed,
                    y_true=saved["y_true_W"].astype(np.float64),
                    prediction=saved["quantiles_W"][..., 1].astype(np.float64),
                    indices=saved["window_indices"].astype(np.int32),
                )
            )

    classical_frame = pd.read_csv(classical_metrics_path)
    classical_frame["seed"] = args.seeds[0]
    classical_frame["family"] = "classical"
    metric_frames.append(classical_frame)

    classical = np.load(classical_predictions_path)
    for model in CLASSICAL_MODELS:
        prediction_records.append(
            dict(
                model=model,
                seed=args.seeds[0],
                y_true=classical["y_true_W"].astype(np.float64),
                prediction=classical[f"{model}_W"].astype(np.float64),
                indices=classical["window_indices"].astype(np.int32),
            )
        )

    metrics = pd.concat(metric_frames, ignore_index=True, sort=False)
    for column in COMMON_METRICS:
        metrics[column] = pd.to_numeric(metrics[column], errors="coerce")
    metrics["RMSE_rank"] = metrics["RMSE_W"].rank(method="min")
    metrics["MAE_rank"] = metrics["MAE_W"].rank(method="min")
    metrics["ramp_rank"] = metrics["ramp_MAE_W"].rank(method="min")
    metrics["mean_common_metric_rank"] = metrics[
        ["RMSE_rank", "MAE_rank", "ramp_rank"]
    ].mean(axis=1)
    return metrics, prediction_records


def verify_pairing(reference, competitor):
    if reference["y_true"].shape != competitor["y_true"].shape:
        raise ValueError("Test prediction shapes differ; paired comparison is invalid")
    if not np.array_equal(reference["indices"], competitor["indices"]):
        raise ValueError("Test window indices differ; paired comparison is invalid")
    if not np.allclose(reference["y_true"], competitor["y_true"], atol=1e-4):
        raise ValueError("Test targets differ; paired comparison is invalid")


def block_statistics(reference, competitor, bootstrap_samples, rng):
    verify_pairing(reference, competitor)
    actual = reference["y_true"]
    pc_prediction = reference["prediction"]
    other_prediction = competitor["prediction"]
    realization_ids = reference["indices"][:, 0]
    unique_ids = np.unique(realization_ids)

    pc_sse, other_sse, counts = [], [], []
    pc_block_mse, other_block_mse = [], []
    for realization in unique_ids:
        mask = realization_ids == realization
        pc_error = actual[mask] - pc_prediction[mask]
        other_error = actual[mask] - other_prediction[mask]
        pc_sum = float(np.sum(pc_error**2))
        other_sum = float(np.sum(other_error**2))
        count = int(pc_error.size)
        pc_sse.append(pc_sum)
        other_sse.append(other_sum)
        counts.append(count)
        pc_block_mse.append(pc_sum / count)
        other_block_mse.append(other_sum / count)

    pc_sse = np.asarray(pc_sse)
    other_sse = np.asarray(other_sse)
    counts = np.asarray(counts)
    pc_block_mse = np.asarray(pc_block_mse)
    other_block_mse = np.asarray(other_block_mse)
    pc_rmse = float(np.sqrt(pc_sse.sum() / counts.sum()))
    other_rmse = float(np.sqrt(other_sse.sum() / counts.sum()))

    sampled = rng.integers(
        0, len(unique_ids), size=(bootstrap_samples, len(unique_ids))
    )
    sampled_counts = counts[sampled].sum(axis=1)
    bootstrap_delta = np.sqrt(pc_sse[sampled].sum(axis=1) / sampled_counts) - np.sqrt(
        other_sse[sampled].sum(axis=1) / sampled_counts
    )
    ci_low, ci_high = np.percentile(bootstrap_delta, [2.5, 97.5])

    try:
        test = wilcoxon(
            pc_block_mse,
            other_block_mse,
            alternative="less",
            zero_method="wilcox",
        )
        p_value = float(test.pvalue)
    except ValueError:
        p_value = 1.0

    return dict(
        pc_RMSE_W=pc_rmse,
        competitor_RMSE_W=other_rmse,
        delta_RMSE_W=pc_rmse - other_rmse,
        relative_RMSE_improvement_pct=100.0 * (other_rmse - pc_rmse) / other_rmse,
        bootstrap_CI_low_W=float(ci_low),
        bootstrap_CI_high_W=float(ci_high),
        wilcoxon_p=p_value,
        test_realizations=len(unique_ids),
    )


def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running_max = 0.0
    total = len(p_values)
    for position, index in enumerate(order):
        candidate = min((total - position) * p_values[index], 1.0)
        running_max = max(running_max, candidate)
        adjusted[index] = running_max
    return adjusted


def paired_comparisons(args, prediction_records):
    rng = np.random.default_rng(args.bootstrap_seed)
    rows = []
    records = {(r["model"], r["seed"]): r for r in prediction_records}
    for seed in args.seeds:
        reference = records[("pctcn", seed)]
        for competitor in args.models:
            if competitor == "pctcn":
                continue
            row = block_statistics(
                reference,
                records[(competitor, seed)],
                args.bootstrap_samples,
                rng,
            )
            row.update(pc_seed=seed, competitor=competitor, competitor_seed=seed)
            rows.append(row)
        for competitor in CLASSICAL_MODELS:
            row = block_statistics(
                reference,
                records[(competitor, args.seeds[0])],
                args.bootstrap_samples,
                rng,
            )
            row.update(
                pc_seed=seed,
                competitor=competitor,
                competitor_seed=args.seeds[0],
            )
            rows.append(row)

    frame = pd.DataFrame(rows)
    frame["holm_adjusted_p"] = holm_adjust(frame["wilcoxon_p"].to_numpy())
    frame["pc_has_lower_RMSE"] = frame["delta_RMSE_W"] < 0.0
    frame["significant_after_Holm_0.05"] = frame["holm_adjusted_p"] < 0.05
    frame["CI_excludes_zero_in_PC_favor"] = frame["bootstrap_CI_high_W"] < 0.0
    return frame


def horizon_metrics(prediction_records):
    rows = []
    for record in prediction_records:
        error = record["y_true"] - record["prediction"]
        for horizon in range(error.shape[1]):
            rows.append(
                dict(
                    model=record["model"],
                    seed=record["seed"],
                    horizon_step=horizon + 1,
                    seconds_ahead=(horizon + 1) / 10.0,
                    MAE_W=float(np.mean(np.abs(error[:, horizon]))),
                    RMSE_W=float(np.sqrt(np.mean(error[:, horizon] ** 2))),
                )
            )
    return pd.DataFrame(rows)


def make_summary(metrics, paired, seeds):
    summary = (
        metrics.groupby("model", as_index=False)
        .agg(
            runs=("RMSE_W", "size"),
            RMSE_W_mean=("RMSE_W", "mean"),
            RMSE_W_SD=("RMSE_W", "std"),
            MAE_W_mean=("MAE_W", "mean"),
            R2_NSE_mean=("R2_NSE", "mean"),
            ramp_MAE_W_mean=("ramp_MAE_W", "mean"),
        )
        .sort_values("RMSE_W_mean")
        .reset_index(drop=True)
    )
    summary.insert(0, "RMSE_rank", np.arange(1, len(summary) + 1))

    pc_row = summary.loc[summary["model"] == "pctcn"].iloc[0]
    pc_pairs = paired[paired["pc_seed"] == seeds[0]]
    all_lower = bool(pc_pairs["pc_has_lower_RMSE"].all())
    all_significant = bool(
        (
            pc_pairs["significant_after_Holm_0.05"]
            & pc_pairs["CI_excludes_zero_in_PC_favor"]
        ).all()
    )
    lines = [
        "PC-TCN model validation summary",
        f"Training seeds completed: {len(seeds)} ({', '.join(map(str, seeds))})",
        f"PC-TCN RMSE rank: {int(pc_row['RMSE_rank'])} of {len(summary)}",
        f"PC-TCN mean RMSE: {pc_row['RMSE_W_mean']:.6f} W",
        f"Lower RMSE than every competitor for seed {seeds[0]}: {all_lower}",
        f"Significantly better than every competitor after Holm correction: {all_significant}",
    ]
    if len(seeds) < 10:
        lines.append(
            "Status: preliminary only; run at least 10 training seeds before a paper-level claim."
        )
    else:
        lines.append(
            "Status: multi-seed run complete; inspect effect sizes and confidence intervals before claiming superiority."
        )
    return summary, "\n".join(lines) + "\n"


def save_figure(summary, output_path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is unavailable; skipped comparison figure")
        return

    plot = summary.sort_values("RMSE_W_mean")
    colors = ["tab:blue" if name == "pctcn" else "0.65" for name in plot["model"]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(plot["model"], plot["RMSE_W_mean"], color=colors)
    axes[0].set_ylabel("Test RMSE (W), lower is better")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(plot["model"], plot["R2_NSE_mean"], color=colors)
    axes[1].set_ylabel("Test R2/NSE, higher is better")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].axhline(0.0, color="black", lw=0.8)
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("Leakage-safe model comparison on identical test realizations")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=root / "data" / "spectrum_realizations_demo.npz",
    )
    parser.add_argument("--models", nargs="+", choices=NEURAL_MODELS, default=list(NEURAL_MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-windows-per-realization", type=int, default=1500)
    parser.add_argument("--test-windows-per-realization", type=int, default=600)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20250905)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--training-output-dir", type=Path, default=root / "outputs")
    parser.add_argument(
        "--comparison-output-dir",
        type=Path,
        default=root / "outputs" / "model_validation",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent
    if "pctcn" not in args.models:
        raise ValueError("pctcn must be included because it is the reference model")
    args.dataset = args.dataset.resolve()
    args.training_output_dir = args.training_output_dir.resolve()
    comparison_dir = args.comparison_output_dir.resolve()
    comparison_dir.mkdir(parents=True, exist_ok=True)

    classical_metrics, classical_predictions = ensure_outputs(
        args, root, comparison_dir
    )
    metrics, predictions = load_results(
        args, classical_metrics, classical_predictions
    )
    paired = paired_comparisons(args, predictions)
    horizons = horizon_metrics(predictions)
    summary, summary_text = make_summary(metrics, paired, args.seeds)

    metrics.sort_values("RMSE_W").to_csv(
        comparison_dir / "all_run_metrics.csv", index=False, encoding="utf-8-sig"
    )
    summary.to_csv(
        comparison_dir / "model_summary.csv", index=False, encoding="utf-8-sig"
    )
    paired.to_csv(
        comparison_dir / "pctcn_paired_tests.csv", index=False, encoding="utf-8-sig"
    )
    horizons.to_csv(
        comparison_dir / "metrics_by_horizon.csv", index=False, encoding="utf-8-sig"
    )
    (comparison_dir / "validation_summary.txt").write_text(
        summary_text, encoding="utf-8"
    )
    save_figure(summary, comparison_dir / "model_comparison.png")

    print("\nFINAL MODEL RANKING (primary metric: test RMSE)")
    print(summary.to_string(index=False))
    print("\n" + summary_text)
    print(f"Saved validation outputs to {comparison_dir}")


if __name__ == "__main__":
    main()
