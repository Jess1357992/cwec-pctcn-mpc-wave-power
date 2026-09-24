"""Persistence, climatology, and causal Ridge/AR baselines on the same splits."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


N_IN = 156
HORIZON = 16


def starts_for_length(length, count, randomize, rng):
    possible = length - N_IN - HORIZON + 1
    count = min(count, possible)
    if randomize and count < possible:
        return rng.choice(possible, size=count, replace=False)
    return np.linspace(0, possible - 1, count, dtype=int)


def collect_windows(
    powers,
    contexts,
    realizations,
    per_realization,
    seed,
    randomize,
    return_indices=False,
):
    rng = np.random.default_rng(seed)
    histories, targets, indices = [], [], []
    for realization in realizations:
        rated = float(contexts[realization, 7])
        series = powers[realization] / rated
        starts = starts_for_length(len(series), per_realization, randomize, rng)
        histories.extend(series[s : s + N_IN] for s in starts)
        targets.extend(series[s + N_IN : s + N_IN + HORIZON] for s in starts)
        indices.extend((int(realization), int(s)) for s in starts)
    result = (
        np.asarray(histories, dtype=np.float64),
        np.asarray(targets, dtype=np.float64),
    )
    if return_indices:
        return (*result, np.asarray(indices, dtype=np.int32))
    return result


def fit_ridge_incremental(powers, contexts, realizations, per_realization, seed, alpha):
    rng = np.random.default_rng(seed)
    xtx = np.zeros((N_IN + 1, N_IN + 1), dtype=np.float64)
    xty = np.zeros((N_IN + 1, HORIZON), dtype=np.float64)
    for realization in realizations:
        rated = float(contexts[realization, 7])
        series = powers[realization] / rated
        starts = starts_for_length(len(series), per_realization, True, rng)
        for offset in range(0, len(starts), 2048):
            chunk = starts[offset : offset + 2048]
            x = np.stack([series[s : s + N_IN] for s in chunk])
            x = np.c_[x, np.ones(len(x))]
            y = np.stack([series[s + N_IN : s + N_IN + HORIZON] for s in chunk])
            xtx += x.T @ x
            xty += x.T @ y
    penalty = np.eye(N_IN + 1) * alpha
    penalty[-1, -1] = 0.0
    return np.linalg.solve(xtx + penalty, xty)


def summarize(name, actual_w, predicted_w, peak_threshold_w):
    error = actual_w - predicted_w
    denominator = np.sum((actual_w - actual_w.mean()) ** 2)
    actual_peak = actual_w >= peak_threshold_w
    predicted_peak = predicted_w >= peak_threshold_w
    tp = np.sum(actual_peak & predicted_peak)
    return dict(
        model=name,
        MAE_W=float(np.mean(np.abs(error))),
        RMSE_W=float(np.sqrt(np.mean(error**2))),
        R2_NSE=float(1.0 - np.sum(error**2) / denominator),
        amplitude_ratio=float(predicted_w.std() / max(actual_w.std(), 1e-12)),
        peak_precision=float(tp / max(np.sum(predicted_peak), 1)),
        peak_recall=float(tp / max(np.sum(actual_peak), 1)),
        ramp_MAE_W=float(
            np.mean(
                np.abs(
                    np.diff(actual_w, axis=1)
                    - np.diff(predicted_w, axis=1)
                )
            )
        ),
    )


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=root / "data" / "spectrum_realizations_demo.npz",
    )
    parser.add_argument("--train-windows-per-realization", type=int, default=1500)
    parser.add_argument("--test-windows-per-realization", type=int, default=600)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "outputs" / "classical_baselines.csv",
    )
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=root / "outputs" / "classical_test_predictions.npz",
    )
    args = parser.parse_args()

    data = np.load(args.dataset, allow_pickle=False)
    powers = data["powers"].astype(np.float64)
    contexts = data["contexts"].astype(np.float64)
    splits = data["splits"]
    train_realizations = np.flatnonzero(splits == 0)
    test_realizations = np.flatnonzero(splits == 2)

    ridge_weights = fit_ridge_incremental(
        powers,
        contexts,
        train_realizations,
        args.train_windows_per_realization,
        args.seed,
        args.ridge_alpha,
    )
    x_test, y_test, test_indices = collect_windows(
        powers,
        contexts,
        test_realizations,
        args.test_windows_per_realization,
        args.seed + 1,
        False,
        return_indices=True,
    )
    # Dataset currently has one rated power, but retain an explicit scale so the
    # reported numbers remain in watts.
    rated_w = float(np.median(contexts[test_realizations, 7]))
    actual_w = y_test * rated_w
    persistence_w = np.repeat(x_test[:, -1:], HORIZON, axis=1) * rated_w
    train_mean_scaled = float(
        np.mean(
            [
                powers[r].mean() / contexts[r, 7]
                for r in train_realizations
            ]
        )
    )
    climatology_w = np.full_like(actual_w, train_mean_scaled * rated_w)
    ridge_w = np.c_[x_test, np.ones(len(x_test))] @ ridge_weights * rated_w
    ridge_w = np.clip(ridge_w, 0.0, rated_w)
    peak_threshold_w = float(np.percentile(powers[train_realizations], 90))

    rows = [
        summarize("persistence", actual_w, persistence_w, peak_threshold_w),
        summarize("climatology", actual_w, climatology_w, peak_threshold_w),
        summarize("ridge_AR156", actual_w, ridge_w, peak_threshold_w),
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False, encoding="utf-8-sig")
    args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.predictions_output,
        y_true_W=actual_w.astype(np.float32),
        persistence_W=persistence_w.astype(np.float32),
        climatology_W=climatology_w.astype(np.float32),
        ridge_AR156_W=ridge_w.astype(np.float32),
        window_indices=test_indices,
    )
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
