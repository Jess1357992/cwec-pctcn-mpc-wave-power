"""Train and evaluate PC-TCN using realization-level train/val/test splits."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from pctcn_model import ControlAwareQuantileLoss, build_model, build_recurrent_baseline


N_IN = 156
HORIZON = 16


def make_indices(powers, realization_indices, max_windows, seed, randomize):
    rng = np.random.default_rng(seed)
    pairs = []
    possible = powers.shape[1] - N_IN - HORIZON + 1
    if possible <= 0:
        raise ValueError("Each realization must be longer than N_IN + HORIZON")
    for realization in realization_indices:
        count = min(max_windows, possible)
        if randomize and count < possible:
            starts = rng.choice(possible, size=count, replace=False)
        else:
            starts = np.linspace(0, possible - 1, count, dtype=int)
        pairs.extend((int(realization), int(start)) for start in starts)
    pairs = np.asarray(pairs, dtype=np.int32)
    if randomize:
        rng.shuffle(pairs)
    return pairs


class WindowSequence(keras.utils.Sequence):
    def __init__(
        self,
        powers,
        contexts_scaled,
        contexts_raw,
        indices,
        batch_size,
        shuffle,
        seed,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.powers = powers
        self.contexts_scaled = contexts_scaled
        self.contexts_raw = contexts_raw
        self.indices = indices.copy()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, batch_index):
        pairs = self.indices[
            batch_index * self.batch_size : (batch_index + 1) * self.batch_size
        ]
        histories, contexts, targets = [], [], []
        for realization, start in pairs:
            rated = float(self.contexts_raw[realization, 7])
            series = self.powers[realization] / rated
            histories.append(series[start : start + N_IN, None])
            targets.append(series[start + N_IN : start + N_IN + HORIZON])
            contexts.append(self.contexts_scaled[realization])
        return (
            (
                np.asarray(histories, dtype=np.float32),
                np.asarray(contexts, dtype=np.float32),
            ),
            np.asarray(targets, dtype=np.float32),
        )

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.indices)

    def all_targets_and_ratings(self):
        targets, ratings = [], []
        for realization, start in self.indices:
            rated = float(self.contexts_raw[realization, 7])
            targets.append(
                self.powers[realization, start + N_IN : start + N_IN + HORIZON]
            )
            ratings.append(rated)
        return np.asarray(targets), np.asarray(ratings)


def metric_summary(y_true_w, quantiles_w, peak_threshold_w):
    median = quantiles_w[..., 1]
    residual = y_true_w - median
    mse = float(np.mean(residual**2))
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(mse))
    denominator = float(np.sum((y_true_w - y_true_w.mean()) ** 2))
    r2 = 1.0 - float(np.sum(residual**2)) / denominator
    amplitude_ratio = float(median.std() / max(y_true_w.std(), 1e-12))
    coverage = float(
        np.mean((y_true_w >= quantiles_w[..., 0]) & (y_true_w <= quantiles_w[..., 2]))
    )
    interval_width = float(np.mean(quantiles_w[..., 2] - quantiles_w[..., 0]))

    actual_peak = y_true_w >= peak_threshold_w
    predicted_peak = median >= peak_threshold_w
    true_positive = int(np.sum(actual_peak & predicted_peak))
    precision = true_positive / max(int(np.sum(predicted_peak)), 1)
    recall = true_positive / max(int(np.sum(actual_peak)), 1)
    if y_true_w.shape[1] > 1:
        ramp_mae = float(
            np.mean(
                np.abs(
                    np.diff(y_true_w, axis=1)
                    - np.diff(median, axis=1)
                )
            )
        )
    else:
        ramp_mae = float("nan")
    return dict(
        MAE_W=mae,
        RMSE_W=rmse,
        R2_NSE=r2,
        amplitude_ratio=amplitude_ratio,
        peak_precision=precision,
        peak_recall=recall,
        ramp_MAE_W=ramp_mae,
        P10_P90_coverage=coverage,
        mean_interval_width_W=interval_width,
    )


def export_control_forecasts(
    model,
    powers,
    contexts_raw,
    contexts_scaled,
    splits,
    realization_ids,
    output_path,
    max_realizations,
    batch_size,
):
    selected = np.flatnonzero(splits == 2)[:max_realizations]
    all_quantiles = []
    all_powers = []
    all_contexts = []
    selected_ids = []
    possible = powers.shape[1] - N_IN - HORIZON + 1
    starts = np.arange(possible, dtype=np.int32)

    for realization in selected:
        rated = float(contexts_raw[realization, 7])
        series = powers[realization] / rated
        predictions = []
        for offset in range(0, possible, batch_size):
            chunk = starts[offset : offset + batch_size]
            histories = np.stack([series[s : s + N_IN] for s in chunk])[..., None]
            context_batch = np.repeat(
                contexts_scaled[realization][None, :], len(chunk), axis=0
            )
            pred = model.predict(
                (histories, context_batch),
                verbose=0,
            )
            predictions.append(pred * rated)
        all_quantiles.append(np.concatenate(predictions, axis=0).astype(np.float32))
        all_powers.append(powers[realization].astype(np.float32))
        all_contexts.append(contexts_raw[realization].astype(np.float32))
        selected_ids.append(realization_ids[realization])

    np.savez_compressed(
        output_path,
        powers=np.stack(all_powers),
        quantiles=np.stack(all_quantiles),
        contexts=np.stack(all_contexts),
        realization_ids=np.asarray(selected_ids),
        forecast_origins=starts + N_IN - 1,
        n_in=np.asarray(N_IN),
        horizon=np.asarray(HORIZON),
        fs_hz=np.asarray(10.0),
    )


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=root / "data" / "spectrum_realizations_demo.npz",
    )
    parser.add_argument(
        "--model",
        choices=["pctcn", "tcn", "rnn", "gru", "cnn_rnn"],
        default="pctcn",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-windows-per-realization", type=int, default=1500)
    parser.add_argument("--eval-windows-per-realization", type=int, default=600)
    parser.add_argument("--control-realizations", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=root / "outputs")
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(args.seed)
    data = np.load(args.dataset, allow_pickle=False)
    powers = data["powers"].astype(np.float32)
    contexts_raw = data["contexts"].astype(np.float32)
    splits = data["splits"]
    realization_ids = data["realization_ids"]
    metadata = json.loads(str(data["metadata_json"]))

    train_realizations = np.flatnonzero(splits == 0)
    val_realizations = np.flatnonzero(splits == 1)
    test_realizations = np.flatnonzero(splits == 2)
    if not (len(train_realizations) and len(val_realizations) and len(test_realizations)):
        raise ValueError("Dataset must contain train, validation, and test realizations")

    context_mean = contexts_raw[train_realizations].mean(axis=0)
    context_std = contexts_raw[train_realizations].std(axis=0)
    context_std[context_std < 1e-8] = 1.0
    contexts_scaled = (contexts_raw - context_mean) / context_std

    train_indices = make_indices(
        powers,
        train_realizations,
        args.train_windows_per_realization,
        args.seed,
        True,
    )
    val_indices = make_indices(
        powers,
        val_realizations,
        args.eval_windows_per_realization,
        args.seed + 1,
        False,
    )
    test_indices = make_indices(
        powers,
        test_realizations,
        args.eval_windows_per_realization,
        args.seed + 2,
        False,
    )

    train_seq = WindowSequence(
        powers,
        contexts_scaled,
        contexts_raw,
        train_indices,
        args.batch_size,
        True,
        args.seed,
    )
    val_seq = WindowSequence(
        powers,
        contexts_scaled,
        contexts_raw,
        val_indices,
        args.batch_size,
        False,
        args.seed + 1,
    )
    test_seq = WindowSequence(
        powers,
        contexts_scaled,
        contexts_raw,
        test_indices,
        args.batch_size,
        False,
        args.seed + 2,
    )

    train_ratings = contexts_raw[train_realizations, 7][:, None]
    peak_threshold_w = float(np.percentile(powers[train_realizations], 90))
    peak_threshold_scaled = float(
        np.percentile(powers[train_realizations] / train_ratings, 90)
    )

    if args.model in {"pctcn", "tcn"}:
        model = build_model(
            n_in=N_IN,
            horizon=HORIZON,
            n_context=contexts_raw.shape[1],
            use_physics=args.model == "pctcn",
        )
    else:
        model = build_recurrent_baseline(
            args.model,
            n_in=N_IN,
            horizon=HORIZON,
            n_context=contexts_raw.shape[1],
        )
    loss = ControlAwareQuantileLoss(peak_threshold=peak_threshold_scaled)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=loss)

    run_dir = args.output_dir / f"{args.model}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    model_path = run_dir / "best_model.keras"
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(model_path), monitor="val_loss", save_best_only=True, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1
        ),
        keras.callbacks.CSVLogger(str(run_dir / "history.csv")),
    ]
    print(
        f"realizations train/val/test = {len(train_realizations)}/"
        f"{len(val_realizations)}/{len(test_realizations)}"
    )
    print(
        f"windows train/val/test = {len(train_indices)}/"
        f"{len(val_indices)}/{len(test_indices)}"
    )
    model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=2,
    )

    model = keras.models.load_model(str(model_path), compile=False)
    # Avoid Keras' Unicode progress bar, which crashes on Windows CP950 terminals.
    quantiles_scaled = model.predict(test_seq, verbose=0)
    y_true_w, ratings = test_seq.all_targets_and_ratings()
    quantiles_w = quantiles_scaled * ratings[:, None, None]

    overall = metric_summary(y_true_w, quantiles_w, peak_threshold_w)
    overall.update(
        model=args.model,
        seed=args.seed,
        params=int(model.count_params()),
        train_realizations=len(train_realizations),
        validation_realizations=len(val_realizations),
        test_realizations=len(test_realizations),
        test_windows=len(test_indices),
    )
    pd.DataFrame([overall]).to_csv(
        run_dir / "test_metrics.csv", index=False, encoding="utf-8-sig"
    )

    horizon_rows = []
    for k in range(HORIZON):
        values = metric_summary(
            y_true_w[:, k : k + 1],
            quantiles_w[:, k : k + 1, :],
            peak_threshold_w,
        )
        values["horizon_step"] = k + 1
        values["seconds_ahead"] = (k + 1) / float(metadata["fs_hz"])
        horizon_rows.append(values)
    pd.DataFrame(horizon_rows).to_csv(
        run_dir / "test_metrics_by_horizon.csv", index=False, encoding="utf-8-sig"
    )

    np.savez_compressed(
        run_dir / "sampled_test_predictions.npz",
        y_true_W=y_true_w.astype(np.float32),
        quantiles_W=quantiles_w.astype(np.float32),
        window_indices=test_seq.indices,
    )
    normalization = dict(
        context_names=metadata["context_names"],
        context_mean=context_mean.tolist(),
        context_std=context_std.tolist(),
        power_scaling="divide each realization by its rated_power_W context",
        peak_threshold_W=peak_threshold_w,
        dataset=str(args.dataset.resolve()),
    )
    (run_dir / "normalization.json").write_text(
        json.dumps(normalization, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if args.control_realizations > 0:
        export_control_forecasts(
            model,
            powers,
            contexts_raw,
            contexts_scaled,
            splits,
            realization_ids,
            run_dir / "test_control_forecasts.npz",
            args.control_realizations,
            args.batch_size,
        )
    print(json.dumps(overall, indent=2, ensure_ascii=False))
    print(f"saved outputs to {run_dir}")


if __name__ == "__main__":
    main()
