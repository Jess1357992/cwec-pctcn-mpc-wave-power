"""Generate independent spectrum-based CWEC electrical-power realizations.

This replaces the square-wave + Gaussian-noise data.  A single power
coefficient is calibrated at the baseline condition and then held fixed for
every sea state and omega0/omega_p condition.  Therefore geometry/sea-state
effects are not erased by per-realization mean normalization.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


G = 9.81
CONTEXT_NAMES = [
    "Hs_m",
    "Te_s",
    "Tp_s",
    "omega0_over_omegap",
    "zeta",
    "channel_length_m",
    "capture_width_m",
    "rated_power_W",
]


def pm_spectrum(f_hz, hs_m, tp_s):
    """Pierson-Moskowitz spectrum S(f) in m^2/Hz."""
    f_hz = np.asarray(f_hz, dtype=float)
    fp = 1.0 / tp_s
    spectrum = np.zeros_like(f_hz)
    mask = f_hz > 0.0
    spectrum[mask] = (
        (5.0 / 16.0)
        * hs_m**2
        * fp**4
        * f_hz[mask] ** -5
        * np.exp(-1.25 * (fp / f_hz[mask]) ** 4)
    )
    return spectrum


def random_phase_surface(hs_m, tp_s, fs_hz, duration_s, seed):
    """Synthesize an irregular surface elevation using independent phases."""
    rng = np.random.default_rng(seed)
    n = int(round(duration_s * fs_hz))
    df = 1.0 / duration_s
    f_hz = np.arange(max(0.05, df), fs_hz / 2.0, df)
    amplitudes = np.sqrt(2.0 * pm_spectrum(f_hz, hs_m, tp_s) * df)
    phases = rng.uniform(0.0, 2.0 * np.pi, f_hz.size)

    spectrum_fft = np.zeros(n // 2 + 1, dtype=complex)
    indices = np.round(f_hz * duration_s).astype(int)
    valid = (indices > 0) & (indices < spectrum_fft.size)
    spectrum_fft[indices[valid]] = (
        amplitudes[valid] * np.exp(1j * phases[valid]) * n / 2.0
    )
    return np.fft.irfft(spectrum_fft, n=n)


def sdof_response(surface, fs_hz, omega0, zeta):
    """Causal-system frequency response evaluated over one realization."""
    n = surface.size
    wave_fft = np.fft.rfft(surface)
    omega = 2.0 * np.pi * np.fft.rfftfreq(n, d=1.0 / fs_hz)
    response = omega0**2 / (
        omega0**2 - omega**2 + 2j * zeta * omega0 * omega
    )
    return np.fft.irfft(wave_fft * response, n=n)


def unscaled_electrical_driver(
    hs_m,
    te_s,
    omega0_over_omegap,
    zeta,
    fs_hz,
    duration_s,
    seed,
    velocity_exponent=2.0,
):
    """Return the nonnegative wave-to-wire driver before fixed calibration."""
    tp_s = 1.1 * te_s
    omega_p = 2.0 * np.pi / tp_s
    omega0 = omega0_over_omegap * omega_p
    surface = random_phase_surface(hs_m, tp_s, fs_hz, duration_s, seed)
    channel_level = sdof_response(surface, fs_hz, omega0, zeta)
    velocity = np.gradient(channel_level, 1.0 / fs_hz)
    return np.maximum(velocity, 0.0) ** velocity_exponent


def calibrate_power_coefficient(
    fs_hz, duration_s, calibration_seeds, rated_power_w=80.0
):
    """Calibrate once at the manuscript baseline; never per realization."""
    baseline = dict(
        hs_m=0.075,
        te_s=1.56,
        omega0_over_omegap=1.63,
        zeta=0.15,
        fs_hz=fs_hz,
        duration_s=duration_s,
    )
    raw_realizations = [
        unscaled_electrical_driver(seed=seed, **baseline)
        for seed in calibration_seeds
    ]
    target_mean_w = 12.53
    calibration_raw = np.concatenate(raw_realizations)
    raw_mean = float(calibration_raw.mean())
    low = 0.0
    high = target_mean_w / raw_mean

    # Include converter saturation in the one-time calibration; the resulting
    # coefficient is then frozen for every generated realization.
    while np.minimum(high * calibration_raw, rated_power_w).mean() < target_mean_w:
        high *= 2.0
    for _ in range(60):
        middle = 0.5 * (low + high)
        mean_power = np.minimum(middle * calibration_raw, rated_power_w).mean()
        if mean_power < target_mean_w:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def profile_settings(profile):
    if profile == "demo":
        return dict(
            hs_values=[0.060, 0.090],
            te_values=[1.40, 1.75],
            omega_ratios=[1.00, 1.60, 2.10],
            zeta_values=[0.10, 0.20],
            seeds=list(range(100, 104)),
            duration_s=600.0,
            split_fractions=(0.50, 0.25, 0.25),
        )
    if profile == "paper":
        return dict(
            hs_values=[0.050, 0.075, 0.100],
            te_values=[1.30, 1.56, 1.90],
            omega_ratios=[0.80, 1.00, 1.30, 1.60, 2.10],
            zeta_values=[0.10, 0.15, 0.20],
            seeds=list(range(100, 110)),
            duration_s=1200.0,
            split_fractions=(0.60, 0.20, 0.20),
        )
    raise ValueError(f"Unknown profile: {profile}")


def generate(profile, output_dir):
    cfg = profile_settings(profile)
    output_dir.mkdir(parents=True, exist_ok=True)
    fs_hz = 10.0
    channel_length_m = 0.275
    capture_width_m = 1.173
    rated_power_w = 80.0

    calibration_duration = min(cfg["duration_s"], 600.0)
    coefficient = calibrate_power_coefficient(
        fs_hz, calibration_duration, range(9001, 9011)
    )

    powers = []
    contexts = []
    splits = []
    realization_ids = []
    rows = []
    scenarios = [
        (hs_m, te_s, ratio, zeta)
        for hs_m in cfg["hs_values"]
        for te_s in cfg["te_values"]
        for ratio in cfg["omega_ratios"]
        for zeta in cfg["zeta_values"]
    ]
    # Split entire sea-state/resonance scenarios, not overlapping windows and
    # not merely points from the same scenario.  The test grid is therefore
    # unseen during both fitting and checkpoint selection.
    permutation = np.random.default_rng(20260905).permutation(len(scenarios))
    train_fraction, validation_fraction, _ = cfg["split_fractions"]
    n_train_scenarios = int(round(len(scenarios) * train_fraction))
    n_validation_scenarios = int(round(len(scenarios) * validation_fraction))
    scenario_split = {}
    for rank, scenario_index in enumerate(permutation):
        if rank < n_train_scenarios:
            split = "train"
        elif rank < n_train_scenarios + n_validation_scenarios:
            split = "validation"
        else:
            split = "test"
        scenario_split[int(scenario_index)] = split

    for scenario_index, (hs_m, te_s, ratio, zeta) in enumerate(scenarios):
        split = scenario_split[scenario_index]
        for seed_base in cfg["seeds"]:
            # Unique phases across all physical scenarios.
            seed = int(seed_base + 1000 * scenario_index)
            raw = unscaled_electrical_driver(
                hs_m=hs_m,
                te_s=te_s,
                omega0_over_omegap=ratio,
                zeta=zeta,
                fs_hz=fs_hz,
                duration_s=cfg["duration_s"],
                seed=seed,
            )
            power = np.minimum(coefficient * raw, rated_power_w).astype(np.float32)
            context = np.asarray(
                [
                    hs_m,
                    te_s,
                    1.1 * te_s,
                    ratio,
                    zeta,
                    channel_length_m,
                    capture_width_m,
                    rated_power_w,
                ],
                dtype=np.float32,
            )
            realization_id = (
                f"Hs{hs_m:.3f}_Te{te_s:.2f}_r{ratio:.2f}_seed{seed}"
            )
            powers.append(power)
            contexts.append(context)
            splits.append({"train": 0, "validation": 1, "test": 2}[split])
            realization_ids.append(realization_id)
            rows.append(
                dict(
                    realization_id=realization_id,
                    split=split,
                    seed=seed,
                    samples=power.size,
                    mean_W=float(power.mean()),
                    std_W=float(power.std()),
                    max_W=float(power.max()),
                    zero_fraction=float(np.mean(power < 1e-9)),
                    **dict(zip(CONTEXT_NAMES, context.tolist())),
                )
            )

    metadata = dict(
        profile=profile,
        generator="Pierson-Moskowitz + independent random phases + SDOF + fixed wave-to-wire coefficient",
        fs_hz=fs_hz,
        duration_s=cfg["duration_s"],
        context_names=CONTEXT_NAMES,
        split_encoding={"train": 0, "validation": 1, "test": 2},
        split_rule="Entire (Hs, Te, omega0/omegap, zeta) scenarios are assigned to exactly one split; random-phase seeds are unique across scenarios.",
        power_coefficient=float(coefficient),
        calibration="Baseline Hs=0.075 m, Te=1.56 s, omega0/omegap=1.63; coefficient fixed for every realization",
        warning="omega0/omegap is the current geometry proxy. Do not infer alpha until an experimentally supported alpha-to-omega0 calibration is supplied.",
    )

    dataset_path = output_dir / f"spectrum_realizations_{profile}.npz"
    np.savez_compressed(
        dataset_path,
        powers=np.stack(powers),
        contexts=np.stack(contexts),
        splits=np.asarray(splits, dtype=np.int8),
        realization_ids=np.asarray(realization_ids),
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    manifest_path = output_dir / f"manifest_{profile}.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False, encoding="utf-8-sig")
    print(f"saved {dataset_path}")
    print(f"saved {manifest_path}")
    print(pd.DataFrame(rows).groupby("split")["realization_id"].count())
    print(f"fixed power coefficient = {coefficient:.6g}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["demo", "paper"], default="demo")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
    )
    args = parser.parse_args()
    generate(args.profile, args.output_dir)


if __name__ == "__main__":
    main()
