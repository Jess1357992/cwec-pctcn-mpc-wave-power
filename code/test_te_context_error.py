"""Fixed-model sensitivity to estimated Te errors; no training or new waves.

Run with the Python environment used for training:
    python test_te_context_error.py
Requires numpy, pandas, tensorflow (same versions as the trained model).
"""
import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')


def perturb_context(raw, names, error_pct):
    """Keep omega0 fixed: omega0/omegap scales with estimated Tp."""
    import numpy as np
    c = raw.copy()
    te, tp, ratio = [names.index(n) for n in ('Te_s', 'Tp_s', 'omega0_over_omegap')]
    if not np.allclose(raw[:, tp], 1.1 * raw[:, te], rtol=1e-6):
        raise ValueError('Original context does not satisfy Tp = 1.1 Te.')
    # Preserve the original zero-error context bit for bit.
    if error_pct != 0:
        c[:, te] = raw[:, te] * (1 + error_pct / 100)
        c[:, tp] = 1.1 * c[:, te]
        c[:, ratio] = raw[:, ratio] * c[:, tp] / raw[:, tp]
    if not np.allclose(c[:, ratio] / c[:, tp], raw[:, ratio] / raw[:, tp], rtol=1e-6):
        raise AssertionError('Natural frequency must remain fixed.')
    return c


def scores(target, median):
    import numpy as np
    a, b = np.asarray(target, dtype=float), np.asarray(median, dtype=float)
    sse = np.sum((a - b) ** 2)
    sst = np.sum((a - a.mean()) ** 2)
    return {'R2_NSE': float(1 - sse / sst) if sst > 0 else float('nan'),
            'RMSE_W': float(np.sqrt(sse / a.size))}


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, default=root / 'data/spectrum_realizations_demo.npz')
    p.add_argument('--run-dir', type=Path, default=root / 'outputs/pctcn_seed42')
    p.add_argument('--output-dir', type=Path, default=root / 'outputs/te_context_error')
    p.add_argument('--batch-size', type=int, default=128)
    args = p.parse_args()
    if args.batch_size <= 0:
        p.error('batch-size must be positive')
    import numpy as np
    from tensorflow import keras
    from train_pctcn import WindowSequence, N_IN, HORIZON
    # Import above also registers custom layers from pctcn_model.
    data = np.load(args.dataset, allow_pickle=False)
    powers = data['powers'].astype(np.float32)
    raw = data['contexts'].astype(np.float32)
    saved = np.load(args.run_dir / 'sampled_test_predictions.npz', allow_pickle=False)
    indices = saved['window_indices']
    if not np.all(data['splits'][indices[:, 0]] == 2):
        raise ValueError('Saved windows must all belong to the test partition.')
    norm = json.loads((args.run_dir / 'normalization.json').read_text(encoding='utf-8'))
    names = norm['context_names']
    metadata = json.loads(str(data['metadata_json']))
    if names != metadata['context_names']:
        raise ValueError('Dataset and saved normalization context order differ.')
    mean = np.asarray(norm['context_mean'], dtype=np.float32)
    std = np.asarray(norm['context_std'], dtype=np.float32)
    if not np.all(std > 0):
        raise ValueError('Invalid normalization standard deviations.')
    model = keras.models.load_model(args.run_dir / 'best_model.keras', compile=False)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    overall, horizons, audit = [], [], []
    baseline_rmse = None
    for error in (0, -20, -10, 10, 20):
        c = perturb_context(raw, names, error)
        sequence = WindowSequence(powers, (c - mean) / std, raw, indices,
                                  args.batch_size, False, 44)
        target, ratings = sequence.all_targets_and_ratings()
        np.testing.assert_allclose(target, saved['y_true_W'], rtol=0, atol=0,
                                   err_msg='Dataset targets differ from saved evaluation.')
        prediction = model.predict(sequence, verbose=0) * ratings[:, None, None]
        if not np.all(np.isfinite(prediction)):
            raise ValueError('Nonfinite model prediction.')
        result = scores(target, prediction[..., 1])
        if error == 0:
            max_diff = float(np.max(np.abs(prediction - saved['quantiles_W'])))
            np.testing.assert_allclose(prediction, saved['quantiles_W'], rtol=1e-4, atol=1e-3,
                                       err_msg='Zero-bias inference does not reproduce saved predictions.')
            baseline_rmse = result['RMSE_W']
            if baseline_rmse <= 0:
                raise ValueError('Relative increase undefined for zero baseline RMSE.')
        result = {'Te_error_pct': error, **result,
                  'RMSE_increase_pct': 100 * (result['RMSE_W'] / baseline_rmse - 1),
                  'test_windows': len(indices)}
        overall.append(result)
        print(f"Te error {error:+d}%: R2={result['R2_NSE']:.5f}, "
              f"RMSE={result['RMSE_W']:.5f} W, change={result['RMSE_increase_pct']:+.3f}%", flush=True)
        for k in range(HORIZON):
            horizons.append({'Te_error_pct': error, 'seconds_ahead': (k + 1) / metadata['fs_hz'],
                             **scores(target[:, k], prediction[:, k, 1])})
        for i in np.unique(indices[:, 0]):
            audit.append({'Te_error_pct': error, 'realization_id': str(data['realization_ids'][i]),
                          **{n: float(c[i, names.index(n)]) for n in
                             ('Te_s', 'Tp_s', 'omega0_over_omegap')},
                          'omega0_rad_s': float(2 * np.pi * c[i, 3] / c[i, 2])})
    overall.sort(key=lambda row: row['Te_error_pct'])
    write_csv(out / 'te_error_summary.csv', overall)
    write_csv(out / 'te_error_by_horizon.csv', horizons)
    write_csv(out / 'context_audit.csv', audit)
    (out / 'protocol.json').write_text(json.dumps({
        'dataset': str(args.dataset.resolve()), 'model': str((args.run_dir / 'best_model.keras').resolve()),
        'windows_source': str((args.run_dir / 'sampled_test_predictions.npz').resolve()),
        'normalization': 'Frozen training statistics from normalization.json',
        'input_samples': N_IN, 'horizon': HORIZON,
        'zero_bias_max_abs_prediction_difference_W': max_diff,
        'perturbation': 'Te estimated bias; Tp=1.1Te; ratio=original_ratio*estimated_Tp/original_Tp; omega0 fixed',
        'interpretation': 'Joint consistent context response to one period estimation error; not an isolated Te-feature ablation. Waveforms and targets unchanged. One saved model; no retraining. Signed RMSE change may be negative. Some perturbed contexts may lie outside training coverage.'
    }, indent=2), encoding='utf-8')
    print(f'Results saved to: {out.resolve()}')


if __name__ == '__main__':
    main()
