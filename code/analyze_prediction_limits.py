"""Summarize observed forecast-horizon limits from saved metrics; no training.

Run: python analyze_prediction_limits.py
Uses only Python's standard library. Paths default to locations beside this script.
Thresholds are descriptive choices, not universal engineering requirements.
"""
import argparse
import csv
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=root / 'outputs/pctcn_seed42/test_metrics_by_horizon.csv')
    parser.add_argument('--output-dir', type=Path, default=root / 'outputs/prediction_limits')
    args = parser.parse_args()
    with args.input.open(encoding='utf-8-sig', newline='') as f:
        rows = sorted(csv.DictReader(f), key=lambda r: float(r['seconds_ahead']))
    if not rows:
        raise ValueError('The input metrics file is empty.')
    times = [float(r['seconds_ahead']) for r in rows]
    scores = [float(r['R2_NSE']) for r in rows]
    import math
    if not all(math.isfinite(x) for x in times + scores):
        raise ValueError('Time and R2 values must be finite.')
    if len(set(times)) != len(times):
        raise ValueError('Expected one row per forecast time.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for threshold in (0.5, 0.1, 0.0):
        crossing = next((i for i, score in enumerate(scores) if score < threshold), None)
        summary.append({
            'R2_threshold': threshold,
            'first_below_seconds': '' if crossing is None else times[crossing],
            'previous_sample_seconds': '' if crossing in (None, 0) else times[crossing - 1],
            'R2_at_first_below': '' if crossing is None else scores[crossing],
        })
    with (args.output_dir / 'thresholds.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    with (args.output_dir / 'metrics_by_horizon.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # Standalone SVG keeps this small script dependency-free.
    ymin, ymax = min(-0.3, min(scores) - 0.05), max(1.0, max(scores) + 0.05)
    x = lambda t: 85 + 650 * (t - times[0]) / (times[-1] - times[0] or 1)
    y = lambda r: 355 - 290 * (r - ymin) / (ymax - ymin)
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="800" height="440" viewBox="0 0 800 440">',
           '<rect width="800" height="440" fill="white"/>',
           '<g font-family="Arial,sans-serif" font-size="14" fill="#222">',
           '<text x="400" y="30" text-anchor="middle">Observed R² by forecast lead time</text>',
           '<path d="M85 65 V355 H735" fill="none" stroke="#222"/>']
    for threshold in (-0.2, 0.0, 0.1, 0.5, 1.0):
        if ymin <= threshold <= ymax:
            yy = y(threshold)
            svg += [f'<path d="M85 {yy} H735" stroke="#aaa" stroke-dasharray="4 4"/>',
                    f'<text x="75" y="{yy + 5}" text-anchor="end">{threshold:g}</text>']
    points = ' '.join(f'{x(t):.2f},{y(r):.2f}' for t, r in zip(times, scores))
    svg.append(f'<polyline points="{points}" fill="none" stroke="#2463a0" stroke-width="2"/>')
    for i, (t, r) in enumerate(zip(times, scores)):
        svg.append(f'<circle cx="{x(t)}" cy="{y(r)}" r="3" fill="#2463a0"/>')
        if i % 2 == 0 or i == len(times) - 1:
            svg.append(f'<text x="{x(t)}" y="380" text-anchor="middle">{t:g}</text>')
    svg += ['<text x="400" y="410" text-anchor="middle">Forecast lead time (s)</text>',
            '<text x="25" y="210" transform="rotate(-90 25 210)" text-anchor="middle">R² / NSE</text>',
            '</g></svg>']
    (args.output_dir / 'r2_by_horizon.svg').write_text('\n'.join(svg), encoding='utf-8')

    lines = ['Observed forecast-horizon performance', '', f'Source: {args.input.resolve()}', '',
             f'R2 at {times[0]:g} s: {scores[0]:.4f}',
             f'R2 at {times[-1]:g} s: {scores[-1]:.4f}', '']
    for row in summary:
        when = row['first_below_seconds']
        lines.append(f"First sampled lead time with R2 < {row['R2_threshold']:g}: "
                     + ('not observed' if when == '' else f'{when:g} s'))
    lines += ['',
        'These are observed first crossings on the evaluated time grid, not interpolated limits.',
        'Thresholds 0.5 and 0.1 are descriptive choices, not validated application requirements.',
        'Negative R2 means greater squared error than the mean of the evaluation targets at that lead time.',
        'That reference mean is not a deployable training-mean forecasting baseline.',
        'Results describe the saved model and test set; they do not establish a theoretical predictability limit.',
        'This script does not vary wave height or period, retrain models, or evaluate closed-loop control.',
        'For the default input, results are conditional on training seed 42 and the existing test scenarios.']
    report = '\n'.join(lines) + '\n'
    (args.output_dir / 'interpretation.txt').write_text(report, encoding='utf-8')
    print(report)
    print(f'Outputs: {args.output_dir.resolve()}')


if __name__ == '__main__':
    main()
