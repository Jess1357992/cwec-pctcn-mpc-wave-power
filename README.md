# CWEC wave-power dataset (spectrum-driven, 96 realizations)

Dataset accompanying the manuscript:

> Ou, T.-C., Chen, Y.-T., Ju, Z.-C., Luo, S.-Y. *Linking channel resonance to
> forecast-assisted power smoothing in a coastal wave energy converter with flywheel
> energy storage.* Submitted to **Ocean Engineering**, 2026.

Contact: Ting-Chia Ou, outc@gs.ncku.edu.tw, National Cheng Kung University, Taiwan.

---

## Nature of the data

**Simulated, not measured.** The electrical-power records are synthesized numerically;
no field or laboratory measurement is included in this dataset, and no third-party data
is redistributed.

Each record is produced as follows:

1. A Pierson–Moskowitz variance-density spectrum is evaluated for the record's significant
   wave height and peak period.
2. Independent uniform random phases are drawn and the surface elevation is synthesized by
   inverse FFT.
3. The elevation passes through the single-degree-of-freedom frequency response of the
   converter's tapered collection channel, whose natural frequency is `omega0_over_omegap`
   times the spectral peak and whose damping ratio is `zeta`.
4. The channel velocity is rectified and squared to give a non-negative wave-to-wire driver.
5. The driver is multiplied by a single power coefficient (928.0679 W per unit driver) and
   clipped at the 80 W converter rating.

The coefficient in step 5 was calibrated **once**, at Hs = 0.075 m, Te = 1.56 s,
ω₀/ωp = 1.63, ζ = 0.15, so that the saturated mean power equals 12.53 W. It is then held
fixed for every record. Because it is never refitted per realization, differences in mean
power and in temporal structure between sea states remain in the data.

## File list

| File | Size | Contents |
|---|---|---|
| `data/spectrum_realizations_demo.npz` | 1.1 MB | 96 power records, 6000 samples each (600 s at 10 Hz), with context and split labels. |
| `data/manifest_demo.csv` | 22 KB | 96 rows × 16 columns; one row per record. |

## Field descriptions

### `spectrum_realizations_demo.npz`

Load with `numpy.load(path, allow_pickle=False)`.

| Key | Shape | Type | Unit | Description |
|---|---|---|---|---|
| `powers` | (96, 6000) | float32 | W | Electrical power. Range 0 to 80 (saturated at the rating). |
| `contexts` | (96, 8) | float32 | mixed | Physical context of each record; columns below. |
| `splits` | (96,) | int8 | – | 0 = train, 1 = validation, 2 = test. |
| `realization_ids` | (96,) | str | – | e.g. `Hs0.060_Te1.40_r1.60_seed2100`. |
| `metadata_json` | scalar | str | – | JSON: generator, sampling rate, duration, split rule, power coefficient, calibration point. |

Columns of `contexts`, in order:

| # | Name | Unit | Values |
|---|---|---|---|
| 0 | `Hs_m` | m | 0.060, 0.090 |
| 1 | `Te_s` | s | 1.40, 1.75 |
| 2 | `Tp_s` | s | 1.1 × `Te_s` |
| 3 | `omega0_over_omegap` | – | 1.00, 1.60, 2.10 |
| 4 | `zeta` | – | 0.10, 0.20 |
| 5 | `channel_length_m` | m | 0.275 (constant) |
| 6 | `capture_width_m` | m | 1.173 (constant) |
| 7 | `rated_power_W` | W | 80.0 (constant) |

Only `Hs_m`, `Te_s`, `omega0_over_omegap` and `zeta` vary independently: 24 combinations,
four random-phase realizations each.

### `manifest_demo.csv`

UTF-8 with BOM. 96 rows.

| Column | Unit | Description |
|---|---|---|
| `realization_id` | – | Matches `realization_ids` in the archive. |
| `split` | – | `train` / `validation` / `test`. |
| `seed` | – | Random-phase seed, unique across all scenarios. |
| `samples` | – | Samples in the record (6000). |
| `mean_W`, `std_W`, `max_W` | W | Summary statistics of the record. |
| `zero_fraction` | – | Fraction of samples below 1e-9 W, from the generating-direction rectification (about 0.5). |
| `Hs_m` … `rated_power_W` | – | The eight context values, as above. |

## Partitioning

Complete physical scenarios — not overlapping windows — are assigned to training,
validation and test in a 50:25:25 split. Every realization of a given
(Hs, Te, ω₀/ωp, ζ) combination stays in one partition, so the test grid is unseen during
both fitting and checkpoint selection.

## Licence

Creative Commons Attribution 4.0 International (CC BY 4.0). See `LICENSE`.
If you use this dataset, please cite the article above.

## Funding

[待填：國科會計畫編號]
