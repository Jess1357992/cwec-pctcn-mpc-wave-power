# Data dictionary

## `spectrum_realizations_demo.npz`

NumPy compressed archive. Load with `numpy.load(path, allow_pickle=False)`.

| Key | Shape | Type | Description |
|---|---|---|---|
| `powers` | (96, 6000) | float32 | Electrical power in watts, 600 s sampled at 10 Hz, one row per realization. Saturated at the 80 W converter rating. |
| `contexts` | (96, 8) | float32 | Physical context of each realization; column order given below. |
| `splits` | (96,) | int8 | Partition label: 0 = train, 1 = validation, 2 = test. |
| `realization_ids` | (96,) | str | Identifier of the form `Hs0.060_Te1.40_r1.60_seed2100`. |
| `metadata_json` | scalar | str | JSON string recording the generator, sampling rate, duration, split rule, the fitted power coefficient and the calibration condition. |

### Columns of `contexts`

| # | Name | Unit | Values in this dataset |
|---|---|---|---|
| 0 | `Hs_m` | m | 0.060, 0.090 |
| 1 | `Te_s` | s | 1.40, 1.75 |
| 2 | `Tp_s` | s | 1.1 × `Te_s` |
| 3 | `omega0_over_omegap` | – | 1.00, 1.60, 2.10 |
| 4 | `zeta` | – | 0.10, 0.20 |
| 5 | `channel_length_m` | m | 0.275 (fixed) |
| 6 | `capture_width_m` | m | 1.173 (fixed) |
| 7 | `rated_power_W` | W | 80.0 (fixed) |

Only `Hs_m`, `Te_s`, `omega0_over_omegap` and `zeta` vary independently; `Tp_s` is tied to
`Te_s` and the last three are constant. Conditioning therefore represents sea state and
resonance, not a complete geometric description of the device.

## `manifest_demo.csv`

One row per realization, UTF-8 with BOM.

| Column | Description |
|---|---|
| `realization_id` | Matches `realization_ids` in the archive. |
| `split` | `train`, `validation` or `test`. |
| `seed` | Random-phase seed, unique across all physical scenarios. |
| `samples` | Number of samples in the record (6000). |
| `mean_W`, `std_W`, `max_W` | Summary statistics of the power record, in watts. |
| `zero_fraction` | Fraction of samples below 1e-9 W, a consequence of the generating-direction rectification. |
| `Hs_m` … `rated_power_W` | The eight context values, as above. |

## How the records are generated

1. A Pierson–Moskowitz variance-density spectrum is evaluated for the realization's `Hs_m`
   and `Tp_s`.
2. Independent uniform random phases are drawn and the surface elevation is synthesized by
   inverse FFT.
3. The elevation is passed through the single-degree-of-freedom frequency response of the
   collection channel, whose natural frequency is `omega0_over_omegap` times the spectral
   peak and whose damping is `zeta`.
4. The channel velocity is rectified and squared to form a non-negative wave-to-wire driver.
5. The driver is multiplied by one power coefficient — calibrated once at Hs = 0.075 m,
   Te = 1.56 s, ω0/ωp = 1.63, ζ = 0.15 so that the saturated mean power equals 12.53 W, and
   then held fixed for every realization — and clipped at the 80 W rating.

Step 5 is what preserves the influence of sea state and resonance on both the mean and the
temporal structure of the output. The coefficient is never refitted per record.
