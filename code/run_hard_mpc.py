"""Evaluate spectrum-only forecasts with a true H-step constrained MPC.

Sign convention:
    u > 0  FESS discharges to the grid
    u < 0  FESS charges from the CWEC
    c >= 0 curtailed CWEC power

The optimizer uses P10/P50/P90 scenarios.  Physical storage constraints are
hard constraints; a terminal SOC band prevents free use of initial energy.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def storage_step(
    energy, u, inertia, eta_charge, eta_discharge, dt, parasitic_loss_w=0.0
):
    shaft_power = u / eta_discharge if u >= 0.0 else eta_charge * u
    return energy - (shaft_power + max(float(parasitic_loss_w), 0.0)) * dt


class HardMPC:
    def __init__(
        self,
        horizon,
        dt,
        reference_w,
        rated_power_w,
        inertia,
        omega_min=400.0,
        omega_max=1800.0,
        soc_low=0.10,
        soc_high=0.90,
        command_ramp_wps=1000.0,
        torque_slew_nms=600.0,
        eta_charge=0.95,
        eta_discharge=0.95,
        windage_coefficient=1.0 / 2000.0**2,
        idle_loss_w=0.10,
        terminal_soc=0.50,
        terminal_tolerance=0.01,
    ):
        self.horizon = int(horizon)
        self.dt = float(dt)
        self.reference_w = float(reference_w)
        self.rated_power_w = float(rated_power_w)
        self.inertia = float(inertia)
        self.omega_min = float(omega_min)
        self.omega_max = float(omega_max)
        self.energy_min = 0.5 * inertia * omega_min**2
        self.energy_max = 0.5 * inertia * omega_max**2
        self.energy_span = self.energy_max - self.energy_min
        self.soc_low = float(soc_low)
        self.soc_high = float(soc_high)
        self.command_ramp = float(command_ramp_wps) * dt
        self.torque_slew = float(torque_slew_nms) * dt
        self.eta_charge = float(eta_charge)
        self.eta_discharge = float(eta_discharge)
        self.windage_coefficient = float(windage_coefficient)
        self.idle_loss_w = float(idle_loss_w)
        self.terminal_soc = float(terminal_soc)
        self.terminal_tolerance = float(terminal_tolerance)

    def soc(self, energy):
        return (energy - self.energy_min) / self.energy_span

    def omega(self, energy):
        return np.sqrt(np.maximum(2.0 * energy / self.inertia, 1.0))

    def loss(self, energy):
        omega = self.omega(energy)
        return self.windage_coefficient * omega**2 + self.idle_loss_w

    def rollout_storage(self, u, energy0):
        energies, torques, losses = [], [], []
        energy = float(energy0)
        for command in u:
            omega = self.omega(energy)
            parasitic_loss = self.loss(energy)
            shaft_power = (
                command / self.eta_discharge
                if command >= 0.0
                else self.eta_charge * command
            )
            torques.append(shaft_power / omega)
            losses.append(parasitic_loss)
            energy = storage_step(
                energy,
                command,
                self.inertia,
                self.eta_charge,
                self.eta_discharge,
                self.dt,
                parasitic_loss,
            )
            energies.append(energy)
        return np.asarray(energies), np.asarray(torques), np.asarray(losses)

    def solve(
        self,
        wave_scenarios,
        energy0,
        previous_u,
        previous_torque,
        warm_start,
        terminal_tolerance=None,
    ):
        """Optimize [u(0:H), curtailment(0:H)] for three wave scenarios."""
        scenarios = np.asarray(wave_scenarios, dtype=float)
        if scenarios.ndim != 2 or scenarios.shape[1] != 3:
            raise ValueError("wave_scenarios must have shape (H, 3) for P10/P50/P90")
        horizon = scenarios.shape[0]

        if warm_start is None or len(warm_start) != 2 * horizon:
            x0 = np.zeros(2 * horizon)
        else:
            x0 = warm_start.copy()

        scenario_weights = np.asarray([0.25, 0.50, 0.25])
        power_scale = max(self.rated_power_w, self.reference_w, 1.0)
        terminal_tolerance = (
            self.terminal_tolerance
            if terminal_tolerance is None
            else float(terminal_tolerance)
        )

        def unpack(x):
            return x[:horizon], x[horizon:]

        def objective(x):
            u, curtailment = unpack(x)
            energies, _, _ = self.rollout_storage(u, energy0)
            grid = scenarios + u[:, None] - curtailment[:, None]
            tracking = np.sum(
                scenario_weights[None, :]
                * ((grid - self.reference_w) / power_scale) ** 2
            )
            delta_u = np.diff(np.r_[previous_u, u])
            smoothness = np.sum((delta_u / power_scale) ** 2)
            terminal = (self.soc(energies[-1]) - self.terminal_soc) ** 2
            spill = np.sum((curtailment / power_scale) ** 2)
            # Terminal energy is weighted strongly so a controller cannot
            # appear better merely by spending the initially stored energy.
            return 50.0 * tracking + 0.5 * smoothness + 0.1 * spill + 1.0e6 * terminal

        def constraints(x):
            u, curtailment = unpack(x)
            energies, torques, _ = self.rollout_storage(u, energy0)
            soc = self.soc(energies)
            delta_u = np.diff(np.r_[previous_u, u])
            delta_torque = np.diff(np.r_[previous_torque, torques])
            grid_low = scenarios[:, 0] + u - curtailment
            terminal_margin = terminal_tolerance - abs(
                soc[-1] - self.terminal_soc
            )
            return np.r_[
                soc - self.soc_low,
                self.soc_high - soc,
                (self.command_ramp - np.abs(delta_u))
                / max(self.command_ramp, 1.0),
                (self.torque_slew - np.abs(delta_torque))
                / max(self.torque_slew, 1.0),
                grid_low / power_scale,
                terminal_margin / max(terminal_tolerance, 1e-3),
            ]

        bounds = [(-self.rated_power_w, self.rated_power_w)] * horizon
        bounds += [(0.0, self.rated_power_w)] * horizon
        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints={"type": "ineq", "fun": constraints},
            options={"maxiter": 250, "ftol": 1e-9, "disp": False},
        )
        minimum_constraint = float(np.min(constraints(result.x)))
        # SLSQP often returns status 8 at an otherwise feasible point.  Judge
        # hard feasibility from the scaled constraints, with a numerical-only
        # tolerance, rather than discarding that safe solution.
        feasible = bool(np.all(np.isfinite(result.x)) and minimum_constraint >= -1e-4)
        message = str(result.message)
        if feasible and not result.success:
            message += " (accepted: constraints satisfied within 1e-4)"
        return result.x, feasible, message, minimum_constraint


def make_scenarios(controller, row, current_power, actual_power, origin, horizon):
    if controller == "forecast":
        return row[:horizon]
    if controller == "zoh":
        return np.full((horizon, 3), current_power)
    if controller == "perfect":
        future = actual_power[origin + 1 : origin + 1 + horizon]
        if len(future) < horizon:
            future = np.pad(future, (0, horizon - len(future)), mode="edge")
        return np.repeat(future[:, None], 3, axis=1)
    raise ValueError(controller)


def run_controller(controller, actual_power, quantiles, origins, mpc, steps):
    energy = mpc.energy_min + 0.5 * mpc.energy_span
    initial_energy = energy
    previous_u = 0.0
    previous_torque = 0.0
    warm = None
    records = []
    failures = 0

    for j in range(steps):
        origin = int(origins[j])
        # Shrink the final horizon so the terminal-SOC constraint refers to the
        # actual end of this benchmark, not to unexecuted future actions.
        horizon = min(
            mpc.horizon,
            quantiles.shape[1],
            len(actual_power) - origin - 1,
            steps - j,
        )
        scenarios = make_scenarios(
            controller,
            quantiles[j],
            actual_power[origin],
            actual_power,
            origin,
            horizon,
        )
        if warm is not None:
            old_h = len(warm) // 2
            old_u, old_c = warm[:old_h], warm[old_h:]
            shifted_u = np.r_[old_u[1:], old_u[-1]][:horizon]
            shifted_c = np.r_[old_c[1:], old_c[-1]][:horizon]
            warm_in = np.r_[shifted_u, shifted_c]
        else:
            warm_in = None

        solution, feasible, message, min_constraint = mpc.solve(
            scenarios,
            energy,
            previous_u,
            previous_torque,
            warm_in,
        )
        if not feasible:
            failures += 1
            # Safety fallback respects the one-step storage, ramp, torque and
            # non-import limits whenever their intersection is non-empty.
            requested = mpc.reference_w - float(scenarios[0, 1])
            loss_now = mpc.loss(energy)
            energy_at_low_soc = mpc.energy_min + mpc.soc_low * mpc.energy_span
            energy_at_high_soc = mpc.energy_min + mpc.soc_high * mpc.energy_span
            max_discharge = max(
                0.0,
                ((energy - energy_at_low_soc) / mpc.dt - loss_now)
                * mpc.eta_discharge,
            )
            max_charge = max(
                0.0,
                ((energy_at_high_soc - energy) / mpc.dt + loss_now)
                / mpc.eta_charge,
            )
            omega = mpc.omega(energy)

            def command_from_shaft(shaft_power):
                return (
                    shaft_power * mpc.eta_discharge
                    if shaft_power >= 0.0
                    else shaft_power / mpc.eta_charge
                )

            torque_lower_u = command_from_shaft(
                (previous_torque - mpc.torque_slew) * omega
            )
            torque_upper_u = command_from_shaft(
                (previous_torque + mpc.torque_slew) * omega
            )
            lower = max(
                -mpc.rated_power_w,
                -max_charge,
                previous_u - mpc.command_ramp,
                torque_lower_u,
                -float(scenarios[0, 0]),
            )
            upper = min(
                mpc.rated_power_w,
                max_discharge,
                previous_u + mpc.command_ramp,
                torque_upper_u,
            )
            if lower <= upper:
                u = float(np.clip(requested, lower, upper))
            else:
                # Preserve storage/ramp/torque safety if robust non-import is
                # physically impossible; grid import remains explicitly logged.
                safe_lower = max(
                    -mpc.rated_power_w,
                    -max_charge,
                    previous_u - mpc.command_ramp,
                    torque_lower_u,
                )
                u = float(np.clip(requested, safe_lower, upper))
            curtailment = 0.0
            warm = None
        else:
            u = float(solution[0])
            curtailment = float(solution[horizon])
            warm = solution

        # Low-level causal safety filter.  At actuation time the next sample is
        # the current measured wave power, so charging and curtailment can be
        # reduced without using any future information.  This guarantees the
        # realized grid power is non-negative even when P10 misses the event.
        wave_next = float(actual_power[origin + 1])
        safety_filter_applied = False
        if wave_next + u < 0.0:
            u = -wave_next
            safety_filter_applied = True
        maximum_curtailment = max(wave_next + u, 0.0)
        if curtailment > maximum_curtailment:
            curtailment = maximum_curtailment
            safety_filter_applied = True

        omega = mpc.omega(energy)
        shaft_power = u / mpc.eta_discharge if u >= 0.0 else mpc.eta_charge * u
        torque = shaft_power / omega
        loss = mpc.loss(energy)
        grid_power = wave_next + u - curtailment
        conversion_loss = (
            u / mpc.eta_discharge - u
            if u >= 0.0
            else (-u) - (-mpc.eta_charge * u)
        )
        energy = storage_step(
            energy,
            u,
            mpc.inertia,
            mpc.eta_charge,
            mpc.eta_discharge,
            mpc.dt,
            loss,
        )
        soc = mpc.soc(energy)
        records.append(
            dict(
                step=j,
                time_s=j * mpc.dt,
                origin=origin,
                wave_power_W=wave_next,
                grid_power_W=grid_power,
                fess_command_W=u,
                curtailment_W=curtailment,
                SOC=soc,
                omega_rad_s=mpc.omega(energy),
                torque_Nm=torque,
                parasitic_loss_W=loss,
                conversion_loss_W=conversion_loss,
                safety_filter_applied=safety_filter_applied,
                solver_feasible=feasible,
                minimum_predicted_constraint=min_constraint,
                solver_message=str(message),
            )
        )
        previous_u = u
        previous_torque = torque

    frame = pd.DataFrame(records)
    frame.attrs["initial_energy_J"] = initial_energy
    frame.attrs["final_energy_J"] = energy
    frame.attrs["solver_failures"] = failures
    return frame


def summarize(name, frame, raw_sigma, reference_w):
    wave = frame["wave_power_W"].to_numpy()
    grid = frame["grid_power_W"].to_numpy()
    command = frame["fess_command_W"].to_numpy()
    soc = frame["SOC"].to_numpy()
    curtailment = frame["curtailment_W"].to_numpy()
    parasitic_loss = frame["parasitic_loss_W"].to_numpy()
    conversion_loss = frame["conversion_loss_W"].to_numpy()
    dt = float(frame["time_s"].iloc[1] - frame["time_s"].iloc[0]) if len(frame) > 1 else 0.1
    delta_energy = float(frame.attrs["final_energy_J"] - frame.attrs["initial_energy_J"])
    energy_balance_error = float(
        np.sum(wave) * dt
        - np.sum(grid) * dt
        - np.sum(curtailment) * dt
        - delta_energy
        - np.sum(parasitic_loss) * dt
        - np.sum(conversion_loss) * dt
    )
    return dict(
        controller=name,
        mean_wave_W=float(wave.mean()),
        mean_grid_W=float(grid.mean()),
        sigma_grid_W=float(grid.std()),
        sigma_reduction_pct=float(100.0 * (1.0 - grid.std() / raw_sigma)),
        ramp_max_Wps=float(np.max(np.abs(np.diff(grid))) / dt),
        ramp_mean_Wps=float(np.mean(np.abs(np.diff(grid))) / dt),
        LPSP_pct=float(100.0 * np.maximum(reference_w - grid, 0.0).sum() / (reference_w * len(grid))),
        curtailment_pct=float(100.0 * curtailment.sum() / max(wave.sum(), 1e-12)),
        SOC_min=float(soc.min()),
        SOC_max=float(soc.max()),
        terminal_delta_SOC=float(soc[-1] - 0.50),
        grid_import_count=int(np.sum(grid < -1e-8)),
        safety_filter_count=int(frame["safety_filter_applied"].sum()),
        hard_SOC_violation_count=int(np.sum((soc < 0.10 - 1e-6) | (soc > 0.90 + 1e-6))),
        solver_failure_count=int(frame.attrs["solver_failures"]),
        delta_storage_energy_J=delta_energy,
        energy_balance_error_J=energy_balance_error,
        mean_abs_command_W=float(np.mean(np.abs(command))),
    )


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--forecasts",
        type=Path,
        default=root / "outputs" / "pctcn_seed42" / "test_control_forecasts.npz",
    )
    parser.add_argument("--realization", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--reference-W", type=float, default=12.0)
    parser.add_argument("--ride-through-seconds", type=float, default=30.0)
    parser.add_argument("--output-dir", type=Path, default=root / "outputs" / "mpc")
    args = parser.parse_args()

    data = np.load(args.forecasts, allow_pickle=False)
    actual_power = data["powers"][args.realization].astype(float)
    quantiles = data["quantiles"][args.realization].astype(float)
    origins = data["forecast_origins"].astype(int)
    context = data["contexts"][args.realization]
    realization_id = str(data["realization_ids"][args.realization])
    fs_hz = float(data["fs_hz"])
    dt = 1.0 / fs_hz
    rated_power_w = float(context[7])
    horizon = int(data["horizon"])
    steps = min(int(round(args.seconds * fs_hz)), len(origins))

    omega_min, omega_max = 400.0, 1800.0
    usable_energy = args.reference_W * args.ride_through_seconds
    inertia = 2.0 * usable_energy / (omega_max**2 - omega_min**2)
    mpc = HardMPC(
        horizon=horizon,
        dt=dt,
        reference_w=args.reference_W,
        rated_power_w=rated_power_w,
        inertia=inertia,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    origin0 = origins[0]
    raw = actual_power[origin0 + 1 : origin0 + 1 + steps]
    raw_sigma = float(raw.std())
    metrics = [
        dict(
            controller="no_storage",
            mean_wave_W=float(raw.mean()),
            mean_grid_W=float(raw.mean()),
            sigma_grid_W=raw_sigma,
            sigma_reduction_pct=0.0,
            ramp_max_Wps=float(np.max(np.abs(np.diff(raw))) / dt),
            ramp_mean_Wps=float(np.mean(np.abs(np.diff(raw))) / dt),
            LPSP_pct=float(100.0 * np.maximum(args.reference_W - raw, 0.0).sum() / (args.reference_W * len(raw))),
            curtailment_pct=0.0,
            SOC_min=np.nan,
            SOC_max=np.nan,
            terminal_delta_SOC=np.nan,
            grid_import_count=0,
            safety_filter_count=0,
            hard_SOC_violation_count=0,
            solver_failure_count=0,
            delta_storage_energy_J=0.0,
            energy_balance_error_J=0.0,
            mean_abs_command_W=0.0,
        )
    ]

    traces = {}
    for controller in ("zoh", "forecast", "perfect"):
        print(f"running {controller} MPC for {steps} steps")
        trace = run_controller(
            controller, actual_power, quantiles, origins, mpc, steps
        )
        trace.to_csv(
            args.output_dir / f"trace_{controller}.csv",
            index=False,
            encoding="utf-8-sig",
        )
        traces[controller] = trace
        metrics.append(summarize(controller, trace, raw_sigma, args.reference_W))

    metrics_frame = pd.DataFrame(metrics)
    metrics_frame.insert(0, "realization_id", realization_id)
    metrics_frame.to_csv(
        args.output_dir / "mpc_metrics.csv", index=False, encoding="utf-8-sig"
    )
    print(metrics_frame.to_string(index=False))
    print(f"saved outputs to {args.output_dir}")

    try:
        import matplotlib.pyplot as plt

        time = np.arange(steps) * dt
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        axes[0].plot(time, raw, color="0.65", lw=1.0, label="raw spectrum-based CWEC power")
        for name, color in (("zoh", "tab:orange"), ("forecast", "tab:blue"), ("perfect", "tab:green")):
            axes[0].plot(time, traces[name]["grid_power_W"], lw=1.0, color=color, label=name)
        axes[0].axhline(args.reference_W, color="red", ls="--", lw=1.0, label="reference")
        axes[0].set_ylabel("Power (W)")
        axes[0].legend(ncol=2, fontsize=8)
        axes[0].grid(alpha=0.25)
        for name, color in (("zoh", "tab:orange"), ("forecast", "tab:blue"), ("perfect", "tab:green")):
            axes[1].plot(time, traces[name]["SOC"], lw=1.0, color=color, label=name)
        axes[1].axhline(0.10, color="red", ls=":")
        axes[1].axhline(0.90, color="red", ls=":")
        axes[1].set_ylabel("SOC")
        axes[1].set_xlabel("Time (s)")
        axes[1].grid(alpha=0.25)
        fig.suptitle(f"Hard-constrained H={horizon} MPC — {realization_id}")
        fig.tight_layout()
        fig.savefig(args.output_dir / "mpc_comparison.png", dpi=300)
    except ImportError:
        print("matplotlib is not installed; skipped plot")


if __name__ == "__main__":
    main()
