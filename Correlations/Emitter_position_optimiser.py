"""

Backlash-safe 2D emitter position optimizer with automatic scan plotting.

Purpose
-------
This module performs a small XY scan around the current emitter position,
measures count rate at each point, finds the local maximum, optionally moves
the stage to the optimized position, saves the result, and plots the scan map.

Backlash strategy
-----------------
For every final target position, the axis is approached from a consistent
direction:

    pre_position = target - approach_direction * backlash_overshoot
    move to pre_position
    move to target

If approach_direction = +1:
    target is approached from lower coordinate.

If approach_direction = -1:
    target is approached from higher coordinate.

During the scan:
    - each Y row is approached backlash-safe,
    - X is reset to the row start backlash-safe,
    - X is scanned monotonically in the chosen approach direction,
    - no serpentine reversal is used.

Expected external functions
---------------------------
You provide these when creating the optimizer:

    move_absolute(axis, position)
    get_position(axis)
    get_count_rate()

Example:
    optimizer = BacklashSafeEmitterOptimizer(
        move_absolute=StepperDevice.move_absolute,
        get_position=lambda axis: StepperDevice.get_all()[0][axis],
        get_count_rate=lambda: TCSPC_DEVICE.get_channels_rate()[CHANNEL - 1],
        x_axis=0,
        y_axis=1,
    )
    
"""

import os
import time
import json
import traceback
from dataclasses import dataclass, asdict
from typing import Callable, Dict, Tuple, Optional, Any

import numpy as np


try:
    import h5py
    H5PY_AVAILABLE = True
except Exception:
    H5PY_AVAILABLE = False


try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False


@dataclass
class OptimizationResult:
    accepted: bool
    reason: str

    old_position: Dict[int, float]
    new_position: Dict[int, float]
    estimated_position: Dict[int, float]

    initial_rate: float
    final_rate: float
    peak_rate: float
    background_rate: float
    contrast_peak_over_background: float

    peak_index_y: int
    peak_index_x: int
    peak_on_edge: bool

    x_axis: int
    y_axis: int

    scan_time_start: float
    scan_time_end: float

    localization_method: str

    scan_shape_yx: Tuple[int, int]

    hdf5_path: Optional[str]
    hdf5_group: Optional[str]
    plot_path: Optional[str]

    metadata: Dict[str, Any]


class BacklashSafeEmitterOptimizer:
    def __init__(
        self,
        move_absolute: Callable[[int, float], None],
        get_position: Callable[[int], float],
        get_count_rate: Callable[[], float],
        x_axis: int = 0,
        y_axis: int = 1,
        approach_direction: Optional[Dict[int, int]] = None,
        backlash_overshoot: Optional[Dict[int, float]] = None,
        settling_time: float = 0.2,
        sample_interval: float = 0.1,
        verbose: bool = True,
    ):
        """
        Parameters
        ----------
        move_absolute:
            Function used to move one stage axis.

        get_position:
            Function used to read one axis position.

        get_count_rate:
            Function returning current detector count rate.

        x_axis, y_axis:
            Stage axes used for XY optimization.

        approach_direction:
            Dictionary defining final approach direction for each axis.

            Example:
                {0: +1, 1: +1}

            +1 means target is approached from lower coordinate.
            -1 means target is approached from higher coordinate.

        backlash_overshoot:
            Overshoot distance for each axis.
            Must be larger than the real mechanical backlash.

            Example:
                {0: 0.003, 1: 0.003}

            Units must match your stage coordinates.

        settling_time:
            Wait time after each stage move.

        sample_interval:
            Time between repeated count-rate reads during dwell.

        verbose:
            Print progress information.
        """

        self.move_absolute = move_absolute
        self.get_position = get_position
        self.get_count_rate = get_count_rate

        self.x_axis = x_axis
        self.y_axis = y_axis
        self.axes = [x_axis, y_axis]

        if approach_direction is None:
            approach_direction = {
                x_axis: +1,
                y_axis: +1,
            }

        if backlash_overshoot is None:
            backlash_overshoot = {
                x_axis: 0.003,
                y_axis: 0.003,
            }

        self.approach_direction = approach_direction
        self.backlash_overshoot = backlash_overshoot

        self.settling_time = settling_time
        self.sample_interval = sample_interval
        self.verbose = verbose

        for axis in self.axes:
            if axis not in self.approach_direction:
                raise ValueError(f"Missing approach_direction for axis {axis}")
            if axis not in self.backlash_overshoot:
                raise ValueError(f"Missing backlash_overshoot for axis {axis}")
            if int(self.approach_direction[axis]) not in [-1, +1]:
                raise ValueError("approach_direction values must be +1 or -1")

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------

    def log(self, msg: str):
        if self.verbose:
            print(msg)

    # -------------------------------------------------------------------------
    # Position helpers
    # -------------------------------------------------------------------------

    def get_current_xy(self) -> Dict[int, float]:
        return {
            self.x_axis: float(self.get_position(self.x_axis)),
            self.y_axis: float(self.get_position(self.y_axis)),
        }

    def backlash_approach_axis(self, axis: int, target: float):
        """
        Move one axis to target using backlash-compensated approach.

        Final movement is always in the selected approach direction.
        """

        direction = int(self.approach_direction[axis])
        overshoot = float(self.backlash_overshoot[axis])

        pre_position = target - direction * overshoot

        self.log(
            f"Axis {axis}: backlash approach -> "
            f"pre={pre_position:.9f}, target={target:.9f}"
        )

        self.move_absolute(axis, pre_position)
        time.sleep(self.settling_time)

        self.move_absolute(axis, target)
        time.sleep(self.settling_time)

    def backlash_approach_position(self, position: Dict[int, float]):
        """
        Move XY position using backlash-compensated approach on both axes.
        """

        for axis in self.axes:
            self.backlash_approach_axis(axis, float(position[axis]))
            self.time.sleep(2)

    # -------------------------------------------------------------------------
    # Count-rate measurement
    # -------------------------------------------------------------------------

    def measure_rate(
        self,
        dwell_time: float,
        statistic: str = "mean",
    ) -> Tuple[float, float, int]:
        """
        Measure count rate for dwell_time.

        Returns
        -------
        rate:
            Mean or median count rate.

        std:
            Standard deviation of sampled rates.

        n:
            Number of samples.
        """

        values = []
        t0 = time.time()

        while time.time() - t0 < dwell_time:
            try:
                values.append(float(self.get_count_rate()))
            except Exception:
                traceback.print_exc()

            time.sleep(self.sample_interval)

        if len(values) == 0:
            return 0.0, 0.0, 0

        values = np.array(values, dtype=float)

        if statistic == "median":
            rate = float(np.median(values))
        elif statistic == "mean":
            rate = float(np.mean(values))
        else:
            raise ValueError("statistic must be 'mean' or 'median'")

        std = float(np.std(values))

        return rate, std, int(len(values))

    # -------------------------------------------------------------------------
    # Scan grid
    # -------------------------------------------------------------------------

    def make_axis_positions(
        self,
        center: float,
        half_range: float,
        step: float,
        axis: int,
    ) -> np.ndarray:
        """
        Generate scan positions for one axis.

        The order is chosen so that scan motion is mostly in the same direction
        as the backlash approach direction.
        """

        if step <= 0:
            raise ValueError("step must be positive")

        if half_range <= 0:
            raise ValueError("half_range must be positive")

        n_steps_each_side = int(np.floor(half_range / step))

        offsets = (
            np.arange(
                -n_steps_each_side,
                n_steps_each_side + 1,
                dtype=float,
            )
            * step
        )

        positions = center + offsets

        direction = int(self.approach_direction[axis])

        if direction == +1:
            positions = np.sort(positions)
        elif direction == -1:
            positions = np.sort(positions)[::-1]

        return positions

    # -------------------------------------------------------------------------
    # Peak localization
    # -------------------------------------------------------------------------

    def localize_peak(
        self,
        x_positions: np.ndarray,
        y_positions: np.ndarray,
        counts_map: np.ndarray,
        method: str = "centroid",
        centroid_fraction: float = 0.50,
    ) -> Tuple[Dict[int, float], Dict[str, Any]]:
        """
        Estimate emitter position from 2D count map.

        method:
            "max"      -> use brightest measured pixel
            "centroid" -> weighted centroid around the bright region
        """

        counts = np.array(counts_map, dtype=float)

        iy_peak, ix_peak = np.unravel_index(np.argmax(counts), counts.shape)

        peak_rate = float(counts[iy_peak, ix_peak])
        background_rate = float(np.median(counts))

        peak_on_edge = (
            iy_peak == 0
            or iy_peak == counts.shape[0] - 1
            or ix_peak == 0
            or ix_peak == counts.shape[1] - 1
        )

        if method == "max":
            x_est = float(x_positions[ix_peak])
            y_est = float(y_positions[iy_peak])

        elif method == "centroid":
            threshold = background_rate + centroid_fraction * (
                peak_rate - background_rate
            )

            mask = counts >= threshold

            weights = counts - background_rate
            weights[weights < 0] = 0
            weights = weights * mask

            if np.sum(weights) <= 0:
                x_est = float(x_positions[ix_peak])
                y_est = float(y_positions[iy_peak])
            else:
                X, Y = np.meshgrid(x_positions, y_positions)
                x_est = float(np.sum(X * weights) / np.sum(weights))
                y_est = float(np.sum(Y * weights) / np.sum(weights))

        else:
            raise ValueError("method must be 'max' or 'centroid'")

        if background_rate > 0:
            contrast = peak_rate / background_rate
        else:
            contrast = np.inf

        estimated_position = {
            self.x_axis: x_est,
            self.y_axis: y_est,
        }

        info = {
            "peak_rate": peak_rate,
            "background_rate": background_rate,
            "contrast_peak_over_background": float(contrast),
            "peak_index_y": int(iy_peak),
            "peak_index_x": int(ix_peak),
            "peak_on_edge": bool(peak_on_edge),
            "method": method,
        }

        return estimated_position, info

    # -------------------------------------------------------------------------
    # Main function
    # -------------------------------------------------------------------------

    def scan_and_optimize(
        self,
        half_range: Optional[Dict[int, float]] = None,
        step_size: Optional[Dict[int, float]] = None,
        dwell_time: float = 0.3,
        rate_statistic: str = "mean",
        localization_method: str = "centroid",
        centroid_fraction: float = 0.50,
        move_to_result: bool = True,
        reject_edge_maximum: bool = True,
        min_peak_over_background: float = 1.15,
        min_peak_rate: Optional[float] = None,
        max_allowed_shift: Optional[Dict[int, float]] = None,
        verify_final_position: bool = True,
        final_verify_dwell_time: float = 1.0,
        save_hdf5_path: Optional[str] = None,
        hdf5_group_name: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
        plot_result: bool = True,
        show_plot: bool = True,
        plot_block: bool = False,
        plot_pause: float = 0.2,
        close_plot_after: bool = False,
        save_plot: bool = True,
        save_plot_path: Optional[str] = None,
        plot_title: Optional[str] = None,
    ) -> OptimizationResult:
        """
        Perform backlash-safe XY scan, localize emitter, optionally move to it,
        save result, and optionally plot the scan colormap.

        Parameters
        ----------
        half_range:
            Half scan range around current position.

            Example:
                {0: 0.0015, 1: 0.0015}

        step_size:
            Scan step size.

            Example:
                {0: 0.0005, 1: 0.0005}

        dwell_time:
            Count integration time per scan point.

        localization_method:
            "max" or "centroid".

        move_to_result:
            If True, move stage to accepted optimized position.
            If correction is rejected, stage returns to old position.

        reject_edge_maximum:
            If True, reject correction when brightest pixel is on scan edge.

        min_peak_over_background:
            Minimum required peak/median contrast.

        min_peak_rate:
            Optional absolute minimum peak count rate.

        max_allowed_shift:
            Optional maximum allowed shift from old position.

            Example:
                {0: 0.002, 1: 0.002}

        save_hdf5_path:
            Optional HDF5 file where scan data and metadata are appended.

        hdf5_group_name:
            Optional group name inside HDF5 file.

        plot_result:
            If True, create a colormap plot after optimization.

        show_plot:
            If True, show the plot window.

        plot_block:
            If False, plotting is non-blocking and your measurement script
            continues automatically.

        plot_pause:
            Pause after non-blocking show.

        close_plot_after:
            If True, close the figure after showing/saving.

        save_plot:
            If True, save the plot as PNG.

        save_plot_path:
            Optional explicit PNG path.

        plot_title:
            Optional custom plot title.
        """

        if half_range is None:
            half_range = {
                self.x_axis: 0.0015,
                self.y_axis: 0.0015,
            }

        if step_size is None:
            step_size = {
                self.x_axis: 0.0005,
                self.y_axis: 0.0005,
            }

        if extra_metadata is None:
            extra_metadata = {}

        scan_time_start = time.time()

        old_position = self.get_current_xy()

        self.log("")
        self.log("============================================================")
        self.log("Starting backlash-safe emitter optimization")
        self.log("============================================================")
        self.log(f"Old position: {old_position}")

        initial_rate, initial_std, initial_n = self.measure_rate(
            dwell_time=final_verify_dwell_time,
            statistic=rate_statistic,
        )

        self.log(f"Initial count rate: {initial_rate:.2f}")

        x_positions = self.make_axis_positions(
            center=old_position[self.x_axis],
            half_range=half_range[self.x_axis],
            step=step_size[self.x_axis],
            axis=self.x_axis,
        )

        y_positions = self.make_axis_positions(
            center=old_position[self.y_axis],
            half_range=half_range[self.y_axis],
            step=step_size[self.y_axis],
            axis=self.y_axis,
        )

        counts_map = np.zeros((len(y_positions), len(x_positions)), dtype=float)
        counts_std_map = np.zeros_like(counts_map)
        counts_n_map = np.zeros_like(counts_map)

        self.log(f"X positions: {x_positions}")
        self.log(f"Y positions: {y_positions}")
        self.log(f"Scan shape: {len(y_positions)} x {len(x_positions)}")

        # ---------------------------------------------------------------------
        # Backlash-safe XY scan
        # ---------------------------------------------------------------------

        x_start = float(x_positions[0])
        y_start=  float(y_positions[0])

        for iy, y in enumerate(y_positions):
            self.log("")
            self.log(f"Row {iy + 1}/{len(y_positions)}: Y = {y:.9f}")

            # Approach Y from consistent direction.
            self.backlash_approach_axis(self.y_axis, float(y))

            # Return to start of X row from consistent direction.
            self.backlash_approach_axis(self.x_axis, x_start)

            for ix, x in enumerate(x_positions):
                # First point is already approached backlash-safe.
                # Subsequent points move monotonically in approach direction.
                if ix > 0:
                    self.move_absolute(self.x_axis, float(x))
                    time.sleep(self.settling_time)

                rate, std, n = self.measure_rate(
                    dwell_time=dwell_time,
                    statistic=rate_statistic,
                )

                counts_map[iy, ix] = rate
                counts_std_map[iy, ix] = std
                counts_n_map[iy, ix] = n

                self.log(
                    f"  X {ix + 1}/{len(x_positions)}: "
                    f"x={x:.9f}, rate={rate:.2f}"
                )

        # ---------------------------------------------------------------------
        # Localize maximum
        # ---------------------------------------------------------------------

        estimated_position, peak_info = self.localize_peak(
            x_positions=x_positions,
            y_positions=y_positions,
            counts_map=counts_map,
            method=localization_method,
            centroid_fraction=centroid_fraction,
        )

        peak_rate = peak_info["peak_rate"]
        background_rate = peak_info["background_rate"]
        contrast = peak_info["contrast_peak_over_background"]
        peak_on_edge = peak_info["peak_on_edge"]

        self.log("")
        self.log("Localization result:")
        self.log(f"  Estimated position: {estimated_position}")
        self.log(f"  Peak rate: {peak_rate:.2f}")
        self.log(f"  Background/median rate: {background_rate:.2f}")
        self.log(f"  Peak/background contrast: {contrast:.3f}")
        self.log(f"  Peak on edge: {peak_on_edge}")

        # ---------------------------------------------------------------------
        # Safety decision
        # ---------------------------------------------------------------------

        accepted = True
        rejection_reasons = []

        if reject_edge_maximum and peak_on_edge:
            accepted = False
            rejection_reasons.append("peak_on_edge")

        if contrast < min_peak_over_background:
            accepted = False
            rejection_reasons.append(
                f"contrast_too_low_{contrast:.3f}_below_{min_peak_over_background:.3f}"
            )

        if min_peak_rate is not None and peak_rate < min_peak_rate:
            accepted = False
            rejection_reasons.append(
                f"peak_rate_too_low_{peak_rate:.2f}_below_{min_peak_rate:.2f}"
            )

        if max_allowed_shift is not None:
            for axis in self.axes:
                shift = abs(estimated_position[axis] - old_position[axis])
                if shift > max_allowed_shift[axis]:
                    accepted = False
                    rejection_reasons.append(
                        f"axis_{axis}_shift_too_large_{shift:.9f}_"
                        f"above_{max_allowed_shift[axis]:.9f}"
                    )

        if accepted:
            reason = "accepted"
            new_position = dict(estimated_position)
        else:
            reason = "rejected_" + "__".join(rejection_reasons)
            new_position = dict(old_position)

        # ---------------------------------------------------------------------
        # Move to final accepted position or return to old position
        # ---------------------------------------------------------------------

        final_rate = np.nan
        final_std = np.nan
        final_n = 0

        if move_to_result:
            if accepted:
                self.log("")
                self.log("Moving to optimized position using backlash approach...")
                self.log("First moving to the starting point")
                self.backlash_approach_axis(self.x_axis, x_start)
                self.time.sleep(2)
                self.backlash_approach_axis(self.y_axis, y_start)
                self.time.sleep(2)
                self.backlash_approach_position(new_position)
            else:
                self.log("")
                self.log("Correction rejected. Returning to old position using backlash approach...")
                self.log("First moving to the starting point")
                self.backlash_approach_axis(self.x_axis, x_start)
                self.time.sleep(2)
                self.backlash_approach_axis(self.y_axis, y_start)
                self.time.sleep(2)
                self.backlash_approach_position(old_position)

            if verify_final_position:
                final_rate, final_std, final_n = self.measure_rate(
                    dwell_time=final_verify_dwell_time,
                    statistic=rate_statistic,
                )
                self.log(f"Final verified rate: {final_rate:.2f}")
        else:
            self.log("")
            self.log("move_to_result=False, stage position not changed.")

        scan_time_end = time.time()

        metadata = {
            "half_range": {str(k): float(v) for k, v in half_range.items()},
            "step_size": {str(k): float(v) for k, v in step_size.items()},
            "dwell_time": float(dwell_time),
            "rate_statistic": rate_statistic,
            "settling_time": float(self.settling_time),
            "sample_interval": float(self.sample_interval),
            "approach_direction": {
                str(k): int(v) for k, v in self.approach_direction.items()
            },
            "backlash_overshoot": {
                str(k): float(v) for k, v in self.backlash_overshoot.items()
            },
            "initial_rate_std": float(initial_std),
            "initial_rate_n": int(initial_n),
            "final_rate_std": float(final_std),
            "final_rate_n": int(final_n),
            "reject_edge_maximum": bool(reject_edge_maximum),
            "min_peak_over_background": float(min_peak_over_background),
            "min_peak_rate": None if min_peak_rate is None else float(min_peak_rate),
            "max_allowed_shift": (
                None
                if max_allowed_shift is None
                else {str(k): float(v) for k, v in max_allowed_shift.items()}
            ),
            "extra_metadata": extra_metadata,
        }

        result = OptimizationResult(
            accepted=bool(accepted),
            reason=reason,

            old_position={int(k): float(v) for k, v in old_position.items()},
            new_position={int(k): float(v) for k, v in new_position.items()},
            estimated_position={
                int(k): float(v) for k, v in estimated_position.items()
            },

            initial_rate=float(initial_rate),
            final_rate=float(final_rate),
            peak_rate=float(peak_rate),
            background_rate=float(background_rate),
            contrast_peak_over_background=float(contrast),

            peak_index_y=int(peak_info["peak_index_y"]),
            peak_index_x=int(peak_info["peak_index_x"]),
            peak_on_edge=bool(peak_on_edge),

            x_axis=int(self.x_axis),
            y_axis=int(self.y_axis),

            scan_time_start=float(scan_time_start),
            scan_time_end=float(scan_time_end),

            localization_method=localization_method,

            scan_shape_yx=(int(len(y_positions)), int(len(x_positions))),

            hdf5_path=save_hdf5_path,
            hdf5_group=None,
            plot_path=None,

            metadata=metadata,
        )

        # ---------------------------------------------------------------------
        # Save HDF5 result
        # ---------------------------------------------------------------------

        used_hdf5_group_name = None

        if save_hdf5_path is not None:
            used_hdf5_group_name = self.save_result_to_hdf5(
                file_path=save_hdf5_path,
                result=result,
                x_positions=x_positions,
                y_positions=y_positions,
                counts_map=counts_map,
                counts_std_map=counts_std_map,
                counts_n_map=counts_n_map,
                group_name=hdf5_group_name,
            )

            result.hdf5_group = used_hdf5_group_name

        # ---------------------------------------------------------------------
        # Plot result
        # ---------------------------------------------------------------------

        used_plot_path = None

        if plot_result:
            if save_plot:
                used_plot_path = self._make_default_plot_path(
                    save_plot_path=save_plot_path,
                    save_hdf5_path=save_hdf5_path,
                    hdf5_group_name=used_hdf5_group_name or hdf5_group_name,
                )
            else:
                used_plot_path = None

            self.plot_scan_result(
                x_positions=x_positions,
                y_positions=y_positions,
                counts_map=counts_map,
                result=result,
                save_path=used_plot_path,
                show=show_plot,
                block=plot_block,
                pause=plot_pause,
                close_after=close_plot_after,
                title=plot_title,
            )

            result.plot_path = used_plot_path

            # If HDF5 was saved before plot path existed, update the group attr.
            if save_hdf5_path is not None and used_hdf5_group_name is not None:
                self._update_hdf5_plot_path(
                    file_path=save_hdf5_path,
                    group_name=used_hdf5_group_name,
                    plot_path=used_plot_path,
                )

        self.log("")
        self.log("Emitter optimization finished.")
        self.log(f"Accepted: {result.accepted}")
        self.log(f"Reason: {result.reason}")
        self.log(f"HDF5 group: {result.hdf5_group}")
        self.log(f"Plot path: {result.plot_path}")
        self.log("============================================================")
        self.log("")

        return result

    # -------------------------------------------------------------------------
    # HDF5 saving
    # -------------------------------------------------------------------------

    def save_result_to_hdf5(
        self,
        file_path: str,
        result: OptimizationResult,
        x_positions: np.ndarray,
        y_positions: np.ndarray,
        counts_map: np.ndarray,
        counts_std_map: np.ndarray,
        counts_n_map: np.ndarray,
        group_name: Optional[str] = None,
    ) -> str:
        """
        Save scan and optimization result into an HDF5 file.

        If file exists, group is appended.
        If file does not exist, it is created.

        Returns
        -------
        used_group_name:
            Actual group name used in the HDF5 file.
        """

        if not H5PY_AVAILABLE:
            raise ImportError("h5py is not available, cannot save HDF5 result.")

        if group_name is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            group_name = f"EmitterOptimization_{timestamp}"

        file_dir = os.path.dirname(file_path)
        if file_dir != "":
            os.makedirs(file_dir, exist_ok=True)

        self.log(f"Saving optimization result to: {file_path}")
        self.log(f"Requested HDF5 group: {group_name}")

        with h5py.File(file_path, "a") as h5:
            used_group_name = group_name

            if used_group_name in h5:
                base_name = used_group_name
                idx = 1
                while f"{base_name}_{idx}" in h5:
                    idx += 1
                used_group_name = f"{base_name}_{idx}"

            grp = h5.create_group(used_group_name)

            grp.create_dataset("x_positions", data=x_positions)
            grp.create_dataset("y_positions", data=y_positions)
            grp.create_dataset("counts_map", data=counts_map)
            grp.create_dataset("counts_std_map", data=counts_std_map)
            grp.create_dataset("counts_n_map", data=counts_n_map)

            grp.create_dataset(
                "old_position_xy",
                data=np.array(
                    [
                        result.old_position[result.x_axis],
                        result.old_position[result.y_axis],
                    ],
                    dtype=float,
                ),
            )

            grp.create_dataset(
                "estimated_position_xy",
                data=np.array(
                    [
                        result.estimated_position[result.x_axis],
                        result.estimated_position[result.y_axis],
                    ],
                    dtype=float,
                ),
            )

            grp.create_dataset(
                "new_position_xy",
                data=np.array(
                    [
                        result.new_position[result.x_axis],
                        result.new_position[result.y_axis],
                    ],
                    dtype=float,
                ),
            )

            grp.attrs["x_axis"] = int(result.x_axis)
            grp.attrs["y_axis"] = int(result.y_axis)

            result_dict = asdict(result)

            for key, value in result_dict.items():
                if key == "metadata":
                    grp.attrs["metadata_json"] = json.dumps(value)
                elif isinstance(value, dict):
                    grp.attrs[key + "_json"] = json.dumps(value)
                elif isinstance(value, tuple):
                    grp.attrs[key] = np.array(value)
                elif value is None:
                    grp.attrs[key] = "None"
                elif isinstance(value, (str, int, float, bool, np.integer, np.floating)):
                    grp.attrs[key] = value
                else:
                    try:
                        grp.attrs[key] = value
                    except Exception:
                        grp.attrs[key + "_json"] = json.dumps(str(value))

        self.log(f"Used HDF5 group: {used_group_name}")

        return used_group_name

    def _update_hdf5_plot_path(
        self,
        file_path: str,
        group_name: str,
        plot_path: Optional[str],
    ):
        """
        Update HDF5 group with the plot path after the plot is generated.
        """

        if not H5PY_AVAILABLE:
            return

        try:
            with h5py.File(file_path, "a") as h5:
                if group_name in h5:
                    h5[group_name].attrs["plot_path"] = (
                        "None" if plot_path is None else str(plot_path)
                    )
        except Exception:
            traceback.print_exc()

    # -------------------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------------------

    def plot_scan_result(
        self,
        x_positions: np.ndarray,
        y_positions: np.ndarray,
        counts_map: np.ndarray,
        result: OptimizationResult,
        save_path: Optional[str] = None,
        show: bool = True,
        block: bool = False,
        pause: float = 0.2,
        close_after: bool = False,
        title: Optional[str] = None,
    ):
        """
        Plot XY scan colormap and mark old, estimated, final, and peak-pixel
        positions.

        The function can be non-blocking so the main measurement script keeps
        running automatically.
        """

        if not MATPLOTLIB_AVAILABLE:
            self.log("matplotlib is not available. Skipping plot.")
            return

        x = np.array(x_positions, dtype=float)
        y = np.array(y_positions, dtype=float)
        z = np.array(counts_map, dtype=float)

        X, Y = np.meshgrid(x, y)

        fig, ax = plt.subplots(figsize=(7.5, 6.2))

        pcm = ax.pcolormesh(
            X,
            Y,
            z,
            shading="auto",
            cmap="viridis",
        )

        cbar = fig.colorbar(pcm, ax=ax)
        cbar.set_label("Count rate", fontsize=11)

        old_x = result.old_position[result.x_axis]
        old_y = result.old_position[result.y_axis]

        est_x = result.estimated_position[result.x_axis]
        est_y = result.estimated_position[result.y_axis]

        new_x = result.new_position[result.x_axis]
        new_y = result.new_position[result.y_axis]

        peak_x = x[result.peak_index_x]
        peak_y = y[result.peak_index_y]

        # Old position
        ax.scatter(
            old_x,
            old_y,
            s=130,
            marker="o",
            facecolors="none",
            edgecolors="white",
            linewidths=2.2,
            label="Old position",
            zorder=5,
        )

        # Estimated optimum
        ax.scatter(
            est_x,
            est_y,
            s=120,
            marker="x",
            color="red",
            linewidths=2.5,
            label="Estimated optimum",
            zorder=6,
        )

        # Final/new position
        if result.accepted:
            new_label = "New accepted position"
            new_color = "cyan"
        else:
            new_label = "Final position, rejected → old"
            new_color = "orange"

        ax.scatter(
            new_x,
            new_y,
            s=150,
            marker="+",
            color=new_color,
            linewidths=2.8,
            label=new_label,
            zorder=7,
        )

        # Brightest measured pixel
        ax.scatter(
            peak_x,
            peak_y,
            s=160,
            marker="*",
            color="magenta",
            edgecolors="black",
            linewidths=0.8,
            label="Brightest pixel",
            zorder=8,
        )

        ax.set_xlabel(f"X position, axis {result.x_axis}", fontsize=11)
        ax.set_ylabel(f"Y position, axis {result.y_axis}", fontsize=11)

        if title is None:
            status = "ACCEPTED" if result.accepted else "REJECTED"
            title = (
                f"Emitter optimization: {status}\n"
                f"peak/median = {result.contrast_peak_over_background:.3f}, "
                f"peak = {result.peak_rate:.1f}, "
                f"reason = {result.reason}"
            )

        ax.set_title(title, fontsize=11)

        ax.legend(loc="best", fontsize=9)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.25)

        fig.tight_layout()

        if save_path is not None:
            save_dir = os.path.dirname(save_path)
            if save_dir != "":
                os.makedirs(save_dir, exist_ok=True)

            fig.savefig(save_path, dpi=200, bbox_inches="tight")
            self.log(f"Saved optimization plot to: {save_path}")

        if show:
            plt.show(block=block)

            if not block:
                plt.pause(pause)

        if close_after:
            plt.close(fig)

    # -------------------------------------------------------------------------
    # Plot path helpers
    # -------------------------------------------------------------------------

    def _make_default_plot_path(
        self,
        save_plot_path: Optional[str],
        save_hdf5_path: Optional[str],
        hdf5_group_name: Optional[str],
    ) -> str:
        """
        Create default PNG path when explicit save_plot_path is not provided.
        """

        if save_plot_path is not None:
            return save_plot_path

        timestamp = time.strftime("%Y%m%d_%H%M%S")

        if hdf5_group_name is None:
            group_part = f"EmitterOptimization_{timestamp}"
        else:
            group_part = self._sanitize_filename(hdf5_group_name)

        if save_hdf5_path is not None:
            root, _ = os.path.splitext(save_hdf5_path)
            return f"{root}_{group_part}.png"

        return f"EmitterOptimization_{timestamp}.png"

    @staticmethod
    def _sanitize_filename(text: str) -> str:
        """
        Make a string safe for filenames.
        """

        bad_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|', ' ']

        safe = str(text)

        for c in bad_chars:
            safe = safe.replace(c, "_")

        return safe