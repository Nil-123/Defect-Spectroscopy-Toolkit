# -*- coding: utf-8 -*-
"""
Simplified independent-file HydraHarp analysis script.

Assumption:
Each HDF5 file is one complete measurement/run.
No master file is needed.

Expected datasets per file:
    Data
    Delay
    Rates                  optional
    Positions              optional
    Spectrum               optional
    Wavelengths            optional
    Background counts      optional
    Background_counts      optional

Author: Nilesh Dalla + ChatGPT
"""

import re
import time
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


# ============================================================
# USER SETTINGS
# ============================================================

DATA_FOLDER = Path(
    r"" # Add data path 
)



FILE_GLOB = "*.hdf5"
RECURSIVE = False

# Optional filename filter.
# Example: REQUIRED_FILENAME_TEXT = "Emitter_642"
REQUIRED_FILENAME_TEXT = None

OUTPUT_PREFIX = "selected_analysis"

# Signal/background wavelengths
SIG_WAV = float(str(DATA_FOLDER).split('\\')[-1].split(" ")[-1].replace(",", "."))      # nm
BCK_WAV =  SIG_WAV - 12      # nm

# Spectral integration half-width around signal wavelength
SIG_HALF_WIDTH = 0.5   # nm

# Histogram plotting range
PLOT_XLIM = (0, 400)   # ns
# PLOT_XLIM = None

# Spectrum plotting range
SPECTRUM_XLIM = (SIG_WAV - 12, SIG_WAV + 12)
# SPECTRUM_XLIM = None

# Skip files with very low histogram counts
DROP_EMPTY_HISTOGRAMS = True
MIN_TOTAL_HIST_COUNTS_TO_LOAD = 0

# If spectra wavelength axes differ, interpolate to common wavelength axis
INTERPOLATE_SPECTRA = True
SPECTRUM_COMMON_POINTS = None

# ============================================================
# SELECTION THRESHOLDS
# Set to None if not used.
# ============================================================

MIN_CHANNEL_RATE = None
MIN_SPECTRAL_SIGNAL = None
MIN_HIST_COUNTS = 50
MAX_BACKGROUND_COUNTS = None

# Manual override using global run numbers shown in script, 1-based
MANUAL_KEEP_1BASED = []
MANUAL_DROP_1BASED = []

# Example:
# MIN_CHANNEL_RATE = 8000
# MIN_HIST_COUNTS = 500
# MANUAL_DROP_1BASED = [3, 8]


# ============================================================
# BASIC HELPERS
# ============================================================

def natural_sort_key(path):
    """
    Natural sorting:
    file_2.hdf5 comes before file_10.hdf5.
    """
    text = str(path)
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", text)
    ]


def parse_run_index(path):
    """
    Extract final run index from filenames like:
        sample_35.hdf5

    Returns:
        base_name, run_index

    If no final _number exists:
        stem, None
    """
    path = Path(path)
    stem = path.stem
    match = re.match(r"^(?P<base>.+)_(?P<idx>\d+)$", stem)
    if match:
        return match.group("base"), int(match.group("idx"))
    return stem, None


def read_dataset(f, key):
    """
    Robustly read HDF5 dataset.

    Important:
    Scalar datasets have shape (), for example Background counts.
    For those, f[key][()] is required.
    """
    if key not in f:
        return None

    dset = f[key]

    try:
        if dset.shape == ():
            return np.array(dset[()])
        else:
            return np.array(dset[:])
    except Exception:
        try:
            return np.array(dset[()])
        except Exception as exc:
            print(f"  Warning: could not read {key}: {exc}")
            return None


def read_first_existing_dataset(f, keys):
    """
    Try multiple possible dataset names.
    """
    for key in keys:
        if key in f:
            return read_dataset(f, key), key
    return None, None


def mean_positive(x):
    """
    Mean ignoring zeros and NaNs.
    """
    if x is None:
        return np.nan

    x = np.asarray(x, dtype=float)
    good = np.isfinite(x) & (x > 0)

    if np.any(good):
        return np.mean(x[good])
    return np.nan


# def median_positive(x):
#     """
#     Median ignoring zeros and NaNs.
#     """
#     if x is None:
#         return np.nan

#     x = np.asarray(x, dtype=float)
#     good = np.isfinite(x) & (x > 0)

#     if np.any(good):
#         return np.median(x[good])
#     return np.nan


# ============================================================
# FILE DISCOVERY
# ============================================================

def discover_files(folder):
    """
    Find valid HDF5 files.
    Ignores temporary/incomplete files.
    """
    folder = Path(folder)

    if not folder.exists:
        raise FileNotFoundError(f"DATA_FOLDER does not exist:\n{folder}")

    if RECURSIVE:
        files = list(folder.rglob(FILE_GLOB))
    else:
        files = list(folder.glob(FILE_GLOB))

    files = [
        f for f in files
        if f.is_file()
        and not f.name.endswith(".incomplete")
        and ".incomplete" not in f.name
    ]

    if REQUIRED_FILENAME_TEXT is not None:
        files = [
            f for f in files
            if REQUIRED_FILENAME_TEXT in f.name
        ]

    files = sorted(files, key=natural_sort_key)

    if len(files) == 0:
        raise RuntimeError(
            f"No HDF5 files found in:\n{folder}\n"
            f"Pattern: {FILE_GLOB}"
        )

    print("\n============================================================")
    print("Found HDF5 files")
    print("============================================================")
    for f in files:
        print(f" - {f.name}")
    print("============================================================")

    return files



def build_spectra_matrix(records):
    """
    Combine spectra into one matrix:
        spectra_matrix shape = number_of_runs x number_of_spectrum_points

    If wavelength axes are identical, use directly.
    If wavelength axes differ, optionally interpolate.
    If no wavelength axis exists, use pixel axis.
    """
    valid = [
        i for i, rec in enumerate(records)
        if rec["spectrum"] is not None
    ]

    if len(valid) == 0:
        return None, None

    spectra = [records[i]["spectrum"] for i in valid]
    wavelengths = [records[i]["wavelength"] for i in valid]

    all_have_wavelengths = all(w is not None for w in wavelengths)

    # --------------------------------------------------------
    # Case A: all spectra have wavelength axes
    # --------------------------------------------------------
    if all_have_wavelengths:
        first_w = wavelengths[0]

        same_axis = True
        for w in wavelengths[1:]:
            if len(w) != len(first_w):
                same_axis = False
                break
            if not np.allclose(w, first_w, rtol=0, atol=1e-6):
                same_axis = False
                break

        if same_axis:
            common_x = first_w
            matrix = np.full((len(records), len(common_x)), np.nan)

            for i in valid:
                matrix[i, :] = records[i]["spectrum"]

            return common_x, matrix

        if INTERPOLATE_SPECTRA:
            min_common = max(np.nanmin(w) for w in wavelengths)
            max_common = min(np.nanmax(w) for w in wavelengths)

            if max_common <= min_common:
                print("Warning: spectra have no common wavelength range. Using pixel axis.")
                return build_pixel_spectra_matrix(records)

            if SPECTRUM_COMMON_POINTS is None:
                n_points = min(len(w) for w in wavelengths)
            else:
                n_points = int(SPECTRUM_COMMON_POINTS)

            common_x = np.linspace(min_common, max_common, n_points)
            matrix = np.full((len(records), n_points), np.nan)

            for i in valid:
                w = np.asarray(records[i]["wavelength"], dtype=float)
                s = np.asarray(records[i]["spectrum"], dtype=float)

                order = np.argsort(w)
                w = w[order]
                s = s[order]

                matrix[i, :] = np.interp(common_x, w, s)

            print("Spectra wavelength axes differed. Interpolated to common axis.")
            return common_x, matrix

        else:
            print("Spectra wavelength axes differed. Using pixel axis.")
            return build_pixel_spectra_matrix(records)

    # --------------------------------------------------------
    # Case B: wavelength axis missing for at least one spectrum
    # --------------------------------------------------------
    return build_pixel_spectra_matrix(records)


def build_pixel_spectra_matrix(records):
    """
    Combine spectra using pixel index.
    """
    valid = [
        i for i, rec in enumerate(records)
        if rec["spectrum"] is not None
    ]

    if len(valid) == 0:
        return None, None

    lengths = [len(records[i]["spectrum"]) for i in valid]
    min_len = min(lengths)

    if len(set(lengths)) > 1:
        print(f"Spectra have different lengths. Interpolating/truncating to {min_len} pixels.")

    x = np.arange(min_len)
    matrix = np.full((len(records), min_len), np.nan)

    for i in valid:
        spec = np.asarray(records[i]["spectrum"], dtype=float)

        if len(spec) == min_len:
            matrix[i, :] = spec
        else:
            old_x = np.linspace(0, min_len - 1, len(spec))
            matrix[i, :] = np.interp(x, old_x, spec)

    return x, matrix


# ============================================================
# LOADING SINGLE FILE
# ============================================================

def load_one_file(path):
    """
    Load one independent HDF5 run file.
    """
    path = Path(path)
    base_name, run_index = parse_run_index(path)

    try:
        with h5py.File(path, "r") as f:

            print(f"\nInspecting {path.name}")
            for key in f.keys():
                try:
                    print(f" - {key}: {f[key].shape}")
                except Exception:
                    print(f" - {key}")

            if "Data" not in f or "Delay" not in f:
                print("  Skipping: missing Data or Delay.")
                return None

            data = read_dataset(f, "Data")
            delay = read_dataset(f, "Delay")

            if data is None or delay is None:
                print("  Skipping: could not read Data or Delay.")
                return None

            data = np.asarray(data, dtype=float)
            delay = np.asarray(delay, dtype=float)

            if data.ndim != 1:
                print(f"  Skipping: Data is not 1D. Shape = {data.shape}")
                return None

            if delay.ndim != 1:
                print(f"  Skipping: Delay is not 1D. Shape = {delay.shape}")
                return None

            if len(data) != len(delay):
                print(
                    f"  Skipping: Data and Delay lengths differ. "
                    f"{len(data)} vs {len(delay)}"
                )
                return None

            total_counts = np.nansum(data)

            if DROP_EMPTY_HISTOGRAMS and total_counts <= MIN_TOTAL_HIST_COUNTS_TO_LOAD:
                print(f"  Skipping empty histogram. Total counts = {total_counts:.0f}")
                return None

            rates = read_dataset(f, "Rates")
            positions = read_dataset(f, "Positions")

            spectrum = read_dataset(f, "Spectrum")
            wavelength = read_dataset(f, "Wavelengths")

            # wavelength, spectrum = parse_spectrum(
            #     spectrum_raw=spectrum_raw,
            #     wavelengths_raw=wavelengths_raw,
            # )

            background_raw, background_key = read_first_existing_dataset(
                f,
                ["Background counts", "Background_counts"],
            )

            background_counts = np.nan

            if background_raw is not None:
                b = np.asarray(background_raw)

                if b.shape == ():
                    background_counts = float(b)

                elif b.size == 1:
                    background_counts = float(b.ravel()[0])

                else:
                    # If it is an array, store integrated/summed value.
                    # Usually background count is expected to be scalar.
                    background_counts = float(np.nansum(b))

            record = {
                "data": data,
                "delay": delay,
                "rates": rates,
                "positions": positions,
                "wavelength": wavelength,
                "spectrum": spectrum,
                "background_counts": background_counts,
                "source_file": path.name,
                "source_path": str(path),
                "source_group": base_name,
                "run_index_0based": run_index if run_index is not None else np.nan,
                "run_number_1based": run_index + 1 if run_index is not None else np.nan,
            }

            print(f"  Loaded histogram. Total counts = {total_counts:.0f}")

            return record

    except OSError as exc:
        print(f"  Skipping {path.name}: HDF5 open error: {exc}")
        return None

    except Exception as exc:
        print(f"  Skipping {path.name}: error while reading: {exc}")
        return None


# ============================================================
# LOAD ALL FILES
# ============================================================

def load_all_files(folder):
    """
    Load all independent HDF5 files from folder.
    """
    files = discover_files(folder)

    records = []

    for path in files:
        rec = load_one_file(path)

        if rec is not None:
            records.append(rec)

    if len(records) == 0:
        raise RuntimeError("No valid HDF5 run files were loaded.")

    return combine_records(records)


def combine_records(records):
    """
    Combine records into one filedata dictionary.
    """
    ref_delay = records[0]["delay"]

    compatible = []

    for rec in records:
        delay = rec["delay"]

        if len(delay) != len(ref_delay):
            print(
                f"Skipping {rec['source_file']}: "
                f"delay length differs: {len(delay)} vs {len(ref_delay)}"
            )
            continue

        if not np.allclose(delay, ref_delay, rtol=0, atol=1e-12):
            print(f"Skipping {rec['source_file']}: delay axis differs.")
            continue

        compatible.append(rec)

    if len(compatible) == 0:
        raise RuntimeError("No files have compatible Delay axes.")

    records = compatible

    data = np.vstack([rec["data"] for rec in records])
    wavelengths, spectra = build_spectra_matrix(records)

    filedata = {
        "data": data,
        "delay": ref_delay,
        "rates": [rec["rates"] for rec in records],
        "positions": [rec["positions"] for rec in records],
        "wavelengths": wavelengths,
        "spectra": spectra,
        "background_counts": np.array(
            [rec["background_counts"] for rec in records],
            dtype=float,
        ),
        "source_file": np.array([rec["source_file"] for rec in records], dtype=object),
        "source_path": np.array([rec["source_path"] for rec in records], dtype=object),
        "source_group": np.array([rec["source_group"] for rec in records], dtype=object),
        "run_index_0based": np.array([rec["run_index_0based"] for rec in records], dtype=float),
        "run_number_1based": np.array([rec["run_number_1based"] for rec in records], dtype=float),
    }

    print("\n============================================================")
    print("Combined loaded data")
    print("============================================================")
    print(f"Number of loaded runs: {data.shape[0]}")
    print(f"Histogram matrix shape: {data.shape}")
    print(f"Delay length: {len(ref_delay)}")

    if spectra is not None:
        print(f"Spectra matrix shape: {spectra.shape}")
        print(f"Spectrum/WL axis length: {len(wavelengths)}")
    else:
        print("No spectra loaded.")

    print("============================================================")

    return filedata


# ============================================================
# METRICS
# ============================================================

def extract_rate_metrics(rates):
    """
    Extract sync/channel rate metrics.

    Expected format:
        Rates[0] = elapsed time
        Rates[1] = sync rate
        Rates[2] = channel rate

    Also supports transposed format:
        Rates[:, 0] = elapsed time
        Rates[:, 1] = sync rate
        Rates[:, 2] = channel rate
    """
    result = {
        "elapsed_time_start": np.nan,
        "elapsed_time_end": np.nan,
        "sync_rate_mean": np.nan,
        "channel_rate_mean": np.nan,
        "channel_rate_median": np.nan,
    }

    if rates is None:
        return result

    r = np.asarray(rates, dtype=float)

    if r.ndim != 2:
        return result

    if r.shape[0] >= 3:
        elapsed = r[0]
        sync = r[1]
        channel = r[2]

    elif r.shape[1] >= 3:
        elapsed = r[:, 0]
        sync = r[:, 1]
        channel = r[:, 2]

    else:
        return result

    elapsed_good = elapsed[np.isfinite(elapsed)]

    if len(elapsed_good) > 0:
        result["elapsed_time_start"] = np.nanmin(elapsed_good)
        result["elapsed_time_end"] = np.nanmax(elapsed_good)

    result["sync_rate_mean"] = mean_positive(sync)
    result["channel_rate_mean"] = mean_positive(channel)
    #result["channel_rate_median"] = median_positive(channel)

    return result


def calculate_metrics(filedata):
    """
    Calculate per-run quality metrics.
    """
    data = filedata["data"]
    delay = filedata["delay"]
    rates = filedata["rates"]
    wavelengths = filedata["wavelengths"]
    spectra = filedata["spectra"]

    n_runs = data.shape[0]

    if PLOT_XLIM is None:
        hist_mask = np.ones_like(delay, dtype=bool)
    else:
        hist_mask = (delay >= PLOT_XLIM[0]) & (delay <= PLOT_XLIM[1])

    rows = []

    for i in range(n_runs):
        row = {}

        row["global_run_1based"] = i + 1
        row["source_file"] = filedata["source_file"][i]
        row["source_path"] = filedata["source_path"][i]
        row["source_group"] = filedata["source_group"][i]
        row["file_run_index_0based"] = filedata["run_index_0based"][i]
        row["file_run_number_1based"] = filedata["run_number_1based"][i]

        row["hist_total_counts"] = np.nansum(data[i])
        row["hist_window_counts"] = np.nansum(data[i, hist_mask])
        row["hist_max_bin"] = np.nanmax(data[i])

        rate_info = extract_rate_metrics(rates[i])
        row.update(rate_info)

        # -----------------------------
        # Spectrum metrics
        # -----------------------------
        row["spectral_peak_max"] = np.nan
        row["spectral_peak_area"] = np.nan
        row["spectral_local_background"] = np.nan
        row["spectral_signal_bg_sub"] = np.nan

        if spectra is not None and wavelengths is not None:
            spec = spectra[i]

            if spec is not None and np.any(np.isfinite(spec)):
                wl = wavelengths

                signal_mask = np.abs(wl - SIG_WAV) <= SIG_HALF_WIDTH

                background_mask = (
                    ((wl >= SIG_WAV - 4.0) & (wl <= SIG_WAV - 2.0))
                    |
                    ((wl >= SIG_WAV + 2.0) & (wl <= SIG_WAV + 4.0))
                )

                if np.any(signal_mask):
                    peak_area = np.nansum(spec[signal_mask])
                    peak_max = np.nanmax(spec[signal_mask])
                else:
                    peak_area = np.nan
                    peak_max = np.nan

                if np.any(background_mask):
                    local_bg = np.nanmedian(spec[background_mask])
                    bg_sub = peak_area - local_bg * np.sum(signal_mask)
                else:
                    local_bg = np.nan
                    bg_sub = np.nan

                row["spectral_peak_max"] = peak_max
                row["spectral_peak_area"] = peak_area
                row["spectral_local_background"] = local_bg
                row["spectral_signal_bg_sub"] = bg_sub

        row["background_counts"] = filedata["background_counts"][i]

        rows.append(row)

    metrics = pd.DataFrame(rows)

    return metrics


# ============================================================
# SELECTION
# ============================================================

def apply_thresholds(metrics):
    """
    Initial automatic keep/drop selection.
    """
    good = np.ones(len(metrics), dtype=bool)

    if MIN_CHANNEL_RATE is not None:
        good &= metrics["channel_rate_mean"].to_numpy() >= MIN_CHANNEL_RATE

    if MIN_SPECTRAL_SIGNAL is not None:
        good &= metrics["spectral_signal_bg_sub"].to_numpy() >= MIN_SPECTRAL_SIGNAL

    if MIN_HIST_COUNTS is not None:
        good &= metrics["hist_window_counts"].to_numpy() >= MIN_HIST_COUNTS

    if MAX_BACKGROUND_COUNTS is not None:
        good &= metrics["background_counts"].to_numpy() <= MAX_BACKGROUND_COUNTS

    if len(MANUAL_KEEP_1BASED) > 0:
        manual = np.zeros(len(metrics), dtype=bool)

        for run in MANUAL_KEEP_1BASED:
            idx = run - 1
            if 0 <= idx < len(metrics):
                manual[idx] = True
            else:
                print(f"Warning: MANUAL_KEEP_1BASED contains invalid run {run}")

        good = manual

    for run in MANUAL_DROP_1BASED:
        idx = run - 1
        if 0 <= idx < len(metrics):
            good[idx] = False
        else:
            print(f"Warning: MANUAL_DROP_1BASED contains invalid run {run}")

    return good


# ============================================================
# PLOTTING
# ============================================================

def add_group_boundaries(axs, metrics):
    """
    Add dashed lines when source group changes.
    Useful if several base filenames are in one folder.
    """
    if not isinstance(axs, (list, tuple, np.ndarray)):
        axs = [axs]

    groups = metrics["source_group"].astype(str).to_numpy()

    boundaries = []

    for i in range(1, len(groups)):
        if groups[i] != groups[i - 1]:
            boundaries.append(i + 0.5)

    for ax in axs:
        for b in boundaries:
            ax.axvline(
                b,
                color="black",
                ls="--",
                lw=0.8,
                alpha=0.35,
            )


def plot_overview(metrics):
    """
    Static overview plots.
    """
    x = metrics["global_run_1based"]

    fig, axs = plt.subplots(
        4,
        1,
        figsize=(12, 10),
        sharex=True,
    )

    axs[0].plot(
        x,
        metrics["hist_window_counts"],
        "o-",
        color="tab:purple",
    )
    axs[0].set_ylabel("Histogram counts\nin window")
    axs[0].grid(True, alpha=0.35)

    axs[1].plot(
        x,
        metrics["channel_rate_mean"],
        "o-",
        color="tab:blue",
    )
    axs[1].set_ylabel("Mean channel rate")
    axs[1].grid(True, alpha=0.35)

    axs[2].plot(
        x,
        metrics["sync_rate_mean"],
        "o-",
        color="tab:green",
    )
    axs[2].set_ylabel("Sync rate mean")
    axs[2].grid(True, alpha=0.35)

    axs[3].plot(
        x,
        metrics["background_counts"],
        "o-",
        color="tab:red",
    )
    axs[3].set_ylabel("Background counts")
    axs[3].set_xlabel("Global run number")
    axs[3].grid(True, alpha=0.35)

    add_group_boundaries(axs, metrics)

    fig.suptitle("Run quality overview")
    plt.tight_layout()
    plt.show()


def summed_histogram(filedata, good_mask):
    """
    Sum selected histograms.
    """
    data = filedata["data"]

    selected = np.where(good_mask)[0]

    if len(selected) == 0:
        return np.zeros(data.shape[1], dtype=float)

    return np.nansum(data[selected, :], axis=0)


def get_spectrum_axis(filedata):
    """
    Return spectrum x-axis and label.
    """
    spectra = filedata["spectra"]
    wavelengths = filedata["wavelengths"]

    if spectra is None:
        return None, None

    if wavelengths is None:
        return np.arange(spectra.shape[1]), "Spectrometer pixel"

    return wavelengths, "Wavelength / nm"


def format_selected_runs(good_mask, max_show=25):
    """
    Format selected run list for titles.
    """
    selected = (np.where(good_mask)[0] + 1).tolist()

    if len(selected) == 0:
        return "None"

    if len(selected) <= max_show:
        return str(selected)

    return f"{selected[:max_show]} ... + {len(selected) - max_show} more"


def describe_run(metrics, idx):
    """
    Description for clicked run.
    """
    row = metrics.iloc[idx]

    return (
        f"global run #{int(row['global_run_1based'])}, "
        f"file={row['source_file']}"
    )


# ============================================================
# INTERACTIVE TOGGLE
# ============================================================

def interactive_selector(metrics, initial_good, filedata):
    """
    Interactive keep/drop selector.

    Click on a point in the top count plot:
        green = keep
        red   = drop

    Live updates:
        - selected/dropped color
        - summed histogram
        - clicked run spectrum
        - average selected spectrum
    """
    data = filedata["data"]
    delay = filedata["delay"]
    spectra = filedata["spectra"]

    good = initial_good.copy()

    x = metrics["global_run_1based"].to_numpy()
    y = metrics["hist_window_counts"].to_numpy()
    y_label = "Histogram counts in window"

    n_runs = len(good)

    has_spectra = spectra is not None

    if has_spectra:
        fig, axs = plt.subplots(
            3,
            1,
            figsize=(13, 11),
            gridspec_kw={"height_ratios": [1.1, 1.4, 1.4]},
        )
        ax_count, ax_hist, ax_spec = axs
    else:
        fig, axs = plt.subplots(
            2,
            1,
            figsize=(13, 8),
            gridspec_kw={"height_ratios": [1.1, 1.5]},
        )
        ax_count, ax_hist = axs
        ax_spec = None

    # --------------------------------------------------------
    # Panel 1: count plot / toggle panel
    # --------------------------------------------------------
    colors = np.where(good, "tab:green", "tab:red")

    scatter = ax_count.scatter(
        x,
        y,
        c=colors,
        s=85,
        edgecolor="black",
        linewidth=0.5,
        zorder=3,
    )

    ax_count.plot(
        x,
        y,
        color="gray",
        alpha=0.4,
        zorder=1,
    )

    ax_count.set_xlabel("Global run number")
    ax_count.set_ylabel(y_label)
    ax_count.set_title(
        "Click points to toggle keep/drop\n"
        "Green = kept, red = dropped"
    )
    ax_count.grid(True, alpha=0.35)

    add_group_boundaries(ax_count, metrics)

    # --------------------------------------------------------
    # Panel 2: summed histogram
    # --------------------------------------------------------
    hist_sum = summed_histogram(filedata, good)

    hist_line, = ax_hist.plot(
        delay,
        hist_sum,
        color="black",
        lw=1.2,
    )

    ax_hist.set_xlabel("Delay / ns")
    ax_hist.set_ylabel("Selected summed histogram")
    ax_hist.grid(True, alpha=0.35)

    if PLOT_XLIM is not None:
        ax_hist.set_xlim(*PLOT_XLIM)

    # --------------------------------------------------------
    # Panel 3: spectrum
    # --------------------------------------------------------
    current_idx = 0

    if has_spectra:
        spec_x, spec_xlabel = get_spectrum_axis(filedata)

        current_spec = spectra[current_idx]

        if np.any(good):
            avg_spec = np.nanmean(spectra[good, :], axis=0)
        else:
            avg_spec = np.zeros_like(current_spec)

        current_spec_line, = ax_spec.plot(
            spec_x,
            current_spec,
            color="tab:blue",
            lw=1.0,
            label="Clicked run",
        )

        avg_spec_line, = ax_spec.plot(
            spec_x,
            avg_spec,
            color="black",
            lw=1.5,
            alpha=0.85,
            label="Average selected",
        )

        if spec_xlabel.startswith("Wavelength"):
            ax_spec.axvline(
                SIG_WAV,
                color="tab:red",
                ls="--",
                lw=1.1,
                label=f"Signal {SIG_WAV} nm",
            )
            ax_spec.axvline(
                BCK_WAV,
                color="tab:orange",
                ls=":",
                lw=1.1,
                label=f"Background {BCK_WAV} nm",
            )

            if SPECTRUM_XLIM is not None:
                ax_spec.set_xlim(*SPECTRUM_XLIM)

        ax_spec.set_xlabel(spec_xlabel)
        ax_spec.set_ylabel("Spectral counts")
        ax_spec.grid(True, alpha=0.35)
        ax_spec.legend(loc="best")

    # --------------------------------------------------------
    # Update function
    # --------------------------------------------------------
    def update(clicked_idx=None):
        nonlocal current_idx

        if clicked_idx is not None:
            current_idx = clicked_idx

        colors = np.where(good, "tab:green", "tab:red")
        scatter.set_color(colors)

        new_hist = summed_histogram(filedata, good)
        hist_line.set_ydata(new_hist)

        n_selected = int(np.sum(good))

        ax_hist.set_title(
            f"Summed histogram from {n_selected} selected run(s)\n"
            f"Selected runs: {format_selected_runs(good)}"
        )

        ymax = np.nanmax(new_hist)

        if not np.isfinite(ymax) or ymax <= 0:
            ymax = 1

        ax_hist.set_ylim(0, ymax * 1.08)

        if PLOT_XLIM is not None:
            ax_hist.set_xlim(*PLOT_XLIM)

        if has_spectra:
            spec_x, spec_xlabel = get_spectrum_axis(filedata)

            current_spec = spectra[current_idx]

            if np.any(good):
                avg_spec = np.nanmean(spectra[good, :], axis=0)
            else:
                avg_spec = np.zeros_like(current_spec)

            current_spec_line.set_xdata(spec_x)
            current_spec_line.set_ydata(current_spec)
            current_spec_line.set_label(f"Clicked run {current_idx + 1}")

            avg_spec_line.set_xdata(spec_x)
            avg_spec_line.set_ydata(avg_spec)

            ax_spec.set_title(
                f"Spectrum of {describe_run(metrics, current_idx)}\n"
                f"Status: {'KEEP' if good[current_idx] else 'DROP'}"
            )

            ax_spec.relim()
            ax_spec.autoscale_view()

            if spec_xlabel.startswith("Wavelength") and SPECTRUM_XLIM is not None:
                ax_spec.set_xlim(*SPECTRUM_XLIM)

            ax_spec.legend(loc="best")

        fig.canvas.draw_idle()

    update(0)

    # --------------------------------------------------------
    # Mouse click callback
    # --------------------------------------------------------
    def onclick(event):
        if event.inaxes != ax_count:
            return

        if event.xdata is None:
            return

        idx = int(np.argmin(np.abs(x - event.xdata)))

        if idx < 0 or idx >= n_runs:
            return

        good[idx] = ~good[idx]

        print(
            f"{describe_run(metrics, idx)}: "
            f"{'KEEP' if good[idx] else 'DROP'}"
        )

        update(idx)

    fig.canvas.mpl_connect("button_press_event", onclick)

    plt.tight_layout()
    plt.show()

    return good


# ============================================================
# SPECTRUM / WL EVOLUTION PLOT
# ============================================================

def plot_spectrum_evolution(filedata, good_mask=None):
    """
    Static WL/spectrum evolution plot.

    No interactive toggle here.
    It simply shows all loaded spectra.
    """
    spectra = filedata["spectra"]

    if spectra is None:
        print("\nNo spectra found. Skipping spectrum evolution plot.")
        return

    spectra = np.asarray(spectra, dtype=float)
    n_runs = spectra.shape[0]

    spec_x, spec_xlabel = get_spectrum_axis(filedata)

    if spec_x is None:
        print("\nNo valid spectrum axis. Skipping spectrum evolution plot.")
        return

    if SPECTRUM_XLIM is not None and spec_xlabel.startswith("Wavelength"):
        mask = (spec_x >= SPECTRUM_XLIM[0]) & (spec_x <= SPECTRUM_XLIM[1])

        if not np.any(mask):
            print("Requested spectrum range contains no data. Showing full spectrum.")
            mask = np.ones_like(spec_x, dtype=bool)
    else:
        mask = np.ones_like(spec_x, dtype=bool)

    x_plot = spec_x[mask]
    spectra_plot = spectra[:, mask]

    finite = spectra_plot[np.isfinite(spectra_plot)]

    if len(finite) > 0:
        vmin = np.nanpercentile(finite, 2)
        vmax = np.nanpercentile(finite, 98)
    else:
        vmin = None
        vmax = None

    fig, axs = plt.subplots(
        1,
        1,
        figsize=(13, 10),
        gridspec_kw={"height_ratios": [1.2]},
    )

    ax_map = axs  #, ax_waterfall

    # --------------------------------------------------------
    # WL/spectrum colormap
    # --------------------------------------------------------
    im = ax_map.imshow(
        spectra_plot,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[
            x_plot[0],
            x_plot[-1],
            0.5,
            n_runs + 0.5,
        ],
        vmin=vmin,
        vmax=vmax,
    )

    ax_map.set_xlabel(spec_xlabel)
    ax_map.set_ylabel("Global run number")
    ax_map.set_title("Spectrum / wavelength evolution")

    cbar = fig.colorbar(im, ax=ax_map)
    cbar.set_label("Spectral counts")

    if spec_xlabel.startswith("Wavelength"):
        ax_map.axvline(SIG_WAV, color="red", ls="--", lw=1.0)
        #ax_map.axvline(BCK_WAV, color="orange", ls=":", lw=1.0)

    # --------------------------------------------------------
    # Waterfall
    # --------------------------------------------------------
    # max_amp = np.nanmax(spectra_plot)

    # if not np.isfinite(max_amp) or max_amp <= 0:
    #     max_amp = 1.0

    # offset = max_amp * 1.15

    # for i in range(n_runs):
    #     if good_mask is None:
    #         color = "tab:blue"
    #         alpha = 0.75
    #     else:
    #         color = "tab:green" if good_mask[i] else "tab:red"
    #         alpha = 0.85 if good_mask[i] else 0.30

    #     ax_waterfall.plot(
    #         x_plot,
    #         spectra_plot[i] + i * offset,
    #         color=color,
    #         lw=0.8,
    #         alpha=alpha,
    #     )

    #     if n_runs <= 80:
    #         ax_waterfall.text(
    #             x_plot[-1],
    #             i * offset,
    #             str(i + 1),
    #             fontsize=7,
    #             va="center",
    #             ha="left",
    #             color=color,
    #         )

    # ax_waterfall.set_xlabel(spec_xlabel)
    # ax_waterfall.set_ylabel("Run number, vertically offset")

    # if good_mask is None:
    #     ax_waterfall.set_title("Waterfall spectra")
    # else:
    #     ax_waterfall.set_title("Waterfall spectra: green = selected, red = rejected")

    # ax_waterfall.grid(True, alpha=0.25)

    # if spec_xlabel.startswith("Wavelength"):
    #     ax_waterfall.axvline(SIG_WAV, color="red", ls="--", lw=1.0)
    #     ax_waterfall.axvline(BCK_WAV, color="orange", ls=":", lw=1.0)

    plt.tight_layout()
    plt.show()


# ============================================================
# FINAL HISTOGRAM PLOT
# ============================================================

# def plot_final_histogram(filedata, good_mask):
#     """
#     Plot selected summed histogram.
#     """
#     selected = np.where(good_mask)[0]

#     if len(selected) == 0:
#         raise RuntimeError("No runs selected. Cannot make final histogram.")

#     delay = filedata["delay"]
#     hist_sum = np.nansum(filedata["data"][selected, :], axis=0)

#     plt.figure(figsize=(11, 5))
#     plt.plot(
#         delay,
#         hist_sum,
#         color="black",
#         lw=1.2,
#     )

#     plt.xlabel("Delay / ns")
#     plt.ylabel("Added histogram counts")
#     plt.title(
#         f"Final summed histogram from {len(selected)} selected run(s)\n"
#         f"Selected runs: {(selected + 1).tolist()}"
#     )
#     plt.grid(True, alpha=0.35)

#     if PLOT_XLIM is not None:
#         plt.xlim(*PLOT_XLIM)

#     plt.tight_layout()
#     plt.show()

#     return selected, hist_sum


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(output_folder, metrics, selected_indices, filedata, summed_hist):
    """
    Save analysis output.
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base = output_folder / f"{OUTPUT_PREFIX}_{timestamp}"

    metrics_csv = base.with_name(base.name + "_metrics.csv")
    selected_csv = base.with_name(base.name + "_selected_runs.csv")
    selected_txt = base.with_name(base.name + "_selected_run_numbers.txt")
    selected_files_txt = base.with_name(base.name + "_selected_source_files.txt")
    histogram_npz = base.with_name(base.name + "_summed_histogram.npz")

    metrics.to_csv(metrics_csv, index=False)

    selected_table = metrics.iloc[selected_indices].copy()
    selected_table.to_csv(selected_csv, index=False)

    selected_runs_1based = selected_indices + 1

    np.savetxt(
        selected_txt,
        selected_runs_1based,
        fmt="%d",
    )

    selected_paths = metrics.iloc[selected_indices]["source_path"].astype(str).tolist()

    with open(selected_files_txt, "w", encoding="utf-8") as f:
        for p in selected_paths:
            f.write(p + "\n")

    np.savez(
        histogram_npz,
        delay_ns=filedata["delay"],
        summed_histogram=summed_hist,
        selected_indices_0based=selected_indices,
        selected_runs_1based=selected_runs_1based,
        selected_source_files=filedata["source_file"][selected_indices],
        selected_source_paths=filedata["source_path"][selected_indices],
    )

    print("\n============================================================")
    print("Saved results")
    print("============================================================")
    print(f"Metrics CSV:               {metrics_csv}")
    print(f"Selected runs CSV:         {selected_csv}")
    print(f"Selected run numbers TXT:  {selected_txt}")
    print(f"Selected source files TXT: {selected_files_txt}")
    print(f"Summed histogram NPZ:      {histogram_npz}")
    print("============================================================")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # 1. Load all independent HDF5 files
    # --------------------------------------------------------
    filedata = load_all_files(DATA_FOLDER)

    # --------------------------------------------------------
    # 2. Calculate metrics
    # --------------------------------------------------------
    metrics = calculate_metrics(filedata)

    print("\n============================================================")
    print("Loaded run metrics")
    print("============================================================")
    print(metrics)

    # --------------------------------------------------------
    # 3. Static overview plots
    # --------------------------------------------------------
    plot_overview(metrics)

    # --------------------------------------------------------
    # 4. Initial threshold selection
    # --------------------------------------------------------
    good_mask = apply_thresholds(metrics)

    print("\nInitial selected runs:")
    print((np.where(good_mask)[0] + 1).tolist())

    cols_to_print = [
        "global_run_1based",
        "source_file",
        "hist_total_counts",
        "hist_window_counts",
        "channel_rate_mean",
        "spectral_signal_bg_sub",
        "background_counts",
    ]

    cols_to_print = [
        c for c in cols_to_print
        if c in metrics.columns
    ]

    print("\nInitial selected source files:")
    print(metrics.loc[good_mask, cols_to_print])

    # --------------------------------------------------------
    # 5. Interactive keep/drop toggle
    # --------------------------------------------------------
    good_mask = interactive_selector(
        metrics=metrics,
        initial_good=good_mask,
        filedata=filedata,
    )

    # --------------------------------------------------------
    # 6. Final selected runs
    # --------------------------------------------------------
    selected_indices = np.where(good_mask)[0]

    print("\nFinal selected runs:")
    print((selected_indices + 1).tolist())

    print("\nFinal selected source files:")
    print(metrics.loc[good_mask, cols_to_print])

    # --------------------------------------------------------
    # 7. WL/spectrum evolution plot
    # --------------------------------------------------------
    plot_spectrum_evolution(
        filedata=filedata,
        good_mask=good_mask,
    )

    # --------------------------------------------------------
    # 8. Final summed histogram
    # --------------------------------------------------------
    # selected_indices, summed_hist = plot_final_histogram(
    #     filedata=filedata,
    #     good_mask=good_mask,
    # )

    # --------------------------------------------------------
    # 9. Save results
    # --------------------------------------------------------
    # save_results(
    #     output_folder=DATA_FOLDER,
    #     metrics=metrics,
    #     selected_indices=selected_indices,
    #     filedata=filedata,
    #     summed_hist=summed_hist,
    # )