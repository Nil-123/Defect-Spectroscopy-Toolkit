from pathlib import Path
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# =========================
# User controls 
# =========================

PROJECT_ROOT = Path.cwd()

EXAMPLE_DATA = PROJECT_ROOT / "examples" / "pl_mapping"

EXAMPLE_DATA = (
    PROJECT_ROOT
    / "Examples"
    / "PL_mapping"
)

path = EXAMPLE_DATA / "example_spectra.dat"
x_input = EXAMPLE_DATA / "example_X.txt"
y_input = EXAMPLE_DATA / "example_Y.txt"


# --------------------------------------------------
# USER SETTINGS
# --------------------------------------------------

# Peak detection wavelength window (user-defined)
min_wavelength = 600  # nm
max_wavelength = 690  # nm
vmin=0.5e6
vmax=0.8e6
save_choice = "n"  #"y"
# Peak shape constraints
min_width_nm = 1
max_width_nm = 10
minHeight = 1
min_prominence = 250
# =========================
# Styling controls 
# =========================
tick_fontsize = 18
label_fontsize = 18
title_fontsize = 18
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["xtick.labelsize"] = tick_fontsize
plt.rcParams["ytick.labelsize"] = tick_fontsize
# =========================
# Peak grouping + color 
# =========================
peak_group_tol_nm = 0.5    # nm tolerance for grouping peaks locally
color_cmap = "rainbow"            # try: 'viridis', 'plasma', 'turbo', 'rainbow'
circle_edge_alpha = 0.95

# Circle color scaling (wavelength -> colorbar). Set to numbers to force scale
# or keep None to use auto min/max of group center wavelengths.
circle_vmin_nm = None
circle_vmax_nm = None
# =========================
# Local grouping controls 
# =========================
local_radius_xy = 1       #  with your spacing ~0.94 to ~1.02, 2.0 is a good start
use_local_grouping = True       # set False to revert to global
# If True, we draw only 1 circle per pixel (best peak)
use_best_peak_only = True



# =========================
# Load & slice wavelength window
# =========================


data = np.loadtxt(path, skiprows=1)
input_wavelength = data[:, 0]
spectra_all = data[:, 1:]  # shape: (N_wavelength, N_pixels)

mask = (input_wavelength >= min_wavelength) & (input_wavelength <= max_wavelength)
if not np.any(mask):
    raise ValueError("No wavelength points found in the specified window. Check min/max values.")

wavelengths = input_wavelength[mask]
counts = spectra_all[mask, :]  # (N_window, N_pixels)

# =========================
# Load pixel coordinates
# =========================
x_file_raw = np.loadtxt(x_input)
x_file = x_file_raw*1-np.mean(x_file_raw*1)
y_file_raw = np.loadtxt(y_input)
y_file = y_file_raw*1 - np.mean(y_file_raw*1)
unique_x = np.sort(np.unique(x_file))
unique_y = np.sort(np.unique(y_file))

if len(x_file) != spectra_all.shape[1]:
    raise ValueError(
        "Number of X coordinates does not match the number of spectra."
    )

if len(y_file) != spectra_all.shape[1]:
    raise ValueError(
        "Number of Y coordinates does not match the number of spectra."
    )

# unique_x=(unique_x*1000)-np.mean(unique_x)
# unique_y=(unique_y*1000)-np.mean(unique_y)
x_length = len(unique_x)
y_length = len(unique_y)

print(f"[INFO] unique_x={x_length}, unique_y={y_length}")
print(f"[INFO] spectra columns (pixels) = {counts.shape[1]}")

# Mapping from coordinate pair -> spectrum column index
coord_to_index = {}
for idx in range(len(x_file)):
    coord_to_index[(x_file[idx], y_file[idx])] = idx

# =========================
# Convert min/max width nm -> pixels width
# =========================
nm_step = (wavelengths.max() - wavelengths.min()) / (len(wavelengths) - 1)
min_width = max(1, float(min_width_nm / nm_step))
max_width = max(min_width + 1e-9, float(max_width_nm / nm_step))  # avoid invalid ranges

# =========================
# Overall PL intensity map
# =========================
PL_integral = np.sum(np.abs(counts), axis=0)  # length N_pixels
PL_map = np.zeros((y_length, x_length))
peak_x_map = np.full((y_length, x_length), np.nan)  # peak wavelength
peak_idx_map = np.full((y_length, x_length), -1, dtype=int)  # which spectrum column

# =========================
# Peak detection + store best peak per pixel
# =========================
background = np.mean(counts, axis=1, keepdims=True)  # (N_window, 1)
# background visualization (keep or remove)
# plt.figure()
# plt.plot(wavelengths, background[:, 0])
# plt.title("Background")
# plt.show()

counts_detrended = counts - background  # (N_window, N_pixels)

for j, uy in enumerate(unique_y):
    for i, ux in enumerate(unique_x):
        col_idx = coord_to_index.get((ux, uy), None)
        if col_idx is None:
            continue
        ydata = counts_detrended[:, col_idx]
        peaks, props = find_peaks(
            ydata,
            height=minHeight,
            width=[min_width, max_width],
            prominence=min_prominence
        )
        if len(peaks) == 0:
            continue
        if use_best_peak_only:
            peak_heights = props.get("peak_heights", ydata[peaks])
            best_local = np.argmax(peak_heights)
            best_peak = peaks[best_local]
            peak_wl = wavelengths[best_peak]
            peak_x_map[j, i] = peak_wl
            peak_idx_map[j, i] = col_idx
        else:
            raise NotImplementedError("Multiple peaks per pixel overlay not implemented.")

# =========================
# Build PL_map grid
# =========================
for j, uy in enumerate(unique_y):
    for i, ux in enumerate(unique_x):
        col_idx = coord_to_index.get((ux, uy), None)
        if col_idx is None:
            continue
        PL_map[j, i] = PL_integral[col_idx]

# =========================
# Plot interactive map with circles
# =========================

fig, ax = plt.subplots(figsize=(7, 6), dpi=200)

im = ax.imshow(
    PL_map,
    cmap="viridis",
    origin="lower",
    extent=[unique_x.min(), unique_x.max(), unique_y.min(), unique_y.max()],
    aspect="auto",
    # vmin=vmin,
    # vmax=vmax
)

cbar = plt.colorbar(im, ax=ax)
cbar.set_label("PL integral(a.u.)", fontsize=label_fontsize)
cbar.ax.tick_params(labelsize=tick_fontsize)
ax.set_xlabel("X-coordinate(µm)", fontsize=label_fontsize)
ax.set_ylabel("Y-coordinate(µm)", fontsize=label_fontsize)
ax.set_title(f"PL map for peaks in range ({int(min_wavelength)}–{int(max_wavelength)} nm)",fontsize=title_fontsize)
ax.tick_params(axis='both', labelsize=tick_fontsize)

# =========================
# Prepare circle artists (local grouping + color-coded)
# =========================

circles = []
circle_to_col = {}
# Gather detected peaks (only where valid)
pts = []  # (j, i, x, y, peak_wl, col_idx)
for j, uy in enumerate(unique_y):
    for i, ux in enumerate(unique_x):
        col_idx = peak_idx_map[j, i]
        wl = peak_x_map[j, i]
        if col_idx < 0 or np.isnan(wl):
            continue
        pts.append((j, i, float(ux), float(uy), float(wl), int(col_idx)))

if len(pts) == 0:
    raise RuntimeError("No peaks detected to plot circles.")

pts = np.array(pts, dtype=object)
x_arr = np.array([p[2] for p in pts], dtype=float)
y_arr = np.array([p[3] for p in pts], dtype=float)
wl_arr = np.array([p[4] for p in pts], dtype=float)
col_arr = np.array([p[5] for p in pts], dtype=int)

N = len(pts)

# Neighborhood radius squared
local_r2 = float(local_radius_xy)**2
# -------------------------
# Grouping: local connected components
# -------------------------
group_id = -np.ones(N, dtype=int)

if use_local_grouping:
    current_gid = 0
    for k in range(N):
        if group_id[k] != -1:
            continue
        stack = [k]
        group_id[k] = current_gid

        while stack:
            u = stack.pop()

            dx = x_arr - x_arr[u]
            dy = y_arr - y_arr[u]
            neigh_idx = np.where((dx * dx + dy * dy) <= local_r2)[0]

            for v in neigh_idx:
                if group_id[v] != -1:
                    continue
                if abs(wl_arr[v] - wl_arr[u]) <= peak_group_tol_nm:
                    group_id[v] = current_gid
                    stack.append(v)

        current_gid += 1

    num_groups = current_gid
else:
    # Global grouping: everything bins by rounded wavelength (simple fallback)
    # (You can replace with a better clustering if desired)
    wl_round = np.round(wl_arr / peak_group_tol_nm).astype(int)
    unique_bins, inv = np.unique(wl_round, return_inverse=True)
    group_id = inv
    num_groups = len(unique_bins)

# -------------------------
# Color mapping by group center wavelength
# -------------------------
group_center_wl = np.zeros(num_groups, dtype=float)
for g in range(num_groups):
    group_center_wl[g] = np.median(wl_arr[group_id == g])

if circle_vmin_nm is None:
    circle_vmin = float(np.min(group_center_wl))
else:
    circle_vmin = float(circle_vmin_nm)

if circle_vmax_nm is None:
    circle_vmax = float(np.max(group_center_wl))
else:
    circle_vmax = float(circle_vmax_nm)

if circle_vmax == circle_vmin:
    circle_vmax = circle_vmin + 1e-9

norm = plt.Normalize(vmin=circle_vmin, vmax=circle_vmax)
cmap = plt.get_cmap(color_cmap)

def color_for_gid(gid: int):
    return cmap(norm(group_center_wl[gid]))

# Create circles
radius = 0.03 * (unique_x.max() - unique_x.min() if unique_x.max() != unique_x.min() else 1)

# Reverse lookup: from circle artist -> spectrum column index
for idx in range(N):
    j, i = int(pts[idx][0]), int(pts[idx][1])
    ux, uy = float(pts[idx][2]), float(pts[idx][3])
    col_idx = int(pts[idx][5])
    g_id = int(group_id[idx])

    col = color_for_gid(g_id)

    circ = plt.Circle(
        (ux, uy),
        radius=radius,
        fill=False,
        edgecolor=col,
        linewidth=2,
        alpha=1.0
    )
    circ.set_picker(True)
    
    circ._map_xy = (ux, uy)  #  store XY for label
    ax.add_patch(circ)

    circles.append(circ)
    circle_to_col[id(circ)] = col_idx


    
def on_pick(event):
    artist = event.artist
    if id(artist) not in circle_to_col:
        return

    col_idx = circle_to_col[id(artist)]
    spectrum_det = counts_detrended[:, col_idx]

    # ✅ map coordinates from the circle artist
    ux, uy = event.artist._map_xy

    fig2, ax2 = plt.subplots(figsize=(8, 4), dpi=150)

    # plot spectrum
    ax2.plot(wavelengths, spectrum_det, label="Detrended signal")

    # peak estimate
    peak_wl_est = wavelengths[np.argmax(spectrum_det)]
    ax2.axvline(
        peak_wl_est,
        color="r",
        linestyle="--",
        label=f"Peak ~ {peak_wl_est:.2f} nm"
    )

    # labels (XY coords)
    ax2.set_xlabel("Wavelength (nm)", fontsize=label_fontsize)
    ax2.set_ylabel("Counts (a.u.)", fontsize=label_fontsize)

    ax2.set_title(
        f"Spectrum at map point (X={ux:.1f} µm, Y={uy:.1f}) µm",
        fontsize=label_fontsize,
        fontweight="bold"
    )

    ax2.tick_params(labelsize=tick_fontsize, width=1.2)
    ax2.legend(fontsize=tick_fontsize)

    # ✅ bold tick labels (extra-safe)
    for t in ax2.get_xticklabels() + ax2.get_yticklabels():
        t.set_fontweight("bold")

    plt.tight_layout()
    plt.show()

    # ============================
    # 💾 Save option to TXT
    # ============================
    save_choice = ""

    if save_choice == "y":
        # choose filename
        fname = "" #input("Enter filename (e.g., spectrum_X10p2_Y-3p1.txt) [default auto]: ").strip()
        if fname == "":
            # auto filename
            fname = f"spectrum_X{ux:.1f}_Y{uy:.1f}.txt".replace(" ", "")
            # sanitize (optional)
            fname = fname.replace("+", "").replace(",", "")

        # save 2-column data: wavelength, detrended counts
        out = np.column_stack([wavelengths, spectrum_det])

        header = (
            f"# Spectrum extracted from clicked map point\n"
            f"# X = {ux} µm\n"
            f"# Y = {uy} µm\n"
            f"# col_idx = {col_idx}\n"
            f"# peak_wl_est_nm = {peak_wl_est}\n"
            f"# Columns: wavelength_nm   detrended_counts_au\n"
        )

        np.savetxt(fname, out, header=header, fmt="%.8e")
        print(f"[OK] Saved: {fname}")
    

fig.canvas.mpl_connect('pick_event', on_pick)

plt.show()



