# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 10:52:48 2025
@author: Nilesh Dalla
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import matplotlib.ticker as ticker
from scipy.optimize import curve_fit

# -----------------------------
# File + data loading
# -----------------------------
file = '' #Add file path
df = pd.read_csv(file, index_col="Unnamed: 0", sep='\t')  # ,header =none

start_det = 1.937
end_det = 1.947
start_exc = 2.132
end_exc = 2.137

# -----------------------------
# Model definitions
# -----------------------------
def gaussian2(x, a, x0, sigma, offset):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2) + offset)

def gaussian(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def double_gaussian(x, A1, A2, mu2, sigma2, c, mu1, sigma1):
    """
    Two-Gaussian model:
    - G1: Gaussian (A1 free; mu1 and sigma1 fixed)
    - G2: Gaussian (A2, mu2, sigma2 free)
    - c: baseline offset
    """
    return gaussian(x, A1, mu1, sigma1) + gaussian(x, A2, mu2, sigma2) + c

def lorentzian(x, A, x0, gamma, offset):
    """
    Lorentzian line shape with HWHM=gamma
    A * (gamma^2 / ((x-x0)^2 + gamma^2)) + offset
    """
    return A * (gamma**2 / ((x - x0)**2 + gamma**2)) + offset

def double_gaussian_lorentzian(x, A1, A2, mu2, sigma2, c, mu1, sigma1):
    """
    Model:
    - G1: Gaussian (fixed mu1, sigma1)
    - L2: Lorentzian (free A2, mu2, sigma2)
    - c: baseline offset
    """
    g1 = gaussian(x, A1, mu1, sigma1)
    # sigma2 plays the role of gamma (HWHM) for lorentzian
    l2 = lorentzian(x, A2, mu2, sigma2, offset=0.0)
    return g1 + l2 + c

# -----------------------------
# Utilities
# -----------------------------
def drop_pandas_suffix_duplicates(df):
    new_cols = []
    for c in df.columns:
        s = str(c)
        if s.count('.') >= 2:
            base, suffix = s.rsplit('.', 1)
            if suffix.isdigit():
                try:
                    float(base)
                    new_cols.append(base)  # normalize
                    continue
                except ValueError:
                    pass
        new_cols.append(s)
    df2 = df.copy()
    df2.columns = new_cols
    df2 = df2.loc[:, ~df2.columns.duplicated()]  # keep first
    return df2

def reduced_chi2(y_obs, y_fit, n_params, y_sigma=None):
    """
    Reduced chi-square (qualitative).
    If y_sigma is None, assumes constant sigma=1 for all points.
    """
    y_obs = np.asarray(y_obs, dtype=float)
    y_fit = np.asarray(y_fit, dtype=float)
    resid = y_obs - y_fit

    if y_sigma is None:
        sigma2 = 1.0
    else:
        y_sigma = np.asarray(y_sigma, dtype=float)
        sigma2 = y_sigma**2

    N = y_obs.size
    dof = max(1, N - n_params)
    chi2 = np.sum((resid**2) / sigma2)
    return chi2 / dof

def fit_model_and_chi2(MODEL, df_sub, mu1_fixed, sigma1_fixed, plot_each=False):
    """
    Fits chosen MODEL for every column (excitation value) in df_sub.
    Returns:
      fit_parameters_df, fit_errs_df, chi2_red_array, x_vals
    """
    fit_params = []
    fit_errs = []
    chi2_red_list = []
    x_vals_local = pd.to_numeric(df_sub.columns).values

    for column in df_sub.columns.values:
        Det_energy = df_sub.index.values.astype(float)
        Counts = df_sub[column].values.astype(float)

        try:
            # Initial guess
            A2_guess = 10
            A1_guess = 63
            mu2_guess = column - 0.192
            sigma2_guess = 0.0005
            c_guess = Counts.min()

            p0 = [A1_guess, A2_guess, mu2_guess, sigma2_guess, c_guess]

            # Fit
            params, cov = curve_fit(
                lambda x, A1, A2, mu2, sigma2, c: (
                    double_gaussian(x, A1, A2, mu2, sigma2, c, mu1_fixed, sigma1_fixed)
                    if MODEL == "gauss_gauss"
                    else double_gaussian_lorentzian(x, A1, A2, mu2, sigma2, c, mu1_fixed, sigma1_fixed)
                ),
                Det_energy, Counts, p0=p0,
                bounds=(
                    [0, 0, column - 0.192 - 0.0005, 0, 0],                 # lower
                    [500, 500, column - 0.192 + 0.0005, 0.001, np.inf]    # upper
                )
            )

            errors = np.sqrt(np.diag(cov))
            fit_params.append(params)
            fit_errs.append(errors)

            # Compute fitted y for chi2
            A1, A2, mu2, sigma2, c = params
            if MODEL == "gauss_gauss":
                fitted_y = double_gaussian(Det_energy, A1, A2, mu2, sigma2, c, mu1_fixed, sigma1_fixed)
            else:
                fitted_y = double_gaussian_lorentzian(Det_energy, A1, A2, mu2, sigma2, c, mu1_fixed, sigma1_fixed)

            # Reduced chi-square (qualitative, constant sigma)
            # Number of fitted parameters = 5: A1, A2, mu2, sigma2, c
            chi2_red = reduced_chi2(Counts, fitted_y, n_params=5, y_sigma=None)
            chi2_red_list.append(chi2_red)

            if plot_each:
                baseline = np.full_like(Det_energy, c)
                g1 = gaussian(Det_energy, A1, mu1_fixed, sigma1_fixed) + baseline

                if MODEL == "gauss_gauss":
                    g2 = gaussian(Det_energy, A2, mu2, sigma2) + baseline
                    free_label = "Gaussian 2 (free)"
                    free_style = 'g--'
                else:
                    g2 = lorentzian(Det_energy, A2, mu2, sigma2, offset=0.0) + baseline
                    free_label = "Lorentzian 2 (free)"
                    free_style = 'g--'

                plt.figure(figsize=(10, 5))
                plt.plot(Det_energy, Counts, 'k.', label="Data")
                plt.plot(Det_energy, fitted_y, 'r-', label="Total Fit")
                plt.plot(Det_energy, g1, 'b--', label="Gaussian 1 (fixed μ1, σ1)")
                plt.plot(Det_energy, g2, free_style, label=free_label)
                plt.plot(Det_energy, baseline, 'm:', label="Baseline offset")
                plt.xlabel("Detection Energy (eV)")
                plt.ylabel("Intensity")
                plt.title(f"[{MODEL}] Exc Wl {column} nm")
                plt.grid(True)
                plt.legend()
                plt.show()

        except Exception as e:
            print(f"[{MODEL}] Fit failed for {column}: {e}")
            fit_params.append([np.NaN] * 5)
            fit_errs.append([np.NaN] * 5)
            chi2_red_list.append(np.NaN)

    fit_parameters_df = pd.DataFrame(fit_params, columns=['A1', 'A2', 'mu2', 'sigma2', 'c'])
    fit_errs_df = pd.DataFrame(fit_errs, columns=['A1_err', 'A2_err', 'mu2_err', 'sigma2_err', 'c_err'])
    return fit_parameters_df, fit_errs_df, np.array(chi2_red_list, dtype=float), x_vals_local

# -----------------------------
# Droping duplicate columns + slicing
# -----------------------------
df = drop_pandas_suffix_duplicates(df)
df.columns = np.round(df.columns.astype(float), 5)
df = df.astype("float")

Exc_wl = df.columns.astype(float).values
mask = (Exc_wl >= start_exc) & (Exc_wl <= end_exc)
df = df.iloc[:, mask]

df_sub = df[(df.index > start_det) & (df.index < end_det)]

# -----------------------------
# Plot resonances map
# -----------------------------
Y, X = np.meshgrid(df_sub.columns.astype(float), df_sub.index.astype(float))
Z = df_sub.values

plt.figure(dpi=600)
plt.pcolormesh(X, Y, Z, shading="auto", cmap="viridis", vmin=1506, vmax=1928)
plt.colorbar(label="Intensity")
plt.xlabel("Detection Energy (eV)")
plt.ylabel("Excitation Energy (eV)")
plt.title("Resonances Map")
plt.show()

# -----------------------------
# Fixed Gaussian parameters
# -----------------------------
mu1_fixed = 1.9424      # fixed center energy (eV)
sigma1_fixed = 0.0005  # fixed width
# A1_fixed not used in fit because A1 is still fitted in your model parameter list

# -----------------------------
# Fit BOTH models + chi2 plots
# -----------------------------
results_gauss, results_gauss_err, chi2_gauss, x_vals = fit_model_and_chi2(
    "gauss_gauss", df_sub, mu1_fixed, sigma1_fixed, plot_each=False
)
results_lor, results_lor_err, chi2_lor, _ = fit_model_and_chi2(
    "gauss_lorentz", df_sub, mu1_fixed, sigma1_fixed, plot_each=False
)

# -----------------------------
# Plot fitted parameters vs Exc (optional: for Gaussian model)
# -----------------------------
fit_parameters = results_gauss
fit_parameters_err = results_gauss_err

x_vals_pd = pd.to_numeric(df_sub.columns)

plt.figure(dpi=600)
plt.errorbar(x_vals_pd, fit_parameters["A1"], yerr=fit_parameters_err["A1_err"], fmt='o-', capsize=4)
plt.xlabel("Exc Wl (nm)")
plt.ylabel("Fit Amp")
plt.title("ZPL gaussian fit vs Exc (Gaussian+Gaussian model)")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(dpi=600)
plt.errorbar(x_vals_pd, fit_parameters["A2"], yerr=fit_parameters_err["A2_err"], fmt='o-', capsize=4)
plt.xlabel("Exc Wl (nm)")
plt.ylabel("Fit Amp")
plt.title("Resonance gaussian fit vs Exc (Gaussian+Gaussian model)")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(dpi=600)
plt.errorbar(x_vals_pd, fit_parameters["mu2"], yerr=fit_parameters_err["mu2_err"], fmt='o-', capsize=4)
plt.xlabel("Exc Wl (nm)")
plt.ylabel(" Center Fit ")
plt.title("Resonance center fit vs Exc (Gaussian+Gaussian model)")
plt.legend()
plt.tight_layout()
plt.show()

# FWHM plot for Gaussian model
FWHM_factor_gauss = 2 * np.sqrt(2 * np.log(2))  # Gaussian FWHM = 2.3548*sigma
plt.figure(dpi=600)
plt.errorbar(
    x_vals_pd,
    FWHM_factor_gauss * fit_parameters["sigma2"] * 1000,
    yerr=FWHM_factor_gauss * fit_parameters_err["sigma2_err"] * 1000,
    fmt='o-', capsize=4
)
plt.xlabel("Exc Wl (nm)")
plt.ylabel("FWHM (meV)")
plt.title("Resonance Gaussian FWHM fit vs Exc (Gaussian+Gaussian model)")
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------
# χ²ᵣ comparison plots (this is the new part)
# -----------------------------
plt.figure(dpi=600)
plt.plot(x_vals, chi2_gauss, 'o-', label='χ²ᵣ: Gaussian 2 (free)')
plt.plot(x_vals, chi2_lor,   'o-', label='χ²ᵣ: Lorentzian 2 (free)')
plt.xlabel("Exc Wl (nm)")
plt.ylabel("Reduced χ² (qualitative)")
plt.title("Model comparison via reduced chi-square")
plt.yscale('log')  # chi-square often spans orders of magnitude
plt.grid(True, which='both', ls='--', alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(dpi=600)
ratio = chi2_lor / chi2_gauss
plt.plot(x_vals, ratio, 'o-', color='purple')
plt.axhline(1.0, color='k', lw=1, ls='--')
plt.xlabel("Exc Wl (nm)")
plt.ylabel("χ²ᵣ(Lorentz) / χ²ᵣ(Gauss)")
plt.title("Which model is better? (ratio < 1 => Lorentz better)")
plt.grid(True, ls='--', alpha=0.4)
plt.ylim(bottom=0)
plt.tight_layout()
plt.show()

# -----------------------------
# If you also want Lorentzian FWHM plot using sigma2=gamma:
# -----------------------------
fit_parameters = results_lor
fit_parameters_err = results_lor_err

# Lorentzian: FWHM = 2 * gamma, and gamma = sigma2 in our definition
FWHM_factor_lor = 2.0

plt.figure(dpi=600)
plt.errorbar(
    x_vals_pd,
    FWHM_factor_lor * fit_parameters["sigma2"] * 1000,
    yerr=FWHM_factor_lor * fit_parameters_err["sigma2_err"] * 1000,
    fmt='o-', capsize=4
)
plt.xlabel("Exc Wl (nm)")
plt.ylabel("FWHM (meV)")
plt.title("Resonance Lorentzian FWHM fit vs Exc (Gaussian+Lorentzian model)")
plt.legend()
plt.tight_layout()
plt.show()