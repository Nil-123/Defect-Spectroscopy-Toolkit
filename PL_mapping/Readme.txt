# PL Mapping and Interactive Spectral Analysis

## Overview

This module provides a Python-based tool for analysing spatially resolved photoluminescence (PL) measurements.

It processes spectral data acquired at different spatial positions, generates two-dimensional PL maps, identifies emission peaks, groups spatially related emitters, and enables interactive inspection of individual spectra.

## Features

- Visualisation of spatially resolved PL measurements.
- Peak detection within a user-defined wavelength range.
- Identification and spatial grouping of emission features.
- Colour-coded visualisation of detected spectral peaks.
- Interactive selection of individual positions to inspect their spectra.
- Option to save selected spectral data.

## Input Data

The tool requires three input files:

1. **Spectral data (`.dat`):** The first column contains wavelength values. Each subsequent column corresponds to a spectrum acquired at a particular spatial position.
2. **X-coordinate file (`.txt`):** Contains the X coordinate corresponding to each spectrum.
3. **Y-coordinate file (`.txt`):** Contains the Y coordinate corresponding to each spectrum.

The coordinate arrays must have the same number of entries as the number of spectra. Their ordering must also match the spectral acquisition order.

## Requirements

- Python 3
- NumPy
- SciPy
- Matplotlib

Install the required packages using:

```bash
pip install numpy scipy matplotlib
```

## Usage

1. Download or clone the repository.
2. Install the required Python packages.
3. Open `PL_mapping.py`.
4. Set the input data paths and adjust the analysis parameters.
5. Run the script to generate the PL map and inspect the detected emission features.

An example dataset is provided in the `examples/PL_mapping/` directory.

## Future Development

Potential extensions include:

- Improved multi-peak detection and fitting.
- Automated extraction of emitter properties.
- Correlation analysis between spectral and spatial characteristics.
- Comparison of measurements acquired under different experimental conditions.

## Author

Nilesh Dalla  
University of Warsaw

Research interests: optical spectroscopy, solid-state defects, quantum emitters, and nanophotonics.