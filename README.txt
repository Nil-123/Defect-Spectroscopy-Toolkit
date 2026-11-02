# Defect Spectroscopy Toolkit

A modular collection of Python tools for processing, analysing, and visualising optical spectroscopy data from solid-state defects and quantum emitters.

The project aims to bring together reusable analysis routines for common experimental techniques, providing a foundation for reproducible data analysis and comparison between different measurements.

## Available Tools

### 1. PL Mapping and Interactive Spectral Analysis

A tool for analysing spatially resolved photoluminescence measurements.

Current capabilities include:
- Generation of two-dimensional PL maps.
- Spectral peak detection.
- Spatial grouping of detected emission features.
- Interactive inspection of individual spectra.

See [PL Mapping documentation](src/defect_spectroscopy/mapping/README.md) for details.

## Planned Modules

The repository is intended to grow with additional analysis tools, including:

- **Spectral analysis:** Peak detection, fitting, linewidth extraction, and background subtraction.
- **Emitter correlation:** Analysis of relationships between emitter properties, such as emission wavelength, linewidth, intensity, and excitation resonances.
- **Temperature-dependent spectroscopy:** Tracking spectral properties as a function of temperature.
- **Polarisation-dependent spectroscopy:** Analysis of emission and excitation polarisation characteristics.
- **PLE analysis:** Identification and characterisation of excitation resonances.
- **Photon-correlation analysis:** Processing and fitting of second-order correlation measurements.

These are planned extensions, not necessarily implemented features.

## Repository Structure

```text
src/                  Python analysis modules
examples/             Example datasets
notebooks/            Interactive demonstrations
docs/                 Additional documentation
tests/                Software tests
results/              Locally generated analysis outputs
```

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Defect-Spectroscopy-Toolkit.git
cd Defect-Spectroscopy-Toolkit
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Getting Started

Each analysis module contains its own documentation and example data. Start with the README corresponding to the tool you want to use.

## Contributing and Development

The project is being developed as a collection of modular, reusable scientific analysis tools. New analysis routines and improvements are welcome.

## Author

Nilesh Dalla  
University of Warsaw

Research interests: optical spectroscopy, solid-state defects, quantum emitters, and nanophotonics.