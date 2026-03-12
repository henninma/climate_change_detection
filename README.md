# Climate Change Detection - Bayesian Trend Analysis

This project performs Bayesian trend analysis on temperature data to detect climate change patterns.

## Project Structure

```
climate_change_detection/
├── data/                    # Temperature data files
├── src/                     # Source code
│   ├── models/             # Bayesian models
│   ├── analysis/           # Analysis scripts
│   └── utils/              # Utility functions
├── notebooks/              # Jupyter notebooks for exploration
├── results/                # Output results and figures
└── config/                 # Configuration files
```

## Overview

This project implements Bayesian statistical methods to analyze temperature trends and quantify uncertainty in climate change detection.

### Key Features

- Bayesian linear regression for trend estimation
- Hierarchical models for spatial/temporal analysis
- Uncertainty quantification
- Posterior predictive checks
- Model comparison using information criteria

## Getting Started

1. Place temperature data in the `data/` directory
2. Configure analysis parameters in `config/params.yaml`
3. Run analysis scripts from `src/analysis/`
4. Explore results in Jupyter notebooks

## Dependencies

- Python 3.8+
- PyMC (for Bayesian inference)
- NumPy, Pandas
- Matplotlib, Seaborn (for visualization)
- ArviZ (for Bayesian analysis diagnostics)

## Citation & Credit

If you use this code or data in your work, please credit:

**Henning Åkesson**

Example citation:
> Åkesson, H. (2026). Bayesian trend analysis to detect climate change patterns. GitHub repository: https://github.com/henninma/climate_change_detection
