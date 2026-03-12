# Getting Started with Climate Change Detection

This guide will help you set up and run the Bayesian trend analysis on temperature data.

## Installation

1. **Create a virtual environment** (recommended):
   ```bash
   cd climate_change_detection
   python -m venv venv
   source venv/bin/activate  # On Linux/Mac
   # or
   venv\Scripts\activate  # On Windows
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### Option 1: Using the Analysis Script

1. **Prepare your data**: 
   - Place your temperature data CSV file in the `data/` folder
   - The file should have columns for date and temperature
   - An example file is provided: `data/example_temperature_data.csv`

2. **Configure analysis**:
   - Edit `config/params.yaml` to specify:
     - Data file path and column names
     - Model parameters (chains, samples, etc.)
     - Prior distributions
     - Output settings

3. **Run the analysis**:
   ```bash
   cd src/analysis
   python run_analysis.py
   ```

4. **View results**:
   - Results will be saved in `results/` directory
   - Includes diagnostic plots, trend analysis plots, and CSV summary

### Option 2: Using Jupyter Notebooks

1. **Launch Jupyter**:
   ```bash
   jupyter notebook
   ```

2. **Open the exploratory notebook**:
   - Navigate to `notebooks/01_exploratory_analysis.ipynb`
   - Follow the notebook to explore your data

3. **Create a Bayesian analysis notebook**:
   - Use the example code below to run Bayesian analysis interactively

## Example Usage in Python

```python
import sys
import yaml
sys.path.append('../src')

from models.bayesian_trend import BayesianTrendModel
from utils.data_loader import load_temperature_data
from utils.plotting import plot_trend_analysis

# Load configuration
with open('../config/params.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Load data
time, temperature, df = load_temperature_data(
    filepath='../data/example_temperature_data.csv',
    date_column='date',
    temp_column='temperature'
)

# Create and fit model
model = BayesianTrendModel(config)
trace = model.fit(time, temperature)

# Get trend summary
summary = model.get_trend_summary()
print(f"Mean trend: {summary['mean_trend']:.4f} °C/year")
print(f"Probability of warming: {summary['prob_warming']:.1%}")

# Generate plots
plot_trend_analysis(time, temperature, model, trace, 
                   save_path='../results/trend_plot.png')
```

## Understanding the Output

### Trend Summary
- **mean_trend**: Average temperature change per year (°C/year)
- **median_trend**: Median of the posterior distribution
- **ci_95_lower/upper**: 95% credible interval bounds
- **prob_warming**: Posterior probability that trend is positive

### Diagnostic Plots
- **Trace plots**: Check for MCMC convergence
- **Posterior distributions**: Uncertainty in parameter estimates
- **Residual plots**: Model fit quality
- **Posterior predictive checks**: Model validation

## Customizing the Analysis

### Adjusting Priors
Edit `config/params.yaml` under the `priors` section:
```yaml
priors:
  intercept_mu: 15.0       # Expected baseline temperature
  intercept_sigma: 10.0    # Uncertainty in baseline
  slope_mu: 0.0            # Expected trend (0 = no prior belief)
  slope_sigma: 0.5         # Uncertainty in trend
  sigma_sigma: 5.0         # Expected observation noise
```

### Increasing Sampling
For more robust results, increase MCMC samples:
```yaml
model:
  n_samples: 4000
  n_chains: 4
  n_tune: 2000
```

### Data Filtering
Filter data by year range:
```yaml
data:
  start_year: 2000
  end_year: 2023
```

## Troubleshooting

### Import Errors
- Make sure you're in the correct directory
- Check that all dependencies are installed: `pip list`

### Convergence Issues
- Increase `n_tune` in config file
- Check for data outliers or missing values
- Try more informative priors

### Memory Issues
- Reduce `n_samples` or `n_chains`
- Process data in smaller time windows

## Next Steps

1. **Explore seasonal patterns**: Decompose data into trend and seasonal components
2. **Hierarchical models**: Analyze multiple locations simultaneously
3. **Change-point detection**: Identify when trends shift
4. **Future projections**: Extend predictions beyond observed data

## References

- PyMC documentation: https://www.pymc.io/
- ArviZ (Bayesian diagnostics): https://python.arviz.org/
- Bayesian Data Analysis (Gelman et al.)
