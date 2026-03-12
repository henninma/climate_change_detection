# Climate Change Detection Examples

This directory contains examples demonstrating the Bayesian trend analysis capabilities.

## Available Examples

### 1. Minimal Example Script
**File**: [`minimal_example.py`](minimal_example.py)

A standalone Python script that demonstrates the complete workflow:
- Loading temperature data
- Configuring and fitting a Bayesian model
- Extracting trend estimates with uncertainty
- Creating publication-quality visualizations

**Run it**:
```bash
cd examples
python minimal_example.py
```

Expected output:
- Console output with trend statistics
- Plot showing data with posterior trend uncertainty
- Plot showing posterior distribution of the trend parameter
- Saved figure in `results/minimal_example_results.png`

**Runtime**: ~30-60 seconds

### 2. Interactive Notebook
**File**: See [`notebooks/02_minimal_example.ipynb`](../notebooks/02_minimal_example.ipynb)

An interactive Jupyter notebook version of the minimal example for step-by-step exploration.

**Run it**:
```bash
jupyter notebook
# Navigate to notebooks/02_minimal_example.ipynb
```

## What You'll Learn

These examples demonstrate:

1. **Bayesian Inference for Climate Data**
   - How to quantify uncertainty in temperature trends
   - Interpreting credible intervals vs confidence intervals
   - Making probabilistic statements about warming

2. **MCMC Sampling**
   - PyMC integration for Bayesian modeling
   - Convergence diagnostics
   - Posterior sampling and visualization

3. **Result Interpretation**
   - Understanding posterior distributions
   - Calculating probabilities of specific outcomes
   - Projecting trends over time periods

## Expected Results

Using the example temperature data (2000-2023):
- **Trend**: ~0.08 °C/year
- **95% CI**: [0.07, 0.09] °C/year  
- **Total change**: ~1.8°C over 23 years
- **P(warming)**: >99%

## Next Steps

After running these examples:
1. Replace example data with your own temperature dataset
2. Adjust priors in the configuration based on domain knowledge
3. Explore the full analysis pipeline in [`src/analysis/run_analysis.py`](../src/analysis/run_analysis.py)
4. Check out the exploratory analysis notebook for data preprocessing
