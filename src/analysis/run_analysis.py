"""
Main script for running Bayesian trend analysis on temperature data.
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.bayesian_trend import BayesianTrendModel
from utils.data_loader import load_temperature_data, aggregate_to_annual
from utils.plotting import plot_trend_analysis, plot_diagnostics


def main(config_path: str = '../../config/params.yaml'):
    """
    Run the complete Bayesian trend analysis pipeline.
    
    Parameters
    ----------
    config_path : str
        Path to configuration file
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("=" * 60)
    print("Bayesian Climate Trend Analysis")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading temperature data...")
    data_config = config['data']
    time, temperature, df = load_temperature_data(
        filepath=data_config['temperature_file'],
        date_column=data_config['date_column'],
        temp_column=data_config['temp_column'],
        start_year=data_config['start_year'],
        end_year=data_config['end_year']
    )
    
    print(f"   Loaded {len(temperature)} data points")
    print(f"   Time range: {time.min():.2f} - {time.max():.2f}")
    print(f"   Temperature range: {temperature.min():.2f}°C - {temperature.max():.2f}°C")
    
    # Initialize and fit model
    print("\n2. Building Bayesian model...")
    model = BayesianTrendModel(config)
    
    print("\n3. Running MCMC sampling...")
    print(f"   Chains: {config['model']['n_chains']}")
    print(f"   Samples per chain: {config['model']['n_samples']}")
    print(f"   Tuning steps: {config['model']['n_tune']}")
    
    trace = model.fit(time, temperature)
    
    # Get trend summary
    print("\n4. Analyzing results...")
    summary = model.get_trend_summary()
    
    print("\n" + "=" * 60)
    print("TREND ANALYSIS RESULTS")
    print("=" * 60)
    print(f"\nTemperature Trend (°C/year):")
    print(f"  Mean:   {summary['mean_trend']:.4f}")
    print(f"  Median: {summary['median_trend']:.4f}")
    print(f"  Std:    {summary['std_trend']:.4f}")
    
    ci = int(config['output']['credible_interval'] * 100)
    print(f"\n{ci}% Credible Interval:")
    print(f"  [{summary[f'ci_{ci}_lower']:.4f}, {summary[f'ci_{ci}_upper']:.4f}]")
    
    print(f"\nProbability of warming: {summary['prob_warming']:.1%}")
    
    # Trend over the analysis period
    years = time.max() - time.min()
    total_trend = summary['mean_trend'] * years
    print(f"\nTotal trend over {years:.1f} years: {total_trend:.2f}°C")
    
    # Model diagnostics
    print("\n5. Running diagnostics...")
    print(az.summary(trace, var_names=['intercept', 'actual_slope', 'sigma']))
    
    # Generate plots
    if config['output']['generate_plots']:
        print("\n6. Generating plots...")
        results_dir = config['output']['results_dir']
        os.makedirs(results_dir, exist_ok=True)
        
        # Diagnostic plots
        plot_diagnostics(trace, save_path=f"{results_dir}/diagnostics.png")
        
        # Trend analysis plot
        plot_trend_analysis(
            time, temperature, model, trace,
            save_path=f"{results_dir}/trend_analysis.png"
        )
        
        print(f"   Plots saved to {results_dir}/")
    
    # Save results
    if config['output']['save_trace']:
        trace_path = f"{config['output']['results_dir']}/trace.nc"
        trace.to_netcdf(trace_path)
        print(f"\n7. Trace saved to {trace_path}")
    
    # Save summary
    summary_df = pd.DataFrame([summary])
    summary_path = f"{config['output']['results_dir']}/trend_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"   Summary saved to {summary_path}")
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
