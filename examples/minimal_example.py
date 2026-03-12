"""
Minimal example demonstrating Bayesian trend analysis on temperature data.

This script shows the core functionality:
1. Load temperature data
2. Fit Bayesian linear regression model
3. Extract trend with uncertainty
4. Visualize results
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.bayesian_trend import BayesianTrendModel
from utils.data_loader import load_temperature_data


def main():
    print("=" * 60)
    print("MINIMAL BAYESIAN CLIMATE TREND ANALYSIS EXAMPLE")
    print("=" * 60)
    
    # 1. Load example data
    print("\n1. Loading temperature data...")
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'example_temperature_data.csv')
    time, temperature, df = load_temperature_data(
        filepath=data_path,
        date_column='date',
        temp_column='temperature'
    )
    
    print(f"   - Data points: {len(temperature)}")
    print(f"   - Time span: {time.min():.1f} - {time.max():.1f} ({time.max()-time.min():.1f} years)")
    print(f"   - Temperature range: {temperature.min():.2f}°C - {temperature.max():.2f}°C")
    
    # 2. Configure minimal model
    print("\n2. Configuring Bayesian model...")
    config = {
        'priors': {
            'beta0_mu': 15.0,      # Expected baseline temperature
            'beta0_sigma': 10.0,   # Uncertainty in baseline
            'beta1_mu': 0.0,       # No prior expectation on trend
            'beta1_sigma': 0.5,    # Allow trends up to ±1°C/year with high probability
            'sigma_prior': 5.0     # Expected observation noise
        },
        'model': {
            'type': 'linear',
            'n_samples': 1000,         # Posterior samples per chain
            'n_chains': 2,             # Number of MCMC chains
            'n_tune': 500,             # Tuning/warmup steps
            'random_seed': 42
        },
        'output': {
            'results_dir': '../results',
            'save_trace': False,
            'generate_plots': True,
            'credible_interval': 0.95
        }
    }
    
    print(f"   - MCMC chains: {config['model']['n_chains']}")
    print(f"   - Samples per chain: {config['model']['n_samples']}")
    
    # 3. Fit Bayesian model
    print("\n3. Running MCMC sampling...")
    print("   (This may take 30-60 seconds...)")
    
    # Flatten priors into config for compatibility with BayesianTrendModel
    config_flat = {**config['priors'], **config['model'], **config['output']}
    config_nested = {
        'priors': config['priors'],
        'model': config['model'],
        'output': config['output']
    }
    model = BayesianTrendModel(config_nested)
    # Provide both nested and flat config for compatibility
    model.config.update(config['priors'])
    model.config.update(config['model'])
    model.config.update(config['output'])
    trace = model.fit(time, temperature)
    
    print("   ✓ Sampling complete!")
    
    # 4. Extract and display results
    print("\n4. Analyzing trend...")
    summary = model.get_trend_summary()
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print(f"\nTemperature Trend:")
    print(f"  • Mean trend:        {summary['mean_trend']:.4f} °C/year")
    print(f"  • 95% Credible Int:  [{summary['ci_95_lower']:.4f}, {summary['ci_95_upper']:.4f}] °C/year")
    print(f"  • Probability warming: {summary['prob_warming']:.1%}")
    
    # Calculate total change over period
    years = time.max() - time.min()
    total_change = summary['mean_trend'] * years
    total_change_lower = summary['ci_95_lower'] * years
    total_change_upper = summary['ci_95_upper'] * years
    
    print(f"\nTotal Temperature Change ({years:.1f} years):")
    print(f"  • Best estimate:     {total_change:.2f}°C")
    print(f"  • 95% Credible Int:  [{total_change_lower:.2f}, {total_change_upper:.2f}]°C")
    
    # 5. Create visualization
    print("\n5. Creating visualization...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left plot: Data with trend
    ax = axes[0]
    ax.plot(time, temperature, marker='o', color='steelblue', linewidth=2, label='Observations')
    
    # Extract posterior samples for trend line
    intercept = trace.posterior['beta0'].values.flatten()
    slope = trace.posterior['beta1'].values.flatten()
    time_mean = np.mean(time)
    time_std = np.std(time)
    
    # Plot credible interval for trend
    time_normalized = (time - time_mean) / time_std
    n_posterior_samples = 200
    for i in range(n_posterior_samples):
        idx = np.random.randint(0, len(intercept))
        trend_line = intercept[idx] + slope[idx] * time_normalized
        ax.plot(time, trend_line, 'red', alpha=0.02, linewidth=1)
    
    # Plot mean trend
    mean_trend = intercept.mean() + slope.mean() * time_normalized
    ax.plot(time, mean_trend, 'red', linewidth=3, label='Mean trend', linestyle='--')
    
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Temperature (°C)', fontsize=12)
    ax.set_title('Temperature Data with Bayesian Trend', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Right plot: Posterior distribution of trend
    ax = axes[1]
    actual_slope = trace.posterior['beta1'].values.flatten()
    
    ax.hist(actual_slope, bins=40, alpha=0.7, color='steelblue', 
            edgecolor='black', density=True)
    ax.axvline(summary['mean_trend'], color='red', linestyle='--', 
               linewidth=2, label=f"Mean: {summary['mean_trend']:.4f} °C/yr")
    ax.axvline(summary['ci_95_lower'], color='orange', linestyle=':', 
               linewidth=2, label='95% CI')
    ax.axvline(summary['ci_95_upper'], color='orange', linestyle=':', linewidth=2)
    ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Temperature Trend (°C/year)', fontsize=12)
    ax.set_ylabel('Posterior Density', fontsize=12)
    ax.set_title('Posterior Distribution of Trend', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save figure
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'minimal_example_results.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ Plot saved to: {output_path}")
    
    # Also show the plot
    plt.show()
    
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    
    if summary['prob_warming'] > 0.95:
        print("\n✓ Strong evidence of warming trend!")
        print(f"  The temperature has increased by approximately {total_change:.2f}°C")
        print(f"  over the {years:.1f}-year period with high confidence.")
    elif summary['prob_warming'] > 0.80:
        print("\n→ Moderate evidence of warming trend.")
        print(f"  The data suggests warming, but with some uncertainty.")
    else:
        print("\n? Insufficient evidence for a clear warming trend.")
        print(f"  More data may be needed for a definitive conclusion.")
    
    print(f"\n  The Bayesian approach provides a full probability distribution")
    print(f"  for the trend, accounting for uncertainty in the data.")
    
    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
