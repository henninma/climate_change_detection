"""
Plotting utilities for Bayesian trend analysis visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
from typing import Optional


def plot_trend_analysis(
    time: np.ndarray,
    temperature: np.ndarray,
    model,
    trace,
    save_path: Optional[str] = None
):
    """
    Create comprehensive trend analysis visualization.
    
    Parameters
    ----------
    time : np.ndarray
        Time points
    temperature : np.ndarray
        Temperature observations
    model : BayesianTrendModel
        Fitted Bayesian model
    trace : az.InferenceData
        MCMC trace
    save_path : str, optional
        Path to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Data with posterior predictive
    ax = axes[0, 0]
    ax.scatter(time, temperature, alpha=0.5, s=20, label='Observations')
    
    # Plot posterior predictive samples
    pred_samples = trace.posterior_predictive['y_obs'].values
    n_samples = min(100, pred_samples.shape[1])
    
    for i in range(n_samples):
        chain_idx = np.random.randint(0, pred_samples.shape[0])
        sample_idx = np.random.randint(0, pred_samples.shape[1])
        ax.plot(time, pred_samples[chain_idx, sample_idx, :], 
                'r-', alpha=0.02, linewidth=1)
    
    # Plot mean trend
    intercept_mean = trace.posterior['intercept'].mean().values
    slope_mean = trace.posterior['slope'].mean().values
    time_mean = trace.posterior['time_mean'].mean().values
    time_std = trace.posterior['time_std'].mean().values
    time_norm = (time - time_mean) / time_std
    trend_mean = intercept_mean + slope_mean * time_norm
    
    ax.plot(time, trend_mean, 'b-', linewidth=2, label='Mean trend')
    ax.set_xlabel('Year')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Temperature Data with Posterior Predictions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Posterior distribution of trend
    ax = axes[0, 1]
    slope_samples = trace.posterior['actual_slope'].values.flatten()
    az.plot_posterior(slope_samples, ax=ax, ref_val=0)
    ax.set_xlabel('Trend (°C/year)')
    ax.set_title('Posterior Distribution of Temperature Trend')
    
    # 3. Residuals
    ax = axes[1, 0]
    mean_prediction = trace.posterior_predictive['y_obs'].mean(dim=['chain', 'draw']).values
    residuals = temperature - mean_prediction
    
    ax.scatter(time, residuals, alpha=0.5, s=20)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Year')
    ax.set_ylabel('Residuals (°C)')
    ax.set_title('Model Residuals')
    ax.grid(True, alpha=0.3)
    
    # 4. Posterior predictive check
    ax = axes[1, 1]
    ax.hist(temperature, bins=30, alpha=0.5, density=True, label='Observed')
    
    # Plot several posterior predictive distributions
    for i in range(min(50, pred_samples.shape[1])):
        chain_idx = np.random.randint(0, pred_samples.shape[0])
        sample_idx = np.random.randint(0, pred_samples.shape[1])
        ax.hist(pred_samples[chain_idx, sample_idx, :], 
                bins=30, alpha=0.02, density=True, color='red')
    
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Density')
    ax.set_title('Posterior Predictive Check')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_diagnostics(trace, save_path: Optional[str] = None):
    """
    Create MCMC diagnostic plots.
    
    Parameters
    ----------
    trace : az.InferenceData
        MCMC trace
    save_path : str, optional
        Path to save the figure
    """
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    
    # Trace plots
    az.plot_trace(
        trace,
        var_names=['intercept', 'actual_slope', 'sigma'],
        axes=axes
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_posterior_predictions(
    time: np.ndarray,
    time_future: np.ndarray,
    model,
    trace,
    save_path: Optional[str] = None
):
    """
    Plot temperature predictions into the future.
    
    Parameters
    ----------
    time : np.ndarray
        Historical time points
    time_future : np.ndarray
        Future time points for prediction
    model : BayesianTrendModel
        Fitted model
    trace : az.InferenceData
        MCMC trace
    save_path : str, optional
        Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Generate predictions
    mean_pred, pred_samples = model.predict(time_future, n_samples=1000)
    
    # Plot uncertainty bands
    percentiles = [5, 25, 50, 75, 95]
    pred_percentiles = np.percentile(pred_samples, percentiles, axis=0)
    
    ax.fill_between(time_future, pred_percentiles[0], pred_percentiles[4],
                     alpha=0.2, color='blue', label='90% CI')
    ax.fill_between(time_future, pred_percentiles[1], pred_percentiles[3],
                     alpha=0.3, color='blue', label='50% CI')
    ax.plot(time_future, pred_percentiles[2], 'b-', linewidth=2, label='Median')
    
    ax.set_xlabel('Year')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Temperature Predictions with Uncertainty')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
