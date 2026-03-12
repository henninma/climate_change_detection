"""
Bayesian trend analysis models for temperature data.
"""

import numpy as np
import pymc as pm
import arviz as az
from typing import Dict, Tuple, Optional


class BayesianTrendModel:
    """
    Bayesian linear regression model for temperature trend analysis.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the Bayesian trend model.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary containing model parameters and priors
        """
        self.config = config
        self.model = None
        self.trace = None
        
    def build_linear_model(self, time: np.ndarray, temperature: np.ndarray) -> pm.Model:
        """
        Build a Bayesian linear regression model.
        
        Parameters
        ----------
        time : np.ndarray
            Time points (e.g., years)
        temperature : np.ndarray
            Temperature observations
            
        Returns
        -------
        pm.Model
            PyMC model object
        """
        # Standardize time for numerical stability
        time_mean = np.mean(time)
        time_std = np.std(time)
        time_normalized = (time - time_mean) / time_std

        with pm.Model() as model:
            # use time_normalized directly in your priors/likelihood, e.g.:
            beta0 = pm.Normal('beta0', mu=self.config['beta0_mu'], sigma=self.config['beta0_sigma'])
            beta1 = pm.Normal('beta1', mu=self.config['beta1_mu'], sigma=self.config['beta1_sigma'])

            sigma = pm.HalfNormal('sigma', sigma=self.config['sigma_prior'])

            mu = beta0 + beta1 * time_normalized

            pm.Normal('obs', mu=mu, sigma=sigma, observed=temperature)

        
        self.model = model
        return model
    
    def fit(self, time: np.ndarray, temperature: np.ndarray) -> az.InferenceData:
        """
        Fit the Bayesian model using MCMC sampling.
        
        Parameters
        ----------
        time : np.ndarray
            Time points (e.g., years)
        temperature : np.ndarray
            Temperature observations
            
        Returns
        -------
        az.InferenceData
            ArviZ inference data object containing the trace
        """
        # Build model
        model_type = self.config['model']['type']
        
        if model_type == 'linear':
            self.build_linear_model(time, temperature)
        else:
            raise ValueError(f"Model type '{model_type}' not implemented yet")
        
        # Sample from posterior
        with self.model:
            self.trace = pm.sample(
                draws=self.config['model']['n_samples'],
                tune=self.config['model']['n_tune'],
                chains=self.config['model']['n_chains'],
                random_seed=self.config['model']['random_seed'],
                return_inferencedata=True
            )
            
            # Posterior predictive sampling
            pm.sample_posterior_predictive(
                self.trace,
                extend_inferencedata=True
            )
        
        return self.trace
    
    def get_trend_summary(self) -> Dict:
        """
        Get summary statistics for the temperature trend.
        
        Returns
        -------
        dict
            Dictionary containing trend statistics
        """
        if self.trace is None:
            raise ValueError("Model must be fit before getting summary")
        
        # Extract slope (trend in °C/year)
        slope_samples = self.trace.posterior['beta1'].values.flatten()
        
        ci = self.config['output']['credible_interval']
        lower = (1 - ci) / 2
        upper = 1 - lower
        
        summary = {
            'mean_trend': float(np.mean(slope_samples)),
            'median_trend': float(np.median(slope_samples)),
            'std_trend': float(np.std(slope_samples)),
            f'ci_{int(ci*100)}_lower': float(np.percentile(slope_samples, lower*100)),
            f'ci_{int(ci*100)}_upper': float(np.percentile(slope_samples, upper*100)),
            'prob_warming': float(np.mean(slope_samples > 0))
        }
        
        return summary
    
    def predict(self, time_new: np.ndarray, n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions for new time points.
        
        Parameters
        ----------
        time_new : np.ndarray
            New time points for prediction
        n_samples : int
            Number of posterior samples to use
            
        Returns
        -------
        tuple
            (mean_predictions, prediction_samples)
        """
        if self.trace is None:
            raise ValueError("Model must be fit before prediction")
        
        # Get posterior samples
        intercept = self.trace.posterior['intercept'].values.flatten()
        slope = self.trace.posterior['slope'].values.flatten()
        sigma = self.trace.posterior['sigma'].values.flatten()
        time_mean = self.trace.posterior['time_mean'].values.flatten()[0]
        time_std = self.trace.posterior['time_std'].values.flatten()[0]
        
        # Random sample indices
        n_post_samples = len(intercept)
        sample_idx = np.random.choice(n_post_samples, size=n_samples, replace=False)
        
        # Normalize new time points
        time_normalized = (time_new - time_mean) / time_std
        
        # Generate predictions
        predictions = np.zeros((n_samples, len(time_new)))
        for i, idx in enumerate(sample_idx):
            mu = intercept[idx] + slope[idx] * time_normalized
            predictions[i] = np.random.normal(mu, sigma[idx])
        
        mean_pred = np.mean(predictions, axis=0)
        
        return mean_pred, predictions
