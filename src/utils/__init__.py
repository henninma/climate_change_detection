"""
Utility functions for data processing and visualization.
"""

from .data_loader import (
    load_temperature_data,
    aggregate_to_annual,
    detect_outliers,
    detrend_data
)

from .plotting import (
    plot_trend_analysis,
    plot_diagnostics,
    plot_posterior_predictions
)

__all__ = [
    'load_temperature_data',
    'aggregate_to_annual',
    'detect_outliers',
    'detrend_data',
    'plot_trend_analysis',
    'plot_diagnostics',
    'plot_posterior_predictions'
]
