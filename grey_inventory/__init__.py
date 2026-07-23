"""Grey forecasting modelling for deteriorating inventory.

Translation of the MATLAB codebase accompanying:
  Wang, Xie, Yang & Wei (2026). Grey forecasting modelling for deteriorating
  inventory with interdependent quality and quantity decay.
  Expert Systems with Applications, 306, 130956.
"""

from .simulation import (
    demand_rate,
    level_at_time,
    inventory_level,
    inventory_level_simulation,
)
from .estimation import (
    lambda_initial,
    lambda_to_alpha_beta,
    lambda_inventory_residual_var,
    lambda_residual,
    irls,
)
from .optimization import profit, profit_approx

__all__ = [
    "demand_rate",
    "level_at_time",
    "inventory_level",
    "inventory_level_simulation",
    "lambda_initial",
    "lambda_to_alpha_beta",
    "lambda_inventory_residual_var",
    "lambda_residual",
    "irls",
    "profit",
    "profit_approx",
]
