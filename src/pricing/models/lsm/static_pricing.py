"""Free functions for static Monte Carlo pricing (no pricer instance needed)."""
import numpy as np
from typing import Callable, Tuple

from .regression import BasisBuilder, LSRegressor
from .path_generator import PathGenerator


def price_european_call(
    S: np.ndarray | float, K: float,
    r: float, T: float,
    sigma: float, q: float = 0.0,
) -> float:
    """Static European call pricing from terminal prices."""
    if isinstance(S, np.ndarray):
        payoffs = np.maximum(S - K, 0.0)
        return float(np.mean(payoffs) * np.exp(-r * T))
    raise ValueError("For S0 input, use the class instance method instead")


def price_european_put(
    S: np.ndarray | float, K: float,
    r: float, T: float,
    sigma: float = 0.0, q: float = 0.0,
) -> float:
    """Static European put pricing from terminal prices."""
    if isinstance(S, np.ndarray):
        payoffs = np.maximum(K - S, 0.0)
        return float(np.mean(payoffs) * np.exp(-r * T))
    raise ValueError("For S0 input, use the class instance method instead")


def price_american_put_ls(
    S_paths: np.ndarray, K: float, r: float, dt: float,
    regression_type: str = "quadratic", degree: int = 2,
) -> float:
    """Static American put pricing via Longstaff-Schwartz."""
    def payoff(S: np.ndarray) -> np.ndarray:
        return np.maximum(K - S, 0.0)
    return _static_ls_price(S_paths, payoff, r, dt, regression_type, degree)


def price_american_call_ls(
    S_paths: np.ndarray, K: float, r: float, dt: float,
    regression_type: str = "quadratic", degree: int = 2,
) -> float:
    """Static American call pricing via Longstaff-Schwartz."""
    def payoff(S: np.ndarray) -> np.ndarray:
        return np.maximum(S - K, 0.0)
    return _static_ls_price(S_paths, payoff, r, dt, regression_type, degree)


def _static_ls_price(
    S_paths: np.ndarray, payoff: Callable,
    r: float, dt: float,
    regression_type: str, degree: int,
) -> float:
    """Pricing américain statique partagé entre put et call."""
    nb_steps = S_paths.shape[1] - 1
    discount_factor = np.exp(-r * dt)
    cashflows = payoff(S_paths[:, -1])
    cashflows = _static_ls_backward(
        S_paths, cashflows, payoff, discount_factor,
        nb_steps, regression_type, degree,
    )
    price = float(np.mean(cashflows * discount_factor))
    immediate = payoff(S_paths[0, 0])
    return max(price, immediate)


def _static_ls_backward(
    S_paths: np.ndarray, cashflows: np.ndarray,
    payoff: Callable, discount_factor: float,
    nb_steps: int, regression_type: str, degree: int,
) -> np.ndarray:
    """Backward recursion pour le pricing LS statique."""
    for t in range(nb_steps - 1, 0, -1):
        cashflows = cashflows * discount_factor
        S_t = S_paths[:, t]
        intrinsic_val = payoff(S_t)
        itm = intrinsic_val > 0
        if np.count_nonzero(itm) > 0:
            cashflows = _static_ls_step(
                cashflows, S_t, intrinsic_val, itm,
                regression_type, degree,
            )
    return cashflows


def _static_ls_step(
    cashflows: np.ndarray, S_t: np.ndarray,
    intrinsic_val: np.ndarray, itm: np.ndarray,
    regression_type: str, degree: int,
) -> np.ndarray:
    """Un pas de la récursion backward LS statique."""
    X, Y = S_t[itm], cashflows[itm]
    _builder = BasisBuilder()
    basis = _builder.build(X, regression_type, degree)
    continuation_val = LSRegressor.static_fit(basis, X, Y, degree)
    exercise = intrinsic_val[itm] > continuation_val
    updated_cf = cashflows.copy()
    updated_cf[itm] = np.where(exercise, intrinsic_val[itm], Y)
    return updated_cf


def generate_gbm_paths(
    S0: float, r: float, sigma: float,
    T: float, nb_steps: int, nb_paths: int,
    q: float = 0.0, seed: int = 42,
    antithetic: bool = True,
) -> Tuple[np.ndarray, float]:
    """Generate GBM price paths for use with static pricing functions."""
    return PathGenerator.generate_gbm_paths_static(
        S0, r, sigma, T, nb_steps, nb_paths, q, seed, antithetic,
    )
