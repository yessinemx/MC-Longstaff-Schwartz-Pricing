import math
import datetime as dt
import numpy as np
from typing import Optional, Tuple

from ...parameters.market import Market
from ...parameters.option import Option
from ...brownian_motion.brownian_motion import BrownianMotion


class PathGenerator:
    """Génération des chemins de prix GBM (avec dividende discret)."""

    def __init__(
        self,
        market: Market,
        option: Option,
        pricing_date: dt.date,
        nb_steps: int,
        bm: BrownianMotion,
    ) -> None:
        self.market = market
        self.option = option
        self.pricing_date = pricing_date
        self.nb_steps = nb_steps
        self.bm = bm

    def generate_price_paths(
        self, nb_paths: int, antithetic: bool = True
    ) -> Tuple[np.ndarray, float, float]:
        """Construit les chemins de prix S_t à partir des dW générés par BrownianMotion."""
        T = (self.option.maturity - self.pricing_date).days / 365.0
        dW, dt_val = self.bm.generate_dw(T, self.nb_steps, nb_paths, antithetic)
        S_paths, log_returns = self._build_gbm_paths(dW, dt_val)
        self._apply_discrete_dividend(S_paths, log_returns, T)
        return S_paths, dt_val, T

    def _build_gbm_paths(self, dW: np.ndarray, dt_val: float) -> Tuple[np.ndarray, np.ndarray]:
        """Construit les chemins GBM à partir des incréments browniens."""
        drift_term = (self.market.int_rate - 0.5 * self.market.sigma**2) * dt_val
        diffusion_term = self.market.sigma * dW
        log_returns = np.cumsum(drift_term + diffusion_term, axis=1)

        S_paths = np.zeros((dW.shape[0], self.nb_steps + 1))
        S_paths[:, 0] = self.market.stock_price
        S_paths[:, 1:] = self.market.stock_price * np.exp(log_returns)
        return S_paths, log_returns

    def _apply_discrete_dividend(self, S_paths: np.ndarray, log_returns: np.ndarray, T: float) -> None:
        """Applique le dividende discret en deux segments si applicable."""
        div_step = self.compute_dividend_step(T)
        if div_step is None or div_step <= 0 or div_step > self.nb_steps:
            return
        S_ex = S_paths[:, div_step] - self.market.div
        S_paths[:, div_step] = S_ex
        if div_step < self.nb_steps:
            incremental = log_returns[:, div_step:] - log_returns[:, div_step - 1:div_step]
            S_paths[:, div_step + 1:] = S_ex[:, np.newaxis] * np.exp(incremental)

    def compute_dividend_step(self, T: float) -> Optional[int]:
        """Compute the time step at which dividend is paid."""
        if not hasattr(self.market, 'div_date') or self.market.div_date is None:
            return None
        if self.market.div <= 0:
            return None
        div_time = (self.market.div_date - self.pricing_date).days / 365.0
        if div_time < 0 or div_time > T:
            return None
        dt_val = T / self.nb_steps
        return int(round(div_time / dt_val))

    def generate_random_matrix(self, nb_paths: int, rng_method: str) -> np.ndarray:
        """Pré-génère les nombres aléatoires pour le pricing scalaire."""
        if rng_method == "uniform_ppf":
            from scipy.stats import norm as _norm
            U = self.bm.rng.uniform(0.0, 1.0, (nb_paths, self.nb_steps))
            U = np.clip(U, 1e-10, 1.0 - 1e-10)
            return _norm.ppf(U)
        return self.bm.rng.standard_normal((nb_paths, self.nb_steps))

    def simulate_single_path(
        self, Z_row: np.ndarray, dt_val: float,
        sqrt_dt: float, div_step: Optional[int]
    ) -> float:
        """Simule un seul chemin de prix et retourne le prix terminal."""
        S = self.market.stock_price
        for step in range(self.nb_steps):
            dW = Z_row[step] * sqrt_dt
            drift = (self.market.int_rate - 0.5 * self.market.sigma**2) * dt_val
            diffusion = self.market.sigma * dW
            S *= math.exp(drift + diffusion)
            if div_step is not None and step + 1 == div_step:
                S -= self.market.div
        return S

    def scalar_generate_paths(
        self, Z: np.ndarray, nb_paths: int, dt_val: float,
        sqrt_dt: float, div_step: Optional[int]
    ) -> np.ndarray:
        """Génère tous les chemins de prix pour le pricing américain scalaire."""
        S_paths = np.zeros((nb_paths, self.nb_steps + 1))
        S_paths[:, 0] = self.market.stock_price
        for path in range(nb_paths):
            S = self.market.stock_price
            for step in range(self.nb_steps):
                dW = Z[path, step] * sqrt_dt
                drift = (self.market.int_rate - 0.5 * self.market.sigma**2) * dt_val
                diffusion = self.market.sigma * dW
                S *= math.exp(drift + diffusion)
                if div_step is not None and step + 1 == div_step:
                    S -= self.market.div
                S_paths[path, step + 1] = S
        return S_paths

    @staticmethod
    def generate_gbm_paths_static(
        S0: float, r: float, sigma: float,
        T: float, nb_steps: int, nb_paths: int,
        q: float = 0.0, seed: int = 42,
        antithetic: bool = True
    ) -> Tuple[np.ndarray, float]:
        """Generate GBM price paths for use with static pricing methods."""
        bm = BrownianMotion(seed=seed)
        dW, dt_val = bm.generate_dw(T, nb_steps, nb_paths, antithetic)
        drift = (r - q - 0.5 * sigma**2) * dt_val
        diffusion = sigma * dW
        log_returns = np.cumsum(drift + diffusion, axis=1)
        S_paths = np.zeros((dW.shape[0], nb_steps + 1))
        S_paths[:, 0] = S0
        S_paths[:, 1:] = S0 * np.exp(log_returns)
        return S_paths, dt_val
