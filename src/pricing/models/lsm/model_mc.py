import numpy as np
import datetime as dt
from typing import Any, Optional, Tuple

from scipy.stats import norm

from ...parameters.market import Market
from ...parameters.option import Option
from ..model import Model
from ...brownian_motion.brownian_motion import BrownianMotion
from .regression import LSRegressor
from .path_generator import PathGenerator

# Mixins
from .pricing_mixins import (
    EuropeanPricerMixin,
    AmericanLSMixin,
    ScalarPricerMixin,
    BatchPricerMixin,
    ExerciseFrontierMixin,
    ConvergenceMixin,
)

# Fonctions statiques
from . import static_pricing as _sp


class MonteCarloPricer(
    EuropeanPricerMixin,
    AmericanLSMixin,
    ScalarPricerMixin,
    BatchPricerMixin,
    ExerciseFrontierMixin,
    ConvergenceMixin,
    Model,
):
    """Monte Carlo pricer (Longstaff-Schwartz) – classe façade."""

    def __init__(
        self,
        market: Market,
        option: Option,
        pricing_date: dt.date,
        nb_steps: int,
        nb_paths: int,
        seed: int = 42,
        batch_size: Optional[int] = None,
        rng_method: str = "direct"
    ) -> None:
        super().__init__(pricing_date)
        self.market, self.option = market, option
        self.nb_steps, self.nb_paths = int(nb_steps), int(nb_paths)
        self.seed, self.rng_method = int(seed), rng_method
        self.bm = BrownianMotion(seed=seed, method=rng_method)
        self.path_gen = PathGenerator(market, option, pricing_date, self.nb_steps, self.bm)
        self.regressor = LSRegressor()
        self.last_variance = 0.0
        self.last_std_error = 0.0
        self.batch_size: Optional[int] = int(batch_size) if batch_size else None

    # ------------------------------------------------------------------ #
    # Génération de chemins (délègue à PathGenerator)
    # ------------------------------------------------------------------ #
    def _generate_price_paths(
        self, antithetic: bool = True, nb_paths: Optional[int] = None
    ) -> Tuple[np.ndarray, float, float]:
        n = nb_paths if nb_paths is not None else self.nb_paths
        return self.path_gen.generate_price_paths(n, antithetic)

    # ------------------------------------------------------------------ #
    # Helpers partagés par les mixins
    # ------------------------------------------------------------------ #
    def _compute_dividend_step(self, T: float) -> Optional[int]:
        return self.path_gen.compute_dividend_step(T)

    def _compute_continuation_value(
        self, X: np.ndarray, Y: np.ndarray,
        regression_type: str, degree: int
    ) -> np.ndarray:
        return self.regressor.compute_continuation_value(X, Y, regression_type, degree)

    # ------------------------------------------------------------------ #
    # Routeur principal
    # ------------------------------------------------------------------ #
    def price(
        self,
        antithetic: bool = True,
        method: str = "vectoriel",
        regression_type: str = "quadratic",
        degree: int = 2,
        american_method: str = "ls",
        control_variate: bool = False
    ) -> float:
        """Main router: European or American (Longstaff-Schwartz or naive)."""
        if method == "scalaire":
            return self._price_scalar()
        elif method == "vectoriel":
            return self._price_vectoriel(
                antithetic, regression_type, degree, american_method, control_variate,
            )
        else:
            raise ValueError("Unknown method: use 'scalaire' or 'vectoriel'")

    def price_with_ci(
        self, confidence: float = 0.95, **kwargs: Any
    ) -> Tuple[float, float, float]:
        """Renvoie (price, lower, upper) avec un intervalle de confiance."""
        p = self.price(**kwargs)
        z = norm.ppf(1 - (1 - confidence) / 2)
        half = z * self.last_std_error
        return p, p - half, p + half

    def _price_vectoriel(
        self, antithetic: bool, regression_type: str,
        degree: int, american_method: str,
        control_variate: bool = False
    ) -> float:
        """Dispatch vectoriel : batch, européen ou américain."""
        use_batch = self.batch_size is not None and self.nb_paths > self.batch_size
        if use_batch:
            return self._price_batched(antithetic, regression_type, degree, american_method)
        if self.option.option_class == "european":
            if control_variate:
                return self._price_european_cv(antithetic)
            return self._price_european(antithetic)
        elif self.option.option_class == "american":
            if american_method == "naive":
                return self._price_american_naive(antithetic)
            return self._price_american_ls(antithetic, regression_type, degree)
        raise ValueError("Unknown option type")

    # ------------------------------------------------------------------ #
    # Méthodes statiques (délèguent à static_pricing)
    # ------------------------------------------------------------------ #
    @staticmethod
    def price_european_call(
        S: np.ndarray | float, K: float, r: float, T: float,
        sigma: float, q: float = 0.0
    ) -> float:
        return _sp.price_european_call(S, K, r, T, sigma, q)

    @staticmethod
    def price_european_put(
        S: np.ndarray | float, K: float, r: float, T: float,
        sigma: float = 0.0, q: float = 0.0
    ) -> float:
        return _sp.price_european_put(S, K, r, T, sigma, q)

    # Options américaines (statique)
    @staticmethod
    def price_american_put_ls(
        S_paths: np.ndarray, K: float, r: float, dt: float,
        regression_type: str = "quadratic", degree: int = 2
    ) -> float:
        return _sp.price_american_put_ls(S_paths, K, r, dt, regression_type, degree)

    @staticmethod
    def price_american_call_ls(
        S_paths: np.ndarray, K: float, r: float, dt: float,
        regression_type: str = "quadratic", degree: int = 2
    ) -> float:
        return _sp.price_american_call_ls(S_paths, K, r, dt, regression_type, degree)

    # Génération de chemins GBM (statique)
    @staticmethod
    def generate_gbm_paths(
        S0: float, r: float, sigma: float,
        T: float, nb_steps: int, nb_paths: int,
        q: float = 0.0, seed: int = 42,
        antithetic: bool = True
    ) -> Tuple[np.ndarray, float]:
        return _sp.generate_gbm_paths(
            S0, r, sigma, T, nb_steps, nb_paths, q, seed, antithetic,
        )