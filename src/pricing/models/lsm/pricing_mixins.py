"""Mixins de pricing pour MonteCarloPricer.

Regroupe toutes les stratégies de pricing en un seul fichier :
  - EuropeanPricerMixin  : MC européen vectorisé + variable de contrôle
  - AmericanLSMixin      : Longstaff-Schwartz vectorisé + méthode naïve
  - ScalarPricerMixin    : pricing chemin par chemin
  - BatchPricerMixin     : mini-batches avec parallélisme ThreadPool
  - ExerciseFrontierMixin: frontière d'exercice anticipé
  - ConvergenceMixin     : table de convergence
"""
import math
import numpy as np
from typing import Any, Optional, Tuple
from scipy.stats import norm


# ================================================================== #
#  European pricing (vectorized)
# ================================================================== #
class EuropeanPricerMixin:
    """European pricing methods: plain MC and control variate."""

    def _price_european(self, antithetic: bool, nb_paths: Optional[int] = None) -> float:
        S, _, T = self._generate_price_paths(antithetic, nb_paths=nb_paths)
        payoffs = self.option.payoff(S[:, -1])

        price = np.mean(payoffs) * np.exp(-self.market.int_rate * T)
        self.last_variance = float(np.var(payoffs, ddof=1))
        self.last_std_error = float(np.sqrt(self.last_variance / len(payoffs))) * np.exp(-self.market.int_rate * T)

        return price

    def _price_european_cv(self, antithetic: bool, nb_paths: Optional[int] = None) -> float:
        """Pricing européen avec variable de contrôle (prix BS analytique)."""
        S, _, T = self._generate_price_paths(antithetic, nb_paths=nb_paths)
        disc = np.exp(-self.market.int_rate * T)

        payoffs = self.option.payoff(S[:, -1])
        control = S[:, -1]  # S_T comme variable de contrôle
        control_mean = self.market.stock_price * np.exp(self.market.int_rate * T)  # E[S_T] sous Q

        # Régression pour trouver le coefficient beta optimal
        cov_xy = np.cov(payoffs, control)[0, 1]
        var_c = np.var(control, ddof=1)
        beta = cov_xy / var_c if var_c > 0 else 0.0

        # Payoffs ajustés
        adjusted = payoffs - beta * (control - control_mean)
        price = float(np.mean(adjusted)) * disc

        self.last_variance = float(np.var(adjusted, ddof=1))
        self.last_std_error = float(np.sqrt(self.last_variance / len(adjusted))) * disc
        return price

    def _bs_euro_price(self, T: float) -> float:
        """Prix BS analytique pour l'option courante (call/put européen)."""
        S = self.market.stock_price
        K = self.option.K
        r = self.market.int_rate
        sigma = self.market.sigma
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if self.option.option_type == "call":
            return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
        return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


# ================================================================== #
#  American pricing – Longstaff-Schwartz + naive
# ================================================================== #
class AmericanLSMixin:
    """American pricing: Longstaff-Schwartz backward recursion and naive method."""

    def _price_american_ls(
        self,
        antithetic: bool,
        regression_type: str = "quadratic",
        degree: int = 2,
        nb_paths: Optional[int] = None
    ) -> float:
        """Longstaff-Schwartz algorithm for American options (vectorized)."""
        S, dt_val, T = self._generate_price_paths(antithetic, nb_paths=nb_paths)
        discount_factor = np.exp(-self.market.int_rate * dt_val)

        cashflows = self.option.payoff(S[:, -1])
        cashflows = self._ls_backward_recursion(
            S, cashflows, discount_factor, regression_type, degree
        )
        return self._ls_final_price(cashflows, discount_factor)

    def _ls_backward_recursion(
        self, S: np.ndarray, cashflows: np.ndarray,
        discount_factor: float, regression_type: str, degree: int
    ) -> np.ndarray:
        """Backward recursion (from T-1 to 1) for Longstaff-Schwartz."""
        for t in range(self.nb_steps - 1, 0, -1):
            cashflows = cashflows * discount_factor
            S_t = S[:, t]
            intrinsic_val = self.option.payoff(S_t)
            itm = intrinsic_val > 0
            if np.count_nonzero(itm) > 0:
                cashflows = self._ls_update_cashflows(
                    cashflows, S_t, intrinsic_val, itm, regression_type, degree
                )
        return cashflows

    def _ls_update_cashflows(
        self, cashflows: np.ndarray, S_t: np.ndarray,
        intrinsic_val: np.ndarray, itm: np.ndarray,
        regression_type: str, degree: int
    ) -> np.ndarray:
        """Update cashflows based on exercise decision at one time step."""
        X, Y = S_t[itm], cashflows[itm]
        continuation_val = self._compute_continuation_value(
            X, Y, regression_type, degree
        )
        exercise = intrinsic_val[itm] > continuation_val
        updated_cf = np.array(cashflows)
        updated_cf[itm] = np.where(exercise, intrinsic_val[itm], Y)
        return updated_cf

    def _ls_final_price(self, cashflows: np.ndarray, discount_factor: float) -> float:
        """Final discounting and comparison with immediate exercise."""
        final_payoffs = cashflows * discount_factor
        price = np.mean(final_payoffs)
        self.last_variance = np.var(final_payoffs, ddof=1)
        self.last_std_error = np.sqrt(self.last_variance / len(final_payoffs))
        immediate_exercise = self.option.payoff(self.market.stock_price)
        return max(price, immediate_exercise)

    def _price_american_naive(self, antithetic: bool, nb_paths: Optional[int] = None) -> float:
        """Naive American option pricing (uses hindsight, biased high)."""
        S, dt_val, T = self._generate_price_paths(antithetic, nb_paths=nb_paths)
        n_sim, nb_steps_plus_one = S.shape

        payoffs = np.zeros_like(S)
        for t in range(nb_steps_plus_one):
            payoffs[:, t] = self.option.payoff(S[:, t])

        time_steps = np.arange(nb_steps_plus_one) * dt_val
        discount_factors = np.exp(-self.market.int_rate * time_steps)
        discounted_payoffs = payoffs * discount_factors

        max_discounted_payoffs = np.max(discounted_payoffs, axis=1)
        return np.mean(max_discounted_payoffs)


# ================================================================== #
#  Scalar pricing (path-by-path loop)
# ================================================================== #
class ScalarPricerMixin:
    """Scalar pricing: loop over individual paths (European + American)."""

    def _price_scalar(self) -> float:
        """Scalar method for European AND American option pricing."""
        T = (self.option.maturity - self.pricing_date).days / 365.0
        dt_val = T / self.nb_steps
        sqrt_dt = math.sqrt(dt_val)
        discount_factor_step = math.exp(-self.market.int_rate * dt_val)
        div_step = self._compute_dividend_step(T)

        Z = self.path_gen.generate_random_matrix(self.nb_paths, self.rng_method)

        if self.option.option_class == "european":
            return self._scalar_european(Z, dt_val, sqrt_dt, div_step)
        elif self.option.option_class == "american":
            return self._scalar_american(Z, dt_val, sqrt_dt, discount_factor_step, div_step)
        else:
            raise ValueError(f"Unknown option class: {self.option.option_class}")

    def _scalar_european(
        self, Z: np.ndarray, dt_val: float,
        sqrt_dt: float, div_step: Optional[int]
    ) -> float:
        """Pricing européen scalaire par moyenne des payoffs."""
        total_payoff = 0.0
        for path in range(self.nb_paths):
            S = self.path_gen.simulate_single_path(Z[path], dt_val, sqrt_dt, div_step)
            total_payoff += self.option.payoff(S)
        T_val = (self.option.maturity - self.pricing_date).days / 365.0
        discount_total = math.exp(-self.market.int_rate * T_val)
        return (total_payoff / self.nb_paths) * discount_total

    def _scalar_american(
        self, Z: np.ndarray, dt_val: float, sqrt_dt: float,
        discount_factor_step: float, div_step: Optional[int]
    ) -> float:
        """Pricing américain scalaire via Longstaff-Schwartz."""
        S_paths = self.path_gen.scalar_generate_paths(
            Z, self.nb_paths, dt_val, sqrt_dt, div_step
        )

        cashflows = np.array(self.option.payoff(S_paths[:, -1]))
        cashflows = self._ls_backward_recursion_scalar(
            S_paths, cashflows, discount_factor_step
        )
        price = float(np.mean(cashflows * discount_factor_step))
        immediate = self.option.payoff(self.market.stock_price)
        return max(price, immediate)

    def _ls_backward_recursion_scalar(
        self, S_paths: np.ndarray, cashflows: np.ndarray,
        discount_factor_step: float
    ) -> np.ndarray:
        """Backward recursion LS pour le pricing scalaire."""
        for t in range(self.nb_steps - 1, 0, -1):
            cashflows = cashflows * discount_factor_step
            S_t = S_paths[:, t]
            intrinsic_val = self.option.payoff(S_t)
            itm = intrinsic_val > 0
            if np.count_nonzero(itm) > 0:
                X, Y = S_t[itm], cashflows[itm]
                continuation_val = self._compute_continuation_value(
                    X, Y, "quadratic", 2
                )
                exercise = intrinsic_val[itm] > continuation_val
                cashflows[itm] = np.where(exercise, intrinsic_val[itm], Y)
        return cashflows


# ================================================================== #
#  Batch / parallel pricing
# ================================================================== #
class BatchPricerMixin:
    """Batch pricing with optional parallelism via ThreadPoolExecutor."""

    def _price_batched(
        self, antithetic: bool, regression_type: str,
        degree: int, american_method: str
    ) -> float:
        """Découpe nb_paths en mini-batches pour limiter la mémoire."""
        bs = self.batch_size
        total = self.nb_paths
        nb_batches = math.ceil(total / bs)
        batch_prices = self._run_batches(
            nb_batches, bs, total, antithetic,
            regression_type, degree, american_method
        )
        return self._aggregate_batch_results(batch_prices)

    def _run_batches(
        self, nb_batches: int, bs: int, total: int,
        antithetic: bool, regression_type: str,
        degree: int, american_method: str
    ) -> list:
        """Exécute le pricing pour chaque batch (parallèle si possible)."""
        from concurrent.futures import ThreadPoolExecutor
        import os

        batch_specs = []
        remaining = total
        for k in range(nb_batches):
            n_k = min(bs, remaining)
            remaining -= n_k
            batch_specs.append((k, n_k))

        # Exécution parallèle si plusieurs workers disponibles
        n_workers = min(nb_batches, os.cpu_count() or 1)
        if n_workers > 1:
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = [
                    pool.submit(
                        self._price_single_batch,
                        k, n_k, antithetic,
                        regression_type, degree, american_method,
                    )
                    for k, n_k in batch_specs
                ]
                return [f.result() for f in futures]

        return [
            self._price_single_batch(
                k, n_k, antithetic,
                regression_type, degree, american_method,
            )
            for k, n_k in batch_specs
        ]

    def _price_single_batch(
        self, k: int, n_k: int, antithetic: bool,
        regression_type: str, degree: int, american_method: str
    ) -> float:
        """Price un seul batch avec un seed spécifique."""
        from ...brownian_motion.brownian_motion import BrownianMotion

        bm_orig = self.bm
        new_bm = BrownianMotion(seed=self.seed + k, method=self.rng_method)
        self.bm = new_bm
        self.path_gen.bm = new_bm
        try:
            if self.option.option_class == "european":
                return self._price_european(antithetic, nb_paths=n_k)
            elif self.option.option_class == "american":
                if american_method == "naive":
                    return self._price_american_naive(antithetic, nb_paths=n_k)
                return self._price_american_ls(
                    antithetic, regression_type, degree, nb_paths=n_k
                )
            raise ValueError("Unknown option type")
        finally:
            self.bm = bm_orig
            self.path_gen.bm = bm_orig

    def _aggregate_batch_results(self, batch_prices: list) -> float:
        """Agrège les prix des batches en prix final."""
        arr = np.array(batch_prices)
        price = float(np.mean(arr))
        if len(arr) > 1:
            self.last_variance = float(np.var(arr, ddof=1))
            self.last_std_error = float(np.sqrt(self.last_variance / len(arr)))
        return price


# ================================================================== #
#  Exercise frontier
# ================================================================== #
class ExerciseFrontierMixin:
    """Exercise frontier via Longstaff-Schwartz backward recursion."""

    def get_exercise_frontier(
        self, antithetic: bool = True,
        regression_type: str = "quadratic", degree: int = 2
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the exercise frontier for American options using L&S."""
        S, dt_val, T = self._generate_price_paths(antithetic)
        discount_factor = np.exp(-self.market.int_rate * dt_val)

        times = np.linspace(0, T, self.nb_steps + 1)
        frontier_prices = np.full(self.nb_steps + 1, np.nan)
        frontier_prices[-1] = self.option.K

        cashflows = self.option.payoff(S[:, -1])
        cashflows, frontier_prices = self._frontier_backward(
            S, cashflows, discount_factor, frontier_prices,
            regression_type, degree,
        )
        return times, frontier_prices

    def _frontier_backward(
        self, S: np.ndarray, cashflows: np.ndarray,
        discount_factor: float, frontier_prices: np.ndarray,
        regression_type: str, degree: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Backward recursion to find exercise frontier."""
        for t in range(self.nb_steps - 1, 0, -1):
            cashflows = cashflows * discount_factor
            S_t = S[:, t]
            intrinsic_val = self.option.payoff(S_t)
            itm = intrinsic_val > 0
            if np.count_nonzero(itm) > 0:
                cashflows = self._frontier_step(
                    cashflows, S_t, intrinsic_val, itm,
                    frontier_prices, t, regression_type, degree,
                )
        return cashflows, frontier_prices

    def _frontier_step(
        self, cashflows: np.ndarray, S_t: np.ndarray,
        intrinsic_val: np.ndarray, itm: np.ndarray,
        frontier_prices: np.ndarray, t: int,
        regression_type: str, degree: int
    ) -> np.ndarray:
        """Un pas de la frontière : régression, frontière, cashflows."""
        X, Y = S_t[itm], cashflows[itm]
        continuation_val = self._compute_continuation_value(
            X, Y, regression_type, degree
        )
        exercise = intrinsic_val[itm] > continuation_val
        self._update_frontier(frontier_prices, t, X, exercise)
        updated_cf = np.array(cashflows)
        updated_cf[itm] = np.where(exercise, intrinsic_val[itm], Y)
        return updated_cf

    def _update_frontier(
        self, frontier_prices: np.ndarray, t: int,
        X: np.ndarray, exercise: np.ndarray
    ) -> None:
        """Met à jour la frontière d'exercice au pas t."""
        if np.any(exercise) and np.any(~exercise):
            exercising_prices = X[exercise]
            if self.option.option_type == "put":
                frontier_prices[t] = np.max(exercising_prices)
            else:
                frontier_prices[t] = np.min(exercising_prices)


# ================================================================== #
#  Convergence diagnostics
# ================================================================== #
class ConvergenceMixin:
    """Convergence table: price for varying nb_paths."""

    def convergence_table(
        self,
        path_counts: Optional[list] = None,
        confidence: float = 0.95,
        **kwargs: Any
    ) -> list:
        """Calcule le prix pour différents nb_paths et renvoie une table de convergence.

        Parameters
        ----------
        path_counts : list[int], optional
            Liste de nombres de chemins à tester.
            Par défaut : [500, 1000, 2000, 5000, 10000, 20000, 50000].
        confidence : float
            Niveau de confiance pour l'intervalle.

        Returns
        -------
        list[dict]
            Chaque entrée contient nb_paths, price, std_error, ci_lower, ci_upper.
        """
        if path_counts is None:
            path_counts = [500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000]
        original = self.nb_paths
        results = []
        z = norm.ppf(1 - (1 - confidence) / 2)
        for n in path_counts:
            self.nb_paths = n
            p = self.price(**kwargs)
            half = z * self.last_std_error
            results.append({
                "nb_paths": n,
                "price": p,
                "std_error": self.last_std_error,
                "ci_lower": p - half,
                "ci_upper": p + half,
            })
        self.nb_paths = original
        return results
