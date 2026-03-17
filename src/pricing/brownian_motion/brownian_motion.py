# Génération de mouvements browniens vectoriels
import numpy as np
import random
from scipy.stats import norm
from typing import Tuple

class BrownianMotion:

    def __init__(self, seed: int = 1234, method: str = "direct") -> None:
        """
        Parameters
        ----------
        seed : int
            Graine du générateur pseudo-aléatoire.
        method : str
            "uniform_ppf" -> uniform draws + norm.ppf
            "direct"      -> standard_normal 
        """
        self.seed = seed
        self.method = method.lower()
        # Initialisation du générateur aléatoire
        self.rng = np.random.default_rng(seed)
        random.seed(seed)

    def generate_dw(
        self,
        T: float,
        nb_steps: int,
        nb_paths: int,
        antithetic: bool = True,
    ) -> Tuple[np.ndarray, float]:
        """Génère une matrice d'incréments Browniens (dW)"""
        dt = T / nb_steps
        sqrt_dt = np.sqrt(dt)
        actual_draws = nb_paths // 2 if antithetic else nb_paths
        # Tirage Normal Standard Z ~ N(0, 1)
        Z = self._draw_normal(actual_draws, nb_steps)
        # Gestion antithétique et conversion en dW
        if antithetic:
            Z = self._apply_antithetic(Z, nb_paths, nb_steps)
        dW = Z * sqrt_dt
        return dW, dt

    def _draw_normal(self, nb_rows: int, nb_cols: int) -> np.ndarray:
        """Tire des N(0,1) selon la méthode choisie"""
        if self.method == "uniform_ppf":
            U = self.rng.uniform(0.0, 1.0, (nb_rows, nb_cols))
            U = np.clip(U, 1e-10, 1.0 - 1e-10)
            return norm.ppf(U)
        return self.rng.standard_normal((nb_rows, nb_cols))

    def _apply_antithetic(
        self, Z: np.ndarray, nb_paths: int, nb_steps: int
    ) -> np.ndarray:
        """Applique la réduction de variance antithétique"""
        Z = np.concatenate([Z, -Z], axis=0)
        # Si nb_paths impair, compléter avec des tirages supplémentaires
        if Z.shape[0] < nb_paths:
            extra = self._draw_normal(nb_paths - Z.shape[0], nb_steps)
            Z = np.concatenate([Z, extra], axis=0)
        return Z

    def next_gaussian(self) -> float:
        """
        Version Scalaire : Retourne UN seul tirage N(0,1)
        """
        if self.method == "uniform_ppf":
            u = self.rng.uniform(0.0, 1.0)
            u = np.clip(u, 1e-10, 1.0 - 1e-10)
            return float(norm.ppf(u))
        else:
            return float(self.rng.standard_normal())
