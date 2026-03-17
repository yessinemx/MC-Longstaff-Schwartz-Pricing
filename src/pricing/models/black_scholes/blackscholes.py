from __future__ import annotations
import datetime as dt
from typing import Literal, Optional

import numpy as np
from scipy.stats import norm


class BlackScholesPricer:
    """
    Pricer Black–Scholes pour options européennes (call/put),
    avec possibilité d’un dividende discret unique.

    Parameters
    ----------
    S : float
        Prix spot du sous-jacent.
    K : float
        Strike de l’option.
    T : float
        Maturité (en années).
    r : float
        Taux sans risque.
    sigma : float
        Volatilité annuelle.
    option_type : {'call', 'put'}
        Type d’option.
    dividend : float, optional
        Montant du dividende (par défaut 0).
    dividend_date : datetime, optional
        Date du dividende si applicable.
    """

    def __init__(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: Literal["call", "put"],
        dividend: float = 0.0,
        dividend_date: Optional[dt.date] = None,
    ) -> None:
        """Initialise le modèle Black–Scholes."""
        self.S: float = S
        self.K: float = K
        self.T: float = T
        self.r: float = r
        self.sigma: float = sigma
        self.option_type: Literal["call", "put"] = option_type.lower()  # type: ignore
        self.dividend: float = float(dividend)
        self.dividend_date: Optional[dt.date] = dividend_date
        self._N, self._n = norm.cdf, norm.pdf

    # ------------------- utilitaires -------------------
    def _spot_adjusted(self) -> float:
        """
        Calcule le spot ajusté en tenant compte d’un dividende discret.
        """
        S = self.S
        if self.dividend and self.dividend_date:
            tD = max((self.dividend_date - dt.date.today()).days / 365.0, 0.0)
            S -= self.dividend * np.exp(-self.r * tD)
        return S

    def _d1d2(self, S_adj: float) -> tuple[float, float]:
        """
        Renvoie les paramètres d1 et d2 du modèle Black–Scholes.
        """
        T = self.T
        sig = self.sigma
        rt = sig * np.sqrt(T)
        d1 = (np.log(S_adj / self.K) + (self.r + 0.5 * sig**2) * T) / rt
        d2 = d1 - rt
        return d1, d2

    # ------------------- pricing -------------------
    def price(self) -> float:
        """
        Calcule le prix théorique (call ou put) selon Black–Scholes.
        """
        S_adj = self._spot_adjusted()
        d1, d2 = self._d1d2(S_adj)
        discK = self.K * np.exp(-self.r * self.T)
        if self.option_type == "call":
            return S_adj * self._N(d1) - discK * self._N(d2)
        if self.option_type == "put":
            return discK * self._N(-d2) - S_adj * self._N(-d1)
        raise ValueError("option_type must be 'call' or 'put'")

    # ------------------- mise à jour -------------------
    def update(self, **kwargs) -> "BlackScholesPricer":
        """
        Met à jour un ou plusieurs paramètres du pricer.
        Ex : bs.update(K=105, sigma=0.25, option_type='put')
        """
        allowed = {"S", "K", "T", "r", "sigma", "option_type", "dividend", "dividend_date"}
        for k, v in kwargs.items():
            if k not in allowed:
                raise AttributeError(f"Unknown parameter: {k}")
            if k == "option_type":
                self.option_type = str(v).lower()  # type: ignore
            else:
                setattr(self, k, v)
        return self
