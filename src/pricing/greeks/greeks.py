"""
Calcul des grecs par différences finies (bump-and-reprice).

Fonctionne avec n'importe quel pricer : il suffit de passer une callable
qui renvoie le prix et l'objet Market dont on bumpera les paramètres.

Exemple d'utilisation avec le pricer LSM :
    >>> from pricing.greeks import GreeksCalculator
    >>> calc = GreeksCalculator(pricer.price, pricer.market)
    >>> calc.delta()
    >>> calc.greeks()            # dict complet
"""

from __future__ import annotations

import datetime as dt
from typing import Callable, Dict, Optional


class GreeksCalculator:
    """
    Calcule les grecs via différences finies sur un pricer quelconque.

    Parameters
    ----------
    price_fn : Callable[[], float]
        Fonction sans argument qui renvoie le prix de l'option
        en utilisant les paramètres courants du marché.
        Exemple : ``pricer.price``, ``lambda: tree.price(option, build_tree=True)``
    market : Market
        Objet Market dont les attributs (stock_price, sigma, int_rate)
        seront temporairement bumpés pour le calcul des dérivées.
    pricer : object, optional
        Objet possédant un attribut ``pricing_date`` (datetime.date).
        Nécessaire pour le calcul de theta (passage du temps).
    """

    def __init__(self, price_fn: Callable[[], float], market, pricer=None) -> None:
        self.price_fn = price_fn
        self.market = market
        self.pricer = pricer

    # ------------------------------------------------------------------ #
    # Grecs individuels
    # ------------------------------------------------------------------ #

    def delta(self, bump: float = 0.01) -> float:
        """Δ = ∂V/∂S  (bump relatif sur stock_price)."""
        S0 = self.market.stock_price
        self.market.stock_price = S0 * (1 + bump)
        up = self.price_fn()
        self.market.stock_price = S0 * (1 - bump)
        down = self.price_fn()
        self.market.stock_price = S0
        return (up - down) / (2 * S0 * bump)

    def gamma(self, bump: float = 0.01) -> float:
        """Γ = ∂²V/∂S²  (dérivée seconde centrale)."""
        S0 = self.market.stock_price
        v0 = self.price_fn()

        self.market.stock_price = S0 * (1 + bump)
        up = self.price_fn()
        self.market.stock_price = S0 * (1 - bump)
        down = self.price_fn()
        self.market.stock_price = S0

        dS = S0 * bump
        return (up - 2 * v0 + down) / (dS ** 2)

    def vega(self, bump: float = 0.01) -> float:
        """ν = ∂V/∂σ  (par 1 % de vol, bump relatif sur sigma)."""
        sigma0 = self.market.sigma
        self.market.sigma = sigma0 * (1 + bump)
        up = self.price_fn()
        self.market.sigma = sigma0 * (1 - bump)
        down = self.price_fn()
        self.market.sigma = sigma0
        return (up - down) / (2 * sigma0 * bump) / 100

    def theta(self, bump_days: float = 1.0) -> float:
        """Θ = −∂V/∂t  (par jour calendaire).

        Avance ``pricer.pricing_date`` de *bump_days* jours
        et recalcule le prix pour estimer la perte de valeur temps.

        Nécessite que le constructeur ait reçu un ``pricer`` ayant
        un attribut mutable ``pricing_date`` (datetime.date).
        """
        if self.pricer is None or not hasattr(self.pricer, "pricing_date"):
            raise RuntimeError(
                "theta nécessite un pricer avec pricing_date. "
                "Passez le pricer au constructeur : GreeksCalculator(fn, market, pricer=pricer)"
            )
        bump = dt.timedelta(days=int(bump_days))
        date0 = self.pricer.pricing_date

        v0 = self.price_fn()

        # Bump pricing_date sur le pricer et son path_generator s'il existe
        self.pricer.pricing_date = date0 + bump
        if hasattr(self.pricer, "path_gen"):
            self.pricer.path_gen.pricing_date = date0 + bump
        v_bumped = self.price_fn()

        # Restauration
        self.pricer.pricing_date = date0
        if hasattr(self.pricer, "path_gen"):
            self.pricer.path_gen.pricing_date = date0

        return -(v_bumped - v0) / bump_days

    def rho(self, bump: float = 0.0001) -> float:
        """ρ = ∂V/∂r  (par 1 % de taux, bump absolu sur int_rate)."""
        r0 = self.market.int_rate
        self.market.int_rate = r0 + bump
        up = self.price_fn()
        self.market.int_rate = r0 - bump
        down = self.price_fn()
        self.market.int_rate = r0
        return (up - down) / (2 * bump) / 100

    def vanna(self, bump_S: float = 0.01, bump_sigma: float = 0.01) -> float:
        """Vanna = ∂²V / (∂S ∂σ)  (dérivée croisée, par 1 % de vol)."""
        S0, sigma0 = self.market.stock_price, self.market.sigma

        self.market.stock_price = S0 * (1 + bump_S)
        self.market.sigma = sigma0 * (1 + bump_sigma)
        v_uu = self.price_fn()

        self.market.stock_price = S0 * (1 - bump_S)
        self.market.sigma = sigma0 * (1 - bump_sigma)
        v_dd = self.price_fn()

        self.market.stock_price = S0 * (1 + bump_S)
        self.market.sigma = sigma0 * (1 - bump_sigma)
        v_ud = self.price_fn()

        self.market.stock_price = S0 * (1 - bump_S)
        self.market.sigma = sigma0 * (1 + bump_sigma)
        v_du = self.price_fn()

        self.market.stock_price, self.market.sigma = S0, sigma0
        return (v_uu + v_dd - v_ud - v_du) / (4 * S0 * sigma0 * bump_S * bump_sigma) / 100

    # ------------------------------------------------------------------ #
    # Synthèse
    # ------------------------------------------------------------------ #

    def greeks(self, bump_S: float = 0.01, bump_sigma: float = 0.01,
               bump_r: float = 0.0001, bump_days: float = 1.0) -> Dict[str, float]:
        """Renvoie un dictionnaire avec tous les grecs calculés."""
        result = {
            "delta": self.delta(bump_S),
            "gamma": self.gamma(bump_S),
            "vega": self.vega(bump_sigma),
            "rho": self.rho(bump_r),
            "vanna": self.vanna(bump_S, bump_sigma),
        }
        if self.pricer is not None and hasattr(self.pricer, "pricing_date"):
            result["theta"] = self.theta(bump_days)
        return result
