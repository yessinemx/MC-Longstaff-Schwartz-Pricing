import numpy as np
import datetime as dt
import math
from market import Market
from option import Option
from model import Model
from BrownianMotion import BrownianMotion 

class MonteCarloPricer(Model):
    def __init__(self, market: Market, option: Option, pricing_date: dt.date, nb_steps: int, nb_paths: int):
        super().__init__(pricing_date)
        self.market = market
        self.option = option
        self.nb_steps = int(nb_steps)
        self.nb_paths = int(nb_paths)
        self.bm = BrownianMotion(seed=42)

    def _generate_price_paths(self, antithetic: bool = True):
        """
        Construit les chemins de prix S_t à partir des dW générés par BrownianMotion.
        """
        # Calcul de la maturité en années
        T = (self.option.maturity - self.pricing_date).days / 365.0
        
        # 1. Appel au fichier externe pour obtenir les dW (vectoriel)
        dW, dt = self.bm.generate_dw(T, self.nb_steps, self.nb_paths, antithetic)
        
        # 2. Construction des chemins (GBM)
        drift_term = (self.market.r - self.market.dividend_yield - 0.5 * self.market.sigma**2) * dt
        diffusion_term = self.market.sigma * dW
        
        # Log-returns cumulés
        log_returns = np.cumsum(drift_term + diffusion_term, axis=1)
        
        # Initialisation de la matrice des prix (M simulations x N+1 pas)
        S_paths = np.zeros((dW.shape[0], self.nb_steps + 1))
        S_paths[:, 0] = self.market.S0
        
        # Application des returns : S_t = S_0 * exp(somme_returns)
        S_paths[:, 1:] = self.market.S0 * np.exp(log_returns)
        
        return S_paths, dt, T

    def price(self, antithetic: bool = True, method: str = "vectoriel") -> float:
        """Routeur principal : Européen ou Américain (Longstaff-Schwartz)."""
        if method == "scalaire":
            return self._price_scalar()
        elif method == "vectoriel":
            if self.option.option_class == "european":
                return self._price_european(antithetic)
            elif self.option.option_class == "american":
                return self._price_american_ls(antithetic)
            else:
                raise ValueError("Type d'option inconnu")
        else:
            raise ValueError("Méthode inconnue : 'scalaire' ou 'vectoriel'")

    def _price_european(self, antithetic: bool) -> float:
        S, _ , T = self._generate_price_paths(antithetic)
        payoffs = self.option.payoff(S[:, -1])

        return np.mean(payoffs) * np.exp(-self.market.r * T)

    def _price_american_ls(self, antithetic: bool) -> float:
        """Algorithme Longstaff-Schwartz vectoriel."""
        S, dt, _ = self._generate_price_paths(antithetic)
        discount_factor = np.exp(-self.market.r * dt)
        
        # 1. Initialisation à maturité
        cashflows = self.option.payoff(S[:, -1])
        
        # 2. Récurrence Backward (de T-1 à 1)
        for t in range(self.nb_steps - 1, 0, -1):
            cashflows = cashflows * discount_factor 
            S_t = S[:, t]
            intrinsic_val = self.option.payoff(S_t)
            
            # On ne considère que les chemins "In The Money" pour la régression
            itm = intrinsic_val > 0
            if np.count_nonzero(itm) > 0:
                X = S_t[itm]
                Y = cashflows[itm]
                
                # Régression Quadratique : E[Hold] ~ a + bS + cS^2
                coeffs = np.polyfit(X, Y, 2)
                continuation_val = np.polyval(coeffs, X)
                
                # Décision d'exercice
                exercise = intrinsic_val[itm] > continuation_val
                
                # Mise à jour des cashflows là où on exerce
                updated_cf = cashflows.copy()
                updated_cf[itm] = np.where(exercise, intrinsic_val[itm], Y)
                cashflows = updated_cf
                
        # 3. Actualisation finale de t=1 à t=0
        return np.mean(cashflows * discount_factor)

    def _price_scalar(self) -> float:
        """
        Pricing Européen méthode SCALAIRE.
        Boucle explicite sur les chemins et les pas de temps.
        Supporte uniquement les options européennes pour l'instant.
        """
        if self.option.option_class != "european":
            raise ValueError("Méthode scalaire supporte uniquement les options européennes")

        # Calcul du temps jusqu'à maturité en années
        T = (self.option.maturity - self.pricing_date).days / 365.0
        dt = T / self.nb_steps
        discount_factor = math.exp(-self.market.r * dt)

        # Initialisation
        total_payoff = 0.0

        for path in range(self.nb_paths):

            # Prix initial
            S = self.market.S0

            # Simulation du chemin
            for step in range(self.nb_steps):

                # Génération du mouvement brownien
                Z = self.bm.next_gaussian()
                dW = Z * math.sqrt(dt)

                # Actualisation du prix selon Black-Scholes
                S *= math.exp((self.market.r - 0.5 * self.market.sigma**2) * dt + self.market.sigma * dW)

            # Calcul du payoff à maturité
            payoff = self.option.payoff(S)
            total_payoff += payoff

        # Moyenne des payoffs actualisés
        return (total_payoff / self.nb_paths) * (discount_factor ** self.nb_steps)