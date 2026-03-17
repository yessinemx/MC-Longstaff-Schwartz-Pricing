from abc import ABC, abstractmethod
import datetime as dt

class Model(ABC):
    """
    Classe abstraite qui définit l'interface commune à tous les modèles de pricing 
    (Monte Carlo, Black-Scholes, Arbre Binomial, etc.).
    """
    
    def __init__(self, pricing_date: dt.date):
        self.pricing_date = pricing_date

    @abstractmethod
    def price(self) -> float:
        """Tout modèle doit implémenter une méthode price."""
        pass