import datetime as dt


class Market:
    """
    Classe représentant les conditions de marché pour le pricing d'options.

    Cette classe encapsule tous les paramètres de marché nécessaires pour
    construire l'arbre trinomial et pricer une option:
    - Prix actuel du sous-jacent
    - Taux d'intérêt sans risque
    - Volatilité implicite
    - Dividendes
    """

    def __init__(self, stock_price, int_rate, sigma, div=0.0, div_date=None):
        """
        Initialise un objet Market avec les paramètres de marché.

        Paramètres:
        -----------
        stock_price : float
            Prix actuel du sous-jacent (S0)
        int_rate : float
            Taux d'intérêt sans risque
        sigma : float
            Volatilité implicite du sous-jacent
        div : float
            Montant du dividende à détacher (discrete)
        div_date : date
            Date de détachement du dividende
        """
        # ===== Validation =====
        if stock_price <= 0:
            raise ValueError(f"stock_price doit être > 0 (reçu : {stock_price}).")
        if sigma < 0:
            raise ValueError(f"sigma doit être >= 0 (reçu : {sigma}).")
        if div < 0:
            raise ValueError(f"div doit être >= 0 (reçu : {div}).")

        # ===== Prix du sous-jacent =====
        self.stock_price = float(stock_price)

        # ===== Paramètres de marché =====
        self.int_rate = float(int_rate)
        self.sigma = float(sigma)

        # ===== Informations sur les dividendes =====
        self.div = float(div)
        self.div_date = div_date
