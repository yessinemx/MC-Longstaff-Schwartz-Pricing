import datetime as dt

class Market:
    """Conditions de marché : spot, taux, volatilité et dividende."""

    def __init__(
        self,
        S0: float,
        r: float,
        sigma: float,
        dividend: float = 0.0,
        dividend_date: dt.date | None = None,
    ) -> None:
        self.S0 = float(S0)
        self.r = float(r)
        self.sigma = float(sigma)
        self.dividend = float(dividend)
        self.dividend_date = dividend_date

    @property
    def dividend_yield(self) -> float:
        """
        Convertit le dividende montant en rendement continu (approx) 
        pour la formule de diffusion (r - q).
        """
        if self.S0 == 0:
            return 0.0
        return self.dividend / self.S0