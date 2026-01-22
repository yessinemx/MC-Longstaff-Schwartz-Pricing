import time
import datetime as dt
from market import Market
from option import Option
from model_mc import MonteCarloPricer
from blackscholes import BlackScholesPricer

def main():
    # Paramètres
    market = Market(S0=100, r=0.05, sigma=0.3, dividend=2)
    option = Option(K=102, maturity=dt.date(2026, 9, 1), option_type="call", option_class="european")
    pricing_date = dt.date(2025, 9, 1)
    
    nb_steps = 100
    nb_paths = 1000

    print(f"--- Comparaison Scalaire vs Vectoriel ({nb_paths} sims, {nb_steps} steps) ---")

    # 1. Test SCALAIRE
    pricer = MonteCarloPricer(market, option, pricing_date, nb_steps, nb_paths)
    start = time.time()
    res_scalar = pricer.price(method="scalaire")
    end = time.time()
    time_scalar = end - start
    print(f"Scalaire : {res_scalar:.4f} EUR | Temps : {time_scalar:.4f} sec")

    # 2. Test VECTORIEL
    start = time.time()
    res_vector = pricer.price(method="vectoriel", antithetic=False) 
    end = time.time()
    time_vector = end - start
    print(f"Vectoriel: {res_vector:.4f} EUR | Temps : {time_vector:.4f} sec")

    # 3. Test BLACK-SCHOLES (référence)
    T_years = (option.maturity - pricing_date).days / 365.0
    bs_pricer = BlackScholesPricer(
        S=market.S0,
        K=option.K,
        T=T_years,
        r=market.r,
        sigma=market.sigma,
        option_type=option.option_type,
        dividend=market.dividend
    )
    bs_price = bs_pricer.price()
    print(f"Black-Scholes: {bs_price:.4f} EUR")

    # Conclusion
    print(f"---")
    print(f"Accélération Vectorielle : x{time_scalar / time_vector:.1f}")

if __name__ == "__main__":
    main()