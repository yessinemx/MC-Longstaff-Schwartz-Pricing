"""
This module demonstrates:
1. European option pricing (scalar vs vectorial)
2. American option pricing using Longstaff-Schwartz algorithm
3. Static pricing methods with pre-generated paths
4. Comparison of different regression bases
5. Convergence study (PDF page 6)
"""

# Imports standards
import time
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from parameters.market import Market
from parameters.option import Option
from models.lsm.model_mc import MonteCarloPricer
from models.lsm.regression import RegressionType
from models.black_scholes.blackscholes import BlackScholesPricer


# Paramètres communs pour les exemples européens et américains
def _build_eu_params() -> tuple:
    """Construit les objets marché/option/date pour l'exemple européen."""
    market = Market(stock_price=100, int_rate=0.1, sigma=0.2, div=3, div_date=dt.date(2026, 5, 29))
    option = Option(K=100, maturity=dt.date(2026, 12, 26), option_type="call", option_class="american")
    pricing_date = dt.date(2026, 3, 1)
    return market, option, pricing_date


def _print_header(title: str) -> None:
    """Affiche un en-tête formaté."""
    print("=" * 60)
    print(title)
    print("=" * 60)


def _price_method(
    market: Market, option: Option, pricing_date: dt.date,
    nb_steps: int, nb_paths: int, seed: int,
    method: str, antithetic: bool, label: str
) -> float:
    """Price une option et affiche le résultat avec le temps."""
    pricer = MonteCarloPricer(market, option, pricing_date, nb_steps, nb_paths, seed=seed)
    start = time.time()
    # Pricing selon la méthode choisie
    if method == "scalaire":
        result = pricer.price(method="scalaire")
    else:
        result = pricer.price(method="vectoriel", antithetic=antithetic)
    elapsed = time.time() - start
    print(f"{label}: {result:.4f} EUR | Time: {elapsed:.4f} sec")
    return elapsed


def example_european_pricing(seed: int = 42) -> None:
    """Compare scalar vs vectorial European option pricing."""
    _print_header("EXAMPLE 1: European Option Pricing (Scalar vs Vectorial)")
    # Création des paramètres
    market, option, pricing_date = _build_eu_params()
    nb_steps, nb_paths = 100, 10000
    _print_params(market, option, nb_paths, nb_steps, seed)
    # Pricing scalaire, vectoriel, antithétique
    time_scalar = _price_method(market, option, pricing_date, nb_steps, nb_paths, seed, "scalaire", False, "Scalar Method   ")
    time_vector = _price_method(market, option, pricing_date, nb_steps, nb_paths, seed, "vectoriel", False, "Vectorial Method")
    _price_method(market, option, pricing_date, nb_steps, nb_paths, seed, "vectoriel", True, "Antithetic      ")
    # Référence Black-Scholes et speedup
    _print_bs_ref(market, option, pricing_date)
    _print_speedup(time_scalar, time_vector)


def _print_params(market: Market, option: Option, nb_paths: int, nb_steps: int, seed: int) -> None:
    """Affiche les paramètres de simulation."""
    print(f"\nParameters:")
    print(f"  S0={market.stock_price}, K={option.K}, r={market.int_rate}, sigma={market.sigma}")
    print(f"  Paths={nb_paths}, Steps={nb_steps}, Seed={seed}")
    print()


def _print_bs_ref(market: Market, option: Option, pricing_date: dt.date) -> None:
    """Affiche le prix Black-Scholes de référence."""
    # Calcul de la maturité en années
    T_years = (option.maturity - pricing_date).days / 365.0
    bs_pricer = BlackScholesPricer(
        S=market.stock_price, K=option.K, T=T_years, r=market.int_rate,
        sigma=market.sigma, option_type=option.option_type, dividend=market.div
    )
    bs_price = bs_pricer.price()
    print(f"Black-Scholes:    {bs_price:.4f} EUR (analytical)")


def _print_speedup(time_scalar: float, time_vector: float) -> None:
    """Affiche le facteur de speedup."""
    if time_vector > 0:
        print(f"\nSpeedup (Vectorial vs Scalar): x{time_scalar / time_vector:.1f}")
    else:
        print(f"\nSpeedup: Vectorial too fast to measure")


def example_american_put_ls(seed: int = 42) -> None:
    """American put option pricing using Longstaff-Schwartz algorithm."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: American Put Option - Longstaff-Schwartz")
    print("=" * 60)
    # Paramètres de marché et option
    market, option, pricing_date = _build_american_params()
    nb_steps, nb_paths = 100, 10000
    _print_params(market, option, nb_paths, nb_steps, seed)
    # Pricing LS avec différentes bases de régression
    _run_regression_types(market, option, pricing_date, nb_steps, nb_paths, seed)
    # Comparaison avec méthode naïve et européenne
    _run_naive_and_eu(market, option, pricing_date, nb_steps, nb_paths, seed)


def _build_american_params() -> tuple:
    """Construit les objets marché/option/date pour l'exemple américain."""
    market = Market(stock_price=100, int_rate=0.1, sigma=0.2, div=3, div_date=dt.date(2026, 5, 29))
    option = Option(K=100, maturity=dt.date(2026, 12, 26), option_type="put", option_class="american")
    pricing_date = dt.date(2026, 3, 1)
    return market, option, pricing_date


# Pricing LS avec différentes bases de régression
def _run_regression_types(
    market: Market, option: Option, pricing_date: dt.date,
    nb_steps: int, nb_paths: int, seed: int
) -> None:
    """Teste les différentes bases de régression LS."""
    regression_types = [
        RegressionType.QUADRATIC, RegressionType.LAGUERRE,
        RegressionType.HERMITE, RegressionType.LEGENDRE,
    ]
    print("Regression Type     | Price    | Time")
    print("-" * 45)
    for reg_type in regression_types:
        # Nouveau pricer pour chaque type (même seed)
        pricer = MonteCarloPricer(market, option, pricing_date, nb_steps, nb_paths, seed=seed)
        start = time.time()
        price = pricer.price(method="vectoriel", regression_type=reg_type.value, antithetic=True)
        elapsed = time.time() - start
        print(f"{reg_type.value:18s} | {price:8.4f} | {elapsed:.3f} sec")


def _run_naive_and_eu(
    market: Market, option: Option, pricing_date: dt.date,
    nb_steps: int, nb_paths: int, seed: int
) -> None:
    """Compare avec la méthode naïve et le put européen."""
    # Méthode naïve (biase haute)
    pricer_naive = MonteCarloPricer(market, option, pricing_date, nb_steps, nb_paths, seed=seed)
    start = time.time()
    naive_price = pricer_naive.price(method="vectoriel", american_method="naive", antithetic=True)
    elapsed = time.time() - start
    print(f"{'Naive (wrong!)':18s} | {naive_price:8.4f} | {elapsed:.3f} sec")
    # Put européen pour comparaison (borne inférieure)
    option_eu = Option(K=100, maturity=dt.date(2026, 12, 26), option_type="put", option_class="american")
    pricer_eu = MonteCarloPricer(market, option_eu, pricing_date, nb_steps, nb_paths, seed=seed)
    eu_price = pricer_eu.price(method="vectoriel", antithetic=True)
    print(f"\nEuropean Put (MC): {eu_price:.4f} EUR")
    print("Note: American put >= European put (early exercise premium)")


def example_static_methods(seed: int = 42) -> None:
    """Using static methods for pricing with pre-generated paths."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Static Pricing Methods")
    print("=" * 60)
    # Paramètres de simulation
    S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.25, 1.0
    nb_steps, nb_paths = 100, 5000000
    print(f"\nParameters:")
    print(f"  S0={S0}, K={K}, r={r}, sigma={sigma}, T={T}")
    print(f"  Paths={nb_paths}, Steps={nb_steps}, Seed={seed}")
    # Génération des chemins et pricing
    S_paths, dt_step = _generate_static_paths(S0, r, sigma, T, nb_steps, nb_paths, seed)
    _static_european_prices(S_paths, K, r, T, sigma, S0)
    _static_american_price(S_paths, K, r, dt_step)


def _generate_static_paths(
    S0: float, r: float, sigma: float,
    T: float, nb_steps: int, nb_paths: int, seed: int
) -> tuple:
    """Génère les chemins GBM pour le pricing statique."""
    print()
    print("Generating GBM paths...")
    # Appel à la méthode statique de génération
    S_paths, dt_step = MonteCarloPricer.generate_gbm_paths(
        S0=S0, r=r, sigma=sigma, T=T,
        nb_steps=nb_steps, nb_paths=nb_paths,
        seed=seed, antithetic=True
    )
    print(f"  Path shape: {S_paths.shape} (paths x steps)")
    print()
    return S_paths, dt_step


# Pricing européen statique avec référence BS
def _static_european_prices(
    S_paths: np.ndarray, K: float, r: float,
    T: float, sigma: float, S0: float
) -> None:
    """Affiche les prix européens statiques et la référence BS."""
    S_T = S_paths[:, -1]
    call_price = MonteCarloPricer.price_european_call(S_T, K, r, T, sigma)
    put_price = MonteCarloPricer.price_european_put(S_T, K, r, T, sigma)
    print(f"European Call (static): {call_price:.4f}")
    print(f"European Put (static):  {put_price:.4f}")
    # Référence Black-Scholes pour comparaison
    bs_call = BlackScholesPricer(S=S0, K=K, T=T, r=r, sigma=sigma, option_type="call").price()
    bs_put = BlackScholesPricer(S=S0, K=K, T=T, r=r, sigma=sigma, option_type="put").price()
    print(f"BS Call:                {bs_call:.4f}")
    print(f"BS Put:                 {bs_put:.4f}")


def _static_american_price(S_paths: np.ndarray, K: float, r: float, dt_step: float) -> None:
    """Pricing américain statique avec Laguerre degré 3."""
    print()
    # Put américain via méthode statique LS
    am_put_price = MonteCarloPricer.price_american_put_ls(
        S_paths=S_paths, K=K, r=r, dt=dt_step,
        regression_type="laguerre", degree=3
    )
    print(f"American Put (static):  {am_put_price:.4f}")


def example_exercise_frontier(seed: int = 42) -> None:
    """Compute and display the exercise frontier for American put."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Exercise Frontier (American Put)")
    print("=" * 60)
    # Paramètres de simulation
    market = Market(stock_price=100, int_rate=0.05, sigma=0.25, div=0, div_date=None)
    option = Option(K=100, maturity=dt.date(2026, 9, 1), option_type="put", option_class="american")
    pricing_date = dt.date(2025, 9, 1)
    nb_steps, nb_paths = 12, 50000
    # Calcul et affichage de la frontière
    pricer = MonteCarloPricer(market, option, pricing_date, nb_steps, nb_paths, seed=seed)
    times, frontier = pricer.get_exercise_frontier()
    _display_frontier(option, frontier, pricing_date, nb_steps)


def _display_frontier(
    option: Option, frontier: np.ndarray,
    pricing_date: dt.date, nb_steps: int
) -> None:
    """Affiche la frontière d'exercice pas par pas."""
    T_years = (option.maturity - pricing_date).days / 365.0
    dt_step = T_years / nb_steps
    print(f"\nExercise Frontier (K={option.K}):")
    print("Time Step | Critical Stock Price")
    print("-" * 35)
    # Affichage de chaque pas de temps
    for i in range(1, len(frontier)):
        _print_frontier_row(i, T_years, dt_step, frontier[i])
    print("\nInterpretation: Exercise when S < S* (critical price)")


def _print_frontier_row(i: int, T_years: float, dt_step: float, S_crit: float) -> None:
    """Affiche une ligne de la frontière d'exercice."""
    time_to_mat = T_years - i * dt_step
    if not np.isnan(S_crit):
        print(f"  {i:3d}     | S* = {S_crit:6.2f} (T-t = {time_to_mat:.2f}y)")
    else:
        print(f"  {i:3d}     | Never exercise")


# Validation contre le papier Longstaff-Schwartz (2001) Table 1
def validate_longstaff_schwartz_paper(seed: int = 42) -> None:
    """Validate implementation against Longstaff-Schwartz (2001) Table 1."""
    print("\n" + "=" * 60)
    print("VALIDATION: Longstaff-Schwartz (2001) Paper - Table 1")
    print("=" * 60)
    # Paramètres du papier
    K, r, T, nb_steps, nb_paths = 40, 0.06, 1.0, 50, 50000
    test_cases = _ls_paper_test_cases()
    print(f"\nParameters: K={K}, r={r}, T={T}, Steps={nb_steps}, Paths={nb_paths}")
    # Affichage des résultats
    _run_ls_validation(test_cases, K, r, nb_steps, nb_paths, seed)


def _ls_paper_test_cases() -> list:
    """Cas de test du Table 1 du papier LS (2001)."""
    return [
        (36, 0.20, 4.478, 4.472),
        (40, 0.20, 2.314, 2.313),
        (44, 0.20, 1.110, 1.118),
        (40, 0.40, 5.312, 5.308),
    ]


# Exécution de la validation LS pour chaque cas de test
def _run_ls_validation(
    test_cases: list, K: int, r: float,
    nb_steps: int, nb_paths: int, seed: int
) -> None:
    """Exécute le pricing LS pour chaque cas et affiche les résultats."""
    print("\n" + "-" * 70)
    print(f"{'S0':>4} | {'sigma':>5} | {'Paper FD':>10} | {'Paper LSM':>10} | {'Our LSM':>10} | {'Error':>8}")
    print("-" * 70)
    pricing_date = dt.date(2025, 1, 1)
    maturity = dt.date(2026, 1, 1)
    # Boucle sur les cas de test
    for S0, sigma, expected_fd, expected_lsm in test_cases:
        _run_single_ls_case(
            S0, sigma, expected_fd, expected_lsm,
            K, r, nb_steps, nb_paths, seed, pricing_date, maturity
        )
    print("-" * 70)


def _run_single_ls_case(
    S0: int, sigma: float, expected_fd: float, expected_lsm: float,
    K: int, r: float, nb_steps: int, nb_paths: int,
    seed: int, pricing_date: dt.date, maturity: dt.date
) -> None:
    """Exécute un seul cas de test LS et affiche le résultat."""
    market = Market(stock_price=S0, int_rate=r, sigma=sigma, div=0, div_date=None)
    option = Option(K=K, maturity=maturity, option_type="put", option_class="american")
    pricer = MonteCarloPricer(market, option, pricing_date, nb_steps, nb_paths, seed=seed)
    # Pricing avec polynômes de Laguerre
    our_price = pricer.price(method="vectoriel", regression_type="laguerre", antithetic=True)
    error = our_price - expected_lsm
    print(f"{S0:>4} | {sigma:>5.2f} | {expected_fd:>10.3f} | {expected_lsm:>10.3f} | {our_price:>10.3f} | {error:>+8.3f}")



# Point d'entrée principal
def main(seed: int = 42) -> None:
    """Run all examples."""
    print("\n" + "#" * 60)
    print(" Monte Carlo Option Pricing - Longstaff-Schwartz")
    print(f" Seed: {seed}")
    print("#" * 60)
    # Exécution des exemples sélectionnés
    _run_examples(seed)
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)

def _run_examples(seed: int) -> None:
    example_american_put_ls(seed)


if __name__ == "__main__":
    import sys
    # Allow seed to be passed as command line argument
    if len(sys.argv) > 1:
        seed = int(sys.argv[1])
    else:
        seed = 10
    main(seed)
