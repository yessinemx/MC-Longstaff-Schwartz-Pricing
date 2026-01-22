import time
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from market import Market
from option import Option
from model_mc import MonteCarloPricer
from blackscholes import BlackScholesPricer
from BrownianMotion import BrownianMotion

def generate_terminal_prices(market, pricing_date, maturity, nb_steps, nb_paths, antithetic=True):
    """
    Fonction helper pour reconstruire la distribution des prix S_T 
    (nécessaire uniquement pour le graphique de l'histogramme).
    """
    bm = BrownianMotion(seed=42)
    T_years = (maturity - pricing_date).days / 365.0
    
    # Génération des browniens
    dW, dt = bm.generate_dw(T_years, nb_steps, nb_paths, antithetic)
    
    # Formule GBM vectorielle
    drift = (market.r - market.dividend_yield - 0.5 * market.sigma**2) * dt
    diffusion = market.sigma * dW
    log_returns = np.sum(drift + diffusion, axis=1)
    
    # Prix terminaux
    return market.S0 * np.exp(log_returns)

def main():
    # ==========================================
    # 1. PARAMÉTRAGE
    # ==========================================
    # Données de marché et Option
    market = Market(S0=100, r=0.05, sigma=0.2, dividend=0)
    pricing_date = dt.date(2026, 1, 15)
    
    # Option pour la calibration (Call Européen)
    option_eu = Option(K=100, maturity=dt.date(2027, 1, 1), option_type="call", option_class="european")
    # Option pour le pricing avancé (Put Américain)
    option_am = Option(K=100, maturity=dt.date(2027, 1, 1), option_type="put", option_class="american")
    
    # Paramètres de simulation
    nb_steps = 50
    nb_paths = 50000 
    
    print(f"--- DÉBUT DE L'ANALYSE ({nb_paths} sims, {nb_steps} steps) ---")

    # ==========================================
    # 2. CALCULS DE PRIX & PERFORMANCE
    # ==========================================
    results = {}
    pricer_eu = MonteCarloPricer(market, option_eu, pricing_date, nb_steps, nb_paths)
    pricer_am = MonteCarloPricer(market, option_am, pricing_date, nb_steps, nb_paths)

    # A. Black-Scholes (Référence absolue)
    T_years = (option_eu.maturity - pricing_date).days / 365.0
    bs = BlackScholesPricer(market.S0, option_eu.K, T_years, market.r, market.sigma, "call")
    results['BS'] = bs.price()
    
    # B. Monte Carlo Scalaire
    print("Calcul Scalaire en cours...")
    t0 = time.time()
    results['Scalar'] = pricer_eu.price(method="scalaire")
    results['Time_Scalar'] = time.time() - t0

    # C. Monte Carlo Vectoriel
    print("Calcul Vectoriel en cours...")
    t0 = time.time()
    results['Vector'] = pricer_eu.price(method="vectoriel", antithetic=True)
    results['Time_Vector'] = time.time() - t0
    
    # D. Monte Carlo Américain (Longstaff-Schwartz)
    print("Calcul Américain (L&S) en cours...")
    results['American'] = pricer_am.price(method="vectoriel", antithetic=True)

    # ==========================================
    # 3. ANALYSE DE CONVERGENCE
    # ==========================================
    print("Analyse de convergence en cours...")
    N_range = [100, 500, 1000, 5000, 10000, 20000, 50000, 200000]
    conv_prices = []
    
    for n in N_range:
        tmp_pricer = MonteCarloPricer(market, option_eu, pricing_date, nb_steps, n)
        conv_prices.append(tmp_pricer.price(method="vectoriel", antithetic=True))

    # ==========================================
    # 4. GÉNÉRATION DES GRAPHIQUES
    # ==========================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # GRAPHIQUE 1 : Comparaison des Prix (Bar Chart)
    labels = ['Black-Scholes', 'MC Scalaire', 'MC Vectoriel']
    vals = [results['BS'], results['Scalar'], results['Vector']]
    colors = ['gray', 'blue', 'green']
    
    bars = axes[0].bar(labels, vals, color=colors, alpha=0.7)
    axes[0].axhline(results['BS'], color='red', linestyle='--', linewidth=1, label='Théorique')
    axes[0].set_title("Précision du Pricing (Call Euro)")
    axes[0].set_ylabel("Prix (€)")
    axes[0].set_ylim(min(vals)*0.95, max(vals)*1.05)
    axes[0].bar_label(bars, fmt='%.4f')
    
    # GRAPHIQUE 2 : Convergence (Line Chart)
    axes[1].plot(N_range, conv_prices, marker='o', linestyle='-', color='purple', label='MC Vectoriel')
    axes[1].axhline(results['BS'], color='red', linestyle='--', label='Black-Scholes')
    axes[1].set_title("Convergence du Prix vs Simulations (N)")
    axes[1].set_xlabel("Nombre de simulations (échelle log)")
    axes[1].set_ylabel("Prix (€)")
    axes[1].set_xscale('log')
    axes[1].grid(True, which="both", linestyle='--', alpha=0.3)
    axes[1].legend()

    # GRAPHIQUE 3 : Histogramme des distributions
    S_T_std = generate_terminal_prices(market, pricing_date, option_eu.maturity, nb_steps, nb_paths, antithetic=False)
    S_T_anti = generate_terminal_prices(market, pricing_date, option_eu.maturity, nb_steps, nb_paths, antithetic=True)
    
    axes[2].hist(S_T_std, bins=60, alpha=0.4, density=True, label='Standard', color='blue')
    axes[2].hist(S_T_anti, bins=60, alpha=0.4, density=True, label='Antithétique', color='orange')
    axes[2].axvline(bs.price(), color='red', linestyle='--', linewidth=2, label='Prix BS')
    axes[2].set_title("Distribution des Prix Terminaux $S_T$")
    axes[2].set_xlabel("Prix $S_T$")
    axes[2].set_xlim(40, 180) 
    axes[2].legend()

    # Finalisation
    plt.tight_layout()
    plt.savefig("resultats_projet.png") # Sauvegarde l'image
    print("\nGraphiques sauvegardés sous 'resultats_projet.png'")
    plt.show()

    # ==========================================
    # 5. RAPPORT CONSOLE
    # ==========================================
    print("\n" + "="*40)
    print(" RÉSULTATS FINAUX")
    print("="*40)
    print(f"1. Call Européen (Black-Scholes) : {results['BS']:.4f} €")
    print("-" * 40)
    print(f"2. MC Scalaire  : {results['Scalar']:.4f} €")
    print(f"   Temps calcul : {results['Time_Scalar']:.4f} sec")
    print("-" * 40)
    print(f"3. MC Vectoriel : {results['Vector']:.4f} €")
    print(f"   Temps calcul : {results['Time_Vector']:.4f} sec")
    print(f"   ACCÉLÉRATION : x{results['Time_Scalar'] / results['Time_Vector']:.1f}")
    print("-" * 40)
    print(f"4. Put Américain (Longstaff-Schwartz) : {results['American']:.4f} €")
    print("="*40)

    # Calcul du prix Black-Scholes pour l'option européenne
    bs_pricer = BlackScholesPricer(option_eu, market)
    bs_price = bs_pricer.price()

    # Tracé de l'histogramme des prix terminaux
    plt.hist(S_T_std, bins=50, alpha=0.5, color='blue', label='Prix terminaux')
    plt.axvline(x=bs_price, color='red', linestyle='--', label='Prix Black-Scholes')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()