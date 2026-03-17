# Streamlit UI pour MC-LONGSTAFF-SCHWARTZ-PRICING
# Pricing d'options américaines avec dividendes discrets
# =====================================================
from __future__ import annotations

import datetime as dt
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# --- On ajoute le dossier src/ au path AVANT tout import pricing ---
REPO_SRC = os.path.join(os.path.dirname(__file__), "src")
if os.path.isdir(REPO_SRC) and REPO_SRC not in sys.path:
    sys.path.insert(0, REPO_SRC)

# --- Imports pour Monte Carlo ---
try:
    from pricing.models.lsm.model_mc import MonteCarloPricer
except ImportError:
    MonteCarloPricer = None

from pricing.parameters.market import Market
from pricing.parameters.option import Option
from pricing.models.tree.trinomial_tree import TrinomialTree
from pricing.models.black_scholes.blackscholes import BlackScholesPricer
from pricing.greeks import GreeksCalculator


DAY_COUNT: float = 365.0  # ACT/365F : compte de jours classique

# =====================================================
# Configuration Streamlit (UNE SEULE FOIS)
# =====================================================
st.set_page_config(
    page_title="Longstaff Schwartz American Option Pricer",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS personnalisé pour améliorer l'apparence
st.markdown("""
<style>
    /* Élargir la sidebar */
    [data-testid="stSidebar"] {
        width: 350px !important;
    }

    [data-testid="stSidebarContent"] {
        width: 350px !important;
    }

    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .metric-value { font-size: 32px; font-weight: bold; }
    .metric-label { font-size: 12px; opacity: 0.9; }
    .success-metric { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }
    .warning-metric { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }
</style>
""", unsafe_allow_html=True)


def yearfrac(start: dt.date, end: dt.date, basis: float = DAY_COUNT) -> float:
    """Calcule la fraction d'année entre deux dates (jamais négatif)."""
    return max(0.0, (end - start).days / basis)


@dataclass
class AppInputs:
    """Regroupe tous les paramètres de l'interface en un seul objet."""
    pricing_date: dt.date
    maturity: dt.date
    S0: float
    K: float
    r: float
    sigma: float
    option_type: str  # 'call' | 'put'
    option_class: str  # 'european' | 'american'
    dividend: float
    dividend_date: Optional[dt.date]
    N: int
    pruning: bool
    epsilon: float
    conv_min: int
    conv_max: int
    conv_step: int
    curve_pts: int
    curve_span_pct: float
    # Monte Carlo parameters
    mc_nb_paths: int = 5000
    mc_nb_steps: int = 50
    mc_method: str = "vectoriel"
    mc_regression_type: str = "quadratic"
    mc_degree: int = 2
    mc_antithetic: bool = True
    mc_seed: int = 42
    mc_batch_size: Optional[int] = None  # None = pas de batch
    mc_rng_method: str = "direct"  # "direct" ou "uniform_ppf"


# ---------------------------------------------------------------------
# Pricing helpers – les petites fonctions utiles
# ---------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def bs_price_cached(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    opt_type: str,
    dividend: float = 0.0,
    dividend_date: Optional[dt.date] = None,
) -> float:
    """Calcule le prix BS avec cache Streamlit pour éviter les recalculs."""
    bs = BlackScholesPricer(
        S,
        K,
        T,
        r,
        sigma,
        opt_type,  # type: ignore
        dividend=dividend,
        dividend_date=dividend_date,
    )
    return float(bs.price())


@st.cache_resource(show_spinner=False)
def tree_cached(market: Market, N: int, pruning: bool, epsilon: float, pricing_date: dt.date) -> TrinomialTree:
    """Cache l'arbre trinomial pour éviter sa reconstruction."""
    return TrinomialTree(
        market,
        N=N,
        pruning=pruning,
        epsilon=epsilon,
        pricing_date=pricing_date,
    )


@st.cache_data(show_spinner=False)
def tree_price_cached(
    S: float,
    K: float,
    r: float,
    sigma: float,
    div: float,
    div_date: Optional[dt.date],
    maturity: dt.date,
    pricing_date: dt.date,
    N: int,
    pruning: bool,
    epsilon: float,
    opt_type: str,
    opt_class: str,
) -> Tuple[float, float]:
    """Calcule le prix et runtime de l'arbre trinomial avec cache."""
    market = Market(stock_price=S, int_rate=r, sigma=sigma, div=div, div_date=div_date)
    option = Option(K, maturity, opt_type, opt_class)
    return tree_price(market, N, pruning, epsilon, pricing_date, option)


def build_market_option(inputs: AppInputs) -> Tuple[Market, Option, float]:
    """Construit les objets Market, Option et calcule T (années)."""
    market = Market(
        stock_price=inputs.S0,
        int_rate=inputs.r,
        sigma=inputs.sigma,
        div=inputs.dividend,
        div_date=inputs.dividend_date,
    )
    option = Option(
        inputs.K,
        inputs.maturity,
        inputs.option_type,
        inputs.option_class,
    )
    T: float = yearfrac(inputs.pricing_date, inputs.maturity)
    return market, option, T


def tree_price(
    market: Market,
    N: int,
    pruning: bool,
    epsilon: float,
    pricing_date: dt.date,
    option: Option,
) -> Tuple[float, float]:
    """Calcule le prix via arbre trinomial et chronomètre le temps."""
    t0: float = time.perf_counter()
    tree = TrinomialTree(
        market,
        N=N,
        pruning=pruning,
        epsilon=epsilon,
        pricing_date=pricing_date,
    )
    price_val: float = float(tree.price(option))
    elapsed: float = time.perf_counter() - t0
    return price_val, elapsed


def _build_tree_for_greeks(
    market: Market,
    option: Option,
    inputs: AppInputs,
) -> TrinomialTree:
    """Construit l'arbre en mode 'compute_greeks' pour récupérer les sensibilités."""
    tree = TrinomialTree(
        market,
        N=inputs.N,
        pruning=inputs.pruning,
        epsilon=inputs.epsilon,
        pricing_date=inputs.pricing_date,
    )
    _ = tree.price(option, compute_greeks=True)
    return tree


def _greeks_from_tree(
    tree: TrinomialTree,
    market: Market,
    option: Option,
    inputs: AppInputs,
) -> Dict[str, float]:
    """Extrait Delta, Gamma, Vega, Rho, et Theta de l'arbre."""
    delta: float = float(tree.delta())
    gamma: float = float(tree.gamma())
    vega: float = float(tree.vega(option))
    rho: float = float(tree.rho(option))
    theta_day: float = compute_theta(tree, market, option, inputs)
    return {
        "delta": delta,
        "gamma": gamma,
        "theta_day": theta_day,
        "vega": vega,
        "rho": rho,
    }


def compute_tree_and_greeks(
    market: Market,
    option: Option,
    inputs: AppInputs,
) -> Tuple[TrinomialTree, Dict[str, float]]:
    """Construit l'arbre et renvoie aussi les grecs calculés."""
    tree_greeks = _build_tree_for_greeks(market, option, inputs)
    greeks = _greeks_from_tree(tree_greeks, market, option, inputs)
    return tree_greeks, greeks


# ---- Greeks via finite differences (méthode du prof - utils.py) ----

def _mc_price_for_spot(inputs: AppInputs, spot: float) -> float:
    """Helper: reprice MC avec un spot différent (pour delta/gamma MC)."""
    market = Market(
        stock_price=spot,
        int_rate=inputs.r,
        sigma=inputs.sigma,
        div=inputs.dividend,
        div_date=inputs.dividend_date,
    )
    option = Option(inputs.K, inputs.maturity, inputs.option_type, inputs.option_class)
    pricer = MonteCarloPricer(
        market, option, inputs.pricing_date,
        nb_steps=inputs.mc_nb_steps, nb_paths=inputs.mc_nb_paths,
        seed=inputs.mc_seed, batch_size=inputs.mc_batch_size,
        rng_method=inputs.mc_rng_method,
    )
    return pricer.price(
        antithetic=inputs.mc_antithetic,
        method="vectoriel",
        regression_type=inputs.mc_regression_type,
        degree=inputs.mc_degree,
    )


def _mc_price_for_sigma(inputs: AppInputs, sigma: float) -> float:
    """Helper: reprice MC avec une vol différente (pour vega MC)."""
    market = Market(
        stock_price=inputs.S0,
        int_rate=inputs.r,
        sigma=sigma,
        div=inputs.dividend,
        div_date=inputs.dividend_date,
    )
    option = Option(inputs.K, inputs.maturity, inputs.option_type, inputs.option_class)
    pricer = MonteCarloPricer(
        market, option, inputs.pricing_date,
        nb_steps=inputs.mc_nb_steps, nb_paths=inputs.mc_nb_paths,
        seed=inputs.mc_seed, batch_size=inputs.mc_batch_size,
        rng_method=inputs.mc_rng_method,
    )
    return pricer.price(
        antithetic=inputs.mc_antithetic,
        method="vectoriel",
        regression_type=inputs.mc_regression_type,
        degree=inputs.mc_degree,
    )


def _mc_price_for_rate(inputs: AppInputs, rate: float) -> float:
    """Helper: reprice MC avec un taux différent (pour rho MC)."""
    market = Market(
        stock_price=inputs.S0,
        int_rate=rate,
        sigma=inputs.sigma,
        div=inputs.dividend,
        div_date=inputs.dividend_date,
    )
    option = Option(inputs.K, inputs.maturity, inputs.option_type, inputs.option_class)
    pricer = MonteCarloPricer(
        market, option, inputs.pricing_date,
        nb_steps=inputs.mc_nb_steps, nb_paths=inputs.mc_nb_paths,
        seed=inputs.mc_seed, batch_size=inputs.mc_batch_size,
        rng_method=inputs.mc_rng_method,
    )
    return pricer.price(
        antithetic=inputs.mc_antithetic,
        method="vectoriel",
        regression_type=inputs.mc_regression_type,
        degree=inputs.mc_degree,
    )


def _mc_price_for_date(inputs: AppInputs, pricing_date: dt.date) -> float:
    """Helper: reprice MC avec une date de pricing différente (pour theta MC)."""
    market = Market(
        stock_price=inputs.S0,
        int_rate=inputs.r,
        sigma=inputs.sigma,
        div=inputs.dividend,
        div_date=inputs.dividend_date,
    )
    option = Option(inputs.K, inputs.maturity, inputs.option_type, inputs.option_class)
    pricer = MonteCarloPricer(
        market, option, pricing_date,
        nb_steps=inputs.mc_nb_steps, nb_paths=inputs.mc_nb_paths,
        seed=inputs.mc_seed, batch_size=inputs.mc_batch_size,
        rng_method=inputs.mc_rng_method,
    )
    return pricer.price(
        antithetic=inputs.mc_antithetic,
        method="vectoriel",
        regression_type=inputs.mc_regression_type,
        degree=inputs.mc_degree,
    )


def compute_mc_greeks(inputs: AppInputs, mc_price_base: float) -> Dict[str, float]:
    """
    Calcule les Greeks MC par différences finies (méthode du professeur - utils.py).

    Delta  = (f(S+h) - f(S-h)) / (2h)          h = 0.1% * S
    Gamma  = (f(S+h) + f(S-h) - 2f(S)) / h^2   h = 5% * S
    Vega   = f(sigma+0.5%) - f(sigma-0.5%)
    Theta  = f(t+1d) - f(t)  (per day)
    Rho    = (f(r+0.5%) - f(r-0.5%)) / (1%)
    """
    results: Dict[str, float] = {}

    # Delta: shift = 0.1% * S (comme OptionPricingParam.UND_SHIFT = 0.001)
    h_delta = inputs.S0 * 0.001
    p_up = _mc_price_for_spot(inputs, inputs.S0 + h_delta)
    p_down = _mc_price_for_spot(inputs, inputs.S0 - h_delta)
    results["delta"] = (p_up - p_down) / (2 * h_delta)

    # Gamma: shift = 5% * S (comme dans utils.py calculate_gamma)
    h_gamma = inputs.S0 * 0.05
    p_up_g = _mc_price_for_spot(inputs, inputs.S0 + h_gamma)
    p_down_g = _mc_price_for_spot(inputs, inputs.S0 - h_gamma)
    results["gamma"] = (p_up_g + p_down_g - 2 * mc_price_base) / (h_gamma ** 2)

    # Vega: shift = 0.5% (comme OptionPricingParam.VOL_SHIFT = 0.005)
    h_vega = 0.005
    p_vup = _mc_price_for_sigma(inputs, inputs.sigma + h_vega)
    p_vdown = _mc_price_for_sigma(inputs, inputs.sigma - h_vega)
    results["vega"] = p_vup - p_vdown  # Vega = price(sigma+0.5%) - price(sigma-0.5%)

    # Theta: shift = 1 day forward
    one_day = dt.timedelta(days=1)
    pricing_date_plus_1d = inputs.pricing_date + one_day
    if pricing_date_plus_1d >= inputs.maturity:
        results["theta"] = 0.0
    else:
        p_theta_tomorrow = _mc_price_for_date(inputs, pricing_date_plus_1d)
        results["theta"] = p_theta_tomorrow - mc_price_base

    # Rho: shift = 0.5% on interest rate
    h_rho = 0.005
    p_rup = _mc_price_for_rate(inputs, inputs.r + h_rho)
    p_rdown = _mc_price_for_rate(inputs, inputs.r - h_rho)
    results["rho"] = (p_rup - p_rdown) / (2 * h_rho)

    return results


@st.cache_data(show_spinner=False)
def price_vs_strike_cached(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    div: float,
    div_date: Optional[dt.date],
    maturity: dt.date,
    pricing_date: dt.date,
    N: int,
    pruning: bool,
    epsilon: float,
    opt_type: str,
    opt_class: str,
    curve_pts: int,
    curve_span_pct: float,
) -> pd.DataFrame:
    """Calcule les prix (arbre et BS) pour une gamme de strikes avec cache."""
    inputs_temp = AppInputs(
        pricing_date=pricing_date,
        maturity=maturity,
        S0=S0,
        K=K,
        r=r,
        sigma=sigma,
        option_type=opt_type,
        option_class=opt_class,
        dividend=div,
        dividend_date=div_date,
        N=N,
        pruning=pruning,
        epsilon=epsilon,
        conv_min=N,
        conv_max=N,
        conv_step=1,
        curve_pts=curve_pts,
        curve_span_pct=curve_span_pct,
    )
    return price_vs_strike(inputs_temp)


def compute_theta(
    tree_greeks: TrinomialTree,
    market: Market,
    option: Option,
    inputs: AppInputs,
) -> float:
    """Theta (par jour) via différence finie : on avance d'un jour."""
    # Build fresh trees for both dates to avoid using a corrupted
    # root.option_value (vega/rho rebuild the tree with bumped params).
    tree_today = TrinomialTree(
        market,
        N=inputs.N,
        pruning=inputs.pruning,
        epsilon=inputs.epsilon,
        pricing_date=inputs.pricing_date,
    )
    tree_next = TrinomialTree(
        market,
        N=inputs.N,
        pruning=inputs.pruning,
        epsilon=inputs.epsilon,
        pricing_date=inputs.pricing_date + dt.timedelta(days=1),
    )
    v0: float = float(tree_today.price(option))
    v1: float = float(tree_next.price(option))
    return v1 - v0


def _strike_grid(inputs: AppInputs) -> np.ndarray:
    """Génère une grille de strikes autour de K selon le span %."""
    span: float = inputs.curve_span_pct / 100.0
    return np.linspace(
        inputs.K * (1.0 - span),
        inputs.K * (1.0 + span),
        inputs.curve_pts,
    )


def price_vs_strike(inputs: AppInputs) -> pd.DataFrame:
    """Calcule les prix (arbre et BS) pour une gamme de strikes."""
    strikes: np.ndarray = _strike_grid(inputs)
    market, _, T = build_market_option(inputs)
    rows: List[Tuple[float, float, float]] = []

    for k in strikes:
        opt_k = Option(k, inputs.maturity, inputs.option_type, inputs.option_class)
        tree_val, _ = tree_price(
            market,
            inputs.N,
            inputs.pruning,
            inputs.epsilon,
            inputs.pricing_date,
            opt_k,
        )
        bs_val: float
        if inputs.option_class.lower() == "european":
            bs_val = compute_bs_price(inputs, float(k), T)
        else:
            bs_val = float("nan")
        rows.append((float(k), tree_val, bs_val))

    return (
        pd.DataFrame(rows, columns=["Strike", "Tree", "BlackScholes"])
        .set_index("Strike")
    )


def compute_bs_price(inputs: AppInputs, strike: float, T: float) -> float:
    """Calcule le prix BS pour un strike donné (avec cache)."""
    return bs_price_cached(
        inputs.S0,
        strike,
        T,
        inputs.r,
        inputs.sigma,
        inputs.option_type,
        dividend=inputs.dividend,
        dividend_date=inputs.dividend_date,
    )


# =====================================================
# Multi-Seed Analysis (loop seeds 1..N, vectoriel)
# =====================================================
def run_multi_seed_analysis(
    inputs: AppInputs,
    seed_min: int,
    seed_max: int,
) -> pd.DataFrame:
    """
    Lance le pricing MC vectoriel pour chaque seed de seed_min à seed_max.
    Retourne un DataFrame avec colonnes: Seed, Price, StdError, Runtime.
    """
    market, option, T = build_market_option(inputs)
    rows: List[Dict] = []

    for seed in range(seed_min, seed_max + 1):
        t0 = time.perf_counter()
        pricer = MonteCarloPricer(
            market, option, inputs.pricing_date,
            nb_steps=inputs.mc_nb_steps,
            nb_paths=inputs.mc_nb_paths,
            seed=seed,
            batch_size=inputs.mc_batch_size,
            rng_method=inputs.mc_rng_method,
        )
        price = pricer.price(
            antithetic=inputs.mc_antithetic,
            method="vectoriel",  # toujours vectoriel (rapide)
            regression_type=inputs.mc_regression_type,
            degree=inputs.mc_degree,
        )
        elapsed = time.perf_counter() - t0
        rows.append({
            "Seed": seed,
            "Price": price,
            "StdError": pricer.last_std_error,
            "Runtime": elapsed,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# UI – interface Streamlit
# ---------------------------------------------------------------------
st.title("🏦 Longstaff Schwartz American Option Pricer")
st.caption("Compare trinomial tree, Black-Scholes, and Monte Carlo LSM pricing methods with comprehensive analysis tools.")
st.divider()

# =====================================================
# Sidebar - Configuration Parameters (Pricing Methods Only)
# =====================================================

with st.sidebar:
    today: dt.date = dt.date.today()

    st.write("**⚙️ PRICING METHODS**")

    # Trinomial tree - PAS DE max_value
    with st.expander("🌳 Trinomial Tree", expanded=False):
        N: int = st.number_input("Steps (N)", min_value=10, value=1000, step=50)
        col1, col2 = st.columns(2)
        with col1:
            pruning: bool = st.toggle("Pruning", value=True)
        with col2:
            epsilon: float = st.number_input("Epsilon", min_value=1e-12, value=1e-7, format="%0.1e")

    # Monte Carlo - PAS DE max_value sur paths et steps
    with st.expander("🎲 Monte Carlo LSM", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            mc_nb_paths: int = st.number_input("Paths", min_value=100, value=20000, step=1000)
            mc_nb_steps: int = st.number_input("Steps (MC)", min_value=5, value=N, step=50)
        with col2:
            mc_regression_type: str = st.selectbox("Regression", ["Quadratic", "Laguerre", "Hermite", "Legendre", "Chebyshev"])
            mc_degree: int = st.slider("Degree", min_value=1, max_value=10, value=2)

        mc_antithetic: bool = st.toggle("Antithetic", value=False)
        mc_method_display: str = st.selectbox("MC Method", ["Scalaire", "Vectoriel"], key="mc_meth", index=1)
        # Convert display names to internal method names
        mc_method: str = "scalaire" if mc_method_display == "Scalaire" else "vectoriel"
        mc_seed: int = st.number_input("Seed", min_value=0, value=42)

        # Choix de la méthode de génération aléatoire
        mc_rng_method_display: str = st.selectbox(
            "Random Generation",
            ["Standard Normal", "Uniform PPF"],
            index=1
        )
        # Convert display names to code names
        mc_rng_method: str = "direct" if mc_rng_method_display == "Standard Normal" else "uniform_ppf"

        use_batch: bool = st.toggle("Mini-batches", value=False,
                                    help="Découpe les simulations en mini-batches pour réduire la mémoire")
        mc_batch_size: Optional[int] = None
        if use_batch:
            # Par défaut: batch_size = nb_steps * nb_paths (pas de découpe)
            default_batch_size = mc_nb_steps * mc_nb_paths
            
            batch_size_total = st.number_input(
                "Batch size",
                min_value=mc_nb_steps,
                value=default_batch_size,
                step=1000,
                format="%.1e",
                help="Batch size = Nb Paths * Nb Steps"
            )
            
            # Calculer nb_paths_per_batch = batch_size_total / nb_steps
            nb_paths_per_batch = max(1, batch_size_total // mc_nb_steps)
            mc_batch_size = nb_paths_per_batch
            
            # Afficher la décomposition en notation scientifique
            nb_batches_preview = math.ceil(mc_nb_paths / nb_paths_per_batch)
            total_elements = nb_paths_per_batch * mc_nb_steps
    
    # Convergence & curves parameters
    conv_min: int = 50
    conv_max: int = 400
    conv_step: int = 25
    curve_pts: int = 41
    curve_span_pct: float = 30
    
    # Tree display parameters (default values - will be overridden in tab4)
    max_depth: int = 100
    proba_min: float = 1e-6
    percentile_clip: int = 0
    edge_alpha: float = 0.35
    linewidth: float = 0.4

    st.divider()
    st.write("**🔴 CALCULATE**")
    calculate_button = st.button("Price", use_container_width=True, key="calc_btn")
    if calculate_button:
        st.session_state.calculate = True

# Initialize session states BEFORE buttons
if 'calculate' not in st.session_state:
    st.session_state.calculate = False
if 'compute_greeks' not in st.session_state:
    st.session_state.compute_greeks = False
if 'compute_plots' not in st.session_state:
    st.session_state.compute_plots = False
if 'last_mc_params_hash' not in st.session_state:
    st.session_state.last_mc_params_hash = None
if 'run_multi_seed' not in st.session_state:
    st.session_state.run_multi_seed = False

# =====================================================
# MAIN BODY - 3 Columns: Dates | Market | Option
# =====================================================

col1, col2, col3 = st.columns([1, 1, 1])

# Column 1: Dates
with col1:
    st.write("**📅 Dates**")
    pricing_date: dt.date = st.date_input("Pricing date", value=today, key="pricing_date")
    maturity: dt.date = st.date_input("Maturity", value=today + dt.timedelta(days=180), key="maturity")
    T_years = yearfrac(pricing_date, maturity)
    st.caption(f"T = {T_years:.4f} years ({(maturity - pricing_date).days} days)")

# Column 2: Market Parameters
with col2:
    st.write("**💹 Market**")
    
    # Row 1: Spot only
    S0: float = st.number_input("Spot (S0)", min_value=0.01, value=100.0, step=1.0, key="spot")
    
    # Row 2: Rate and Volatility
    mk_col1, mk_col2 = st.columns([1, 1])
    with mk_col1:
        r: float = st.number_input("Interest Rate (%)", min_value=0.0, value=2.0, step=0.1, key="rate") / 100
    with mk_col2:
        sigma: float = st.number_input("Volatility (%)", min_value=0.1, value=20.0, step=0.5, key="vol") / 100

    # Moneyness will be calculated after Strike is defined
    
    # Discrete Dividend
    has_div: bool = st.toggle("Enable dividend", value=False, key="div_toggle")
    dividend: float = 0.0
    dividend_date: Optional[dt.date] = None
    if has_div:
        div_col1, div_col2 = st.columns([1, 1])
        with div_col1:
            dividend = st.number_input("Amount", min_value=0.0, value=2.0, step=0.5, key="div_amt")
        with div_col2:
            dividend_date = st.date_input("Payment date", value=today + dt.timedelta(days=90), key="div_date")

# Column 3: Option Configuration
with col3:
    st.write("**📊 Option**")
    
    # Row 1: Strike only
    K: float = st.number_input("Strike (K)", min_value=0.01, value=100.0, step=1.0, key="strike")
    
    # Row 2: Type and Style
    col_opt1, col_opt2 = st.columns([1, 1])
    with col_opt1:
        option_type: str = st.radio("Type", ["call", "put"], horizontal=False, key="opt_type")
    with col_opt2:
        option_class: str = st.radio("Style", ["european", "american"], horizontal=False, key="opt_class")

    # Moneyness indicator (after Strike is defined)
    moneyness = S0 / K if K > 0 else 1.0
    
    # Determine moneyness status (ATM, OTM, ITM) based on option type
    if option_type == "call":
        if moneyness > 1.05:
            status = "ITM"
        elif moneyness < 0.95:
            status = "OTM"
        else:
            status = "ATM"
    else:  # put
        if moneyness < 0.95:
            status = "ITM"
        elif moneyness > 1.05:
            status = "OTM"
        else:
            status = "ATM"
    
    st.caption(f"**Moneyness =** {moneyness*100:.1f}% ({status})")


# Group all inputs
inputs = AppInputs(
    pricing_date=pricing_date,
    maturity=maturity,
    S0=float(S0),
    K=float(K),
    r=float(r),
    sigma=float(sigma),
    option_type=option_type,
    option_class=option_class,
    dividend=float(dividend),
    dividend_date=dividend_date,
    N=int(N),
    pruning=bool(pruning),
    epsilon=float(epsilon),
    conv_min=int(conv_min),
    conv_max=int(conv_max),
    conv_step=int(conv_step),
    curve_pts=int(curve_pts),
    curve_span_pct=float(curve_span_pct),
    mc_nb_paths=int(mc_nb_paths),
    mc_nb_steps=int(mc_nb_steps),
    mc_regression_type=mc_regression_type,
    mc_degree=int(mc_degree),
    mc_antithetic=bool(mc_antithetic),
    mc_seed=int(mc_seed),
    mc_method=mc_method,
    mc_batch_size=int(mc_batch_size) if mc_batch_size else None,
    mc_rng_method=mc_rng_method,
)

# Compute hash of MC parameters to detect changes
import hashlib
current_mc_params_str = f"{inputs.mc_nb_paths}|{inputs.mc_nb_steps}|{inputs.mc_regression_type}|{inputs.mc_degree}|{inputs.mc_antithetic}|{inputs.mc_seed}|{inputs.mc_method}|{inputs.option_class}|{inputs.mc_batch_size}|{inputs.mc_rng_method}"
current_hash = hashlib.md5(current_mc_params_str.encode()).hexdigest()

# If MC parameters changed, auto-reset the calculate flag to force recalculation
if current_hash != st.session_state.last_mc_params_hash:
    st.session_state.calculate = False
    st.session_state.last_mc_params_hash = current_hash

st.divider()

# Construction du marché et de l'option
market, option, T = build_market_option(inputs)

# Tabs for different sections
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Pricing",
    "📈 Greeks",
    "🔬 Plots",
    "🌳 Tree",
    "🔄 Multi-Seed Analysis",
])

# =====================================================
# TAB 1: PRICING (Tree, BS, MC all together)
# =====================================================
with tab1:
    # Show info if price calculation needed
    if not st.session_state.calculate:
        st.info("👉 Adjust your parameters above and click **Price** to see results")

    # Initialize prices with placeholders
    tree_price_val = float("nan")
    bs_val = float("nan")
    mc_price = float("nan")

    col1, col2, col3 = st.columns([1, 1, 1])

    # TRINOMIAL TREE
    with col1:
        st.subheader("🌳 Trinomial Tree")
        if st.session_state.calculate:
            try:
                tree_price_val, t_sec = tree_price(
                    market,
                    inputs.N,
                    inputs.pruning,
                    inputs.epsilon,
                    inputs.pricing_date,
                    option,
                )
                st.metric(
                    "Price",
                    f"{tree_price_val:,.6f}",
                )
                st.markdown(f":green[⚡ Runtime: {t_sec:.4f}s (N={inputs.N})]")
                # Formule d'erreur du professeur
                numerator = (3/8) * inputs.S0 * (np.exp(inputs.sigma**2 * T / inputs.N) - 1) * np.exp(2*inputs.r*T/inputs.N)
                denominator = np.sqrt(2*np.pi) * np.sqrt(np.exp(inputs.sigma**2*T) - 1)
                tree_error = numerator / denominator
                st.caption(f"Error interval: +/- {tree_error:,.4f}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
                tree_price_val = float("nan")
        else:
            st.metric("Price", "--")
            st.markdown(":gray[⚡ Runtime: --]")
            st.caption("Error interval: --")

    # BLACK-SCHOLES
    with col2:
        st.subheader("📐 Black-Scholes")
        if not st.session_state.calculate:
            st.metric("Price", "--")
            st.markdown(":gray[⚡ Runtime: --]")
            st.caption("Error interval: --")
        elif inputs.option_class.lower() == "european":
            try:
                bs_val: float = bs_price_cached(
                    inputs.S0,
                    inputs.K,
                    T,
                    inputs.r,
                    inputs.sigma,
                    inputs.option_type,
                    dividend=inputs.dividend,
                    dividend_date=inputs.dividend_date,
                )
                t_bs_start = time.perf_counter()
                bs_price_cached(
                    inputs.S0,
                    inputs.K,
                    T,
                    inputs.r,
                    inputs.sigma,
                    inputs.option_type,
                    dividend=inputs.dividend,
                    dividend_date=inputs.dividend_date,
                )
                t_bs = time.perf_counter() - t_bs_start

                st.metric(
                    "Price",
                    f"{bs_val:,.6f}",
                )
                st.markdown(f":green[⚡ Runtime: {t_bs:.4f}s]")
            except Exception as e:
                st.error(f"Error: {str(e)}")
                bs_val = float("nan")
        else:
            st.info("BS valid only for European options")
            bs_val = float("nan")

    # MONTE CARLO LSM
    with col3:
        st.subheader("🎲 Monte Carlo LSM")

    pricer_mc = None
    mc_price = float("nan")
    t_mc_sec = 0

    if not st.session_state.calculate:
        with col3:
            st.metric("Price", "--")
            st.markdown(":gray[⚡ Runtime: --]")
            st.caption("Error interval: --")
    elif MonteCarloPricer is not None:
        try:
            t0_mc = time.perf_counter()
            pricer_mc = MonteCarloPricer(
                market,
                option,
                inputs.pricing_date,
                nb_steps=inputs.mc_nb_steps,
                nb_paths=inputs.mc_nb_paths,
                seed=inputs.mc_seed,
                batch_size=inputs.mc_batch_size,
                rng_method=inputs.mc_rng_method,
            )
            mc_price = pricer_mc.price(
                antithetic=inputs.mc_antithetic,
                method=inputs.mc_method,
                regression_type=inputs.mc_regression_type,
                degree=inputs.mc_degree,
                american_method='ls'
            )
            t_mc_sec = time.perf_counter() - t0_mc

            with col3:
                st.metric(
                    "Price",
                    f"{mc_price:,.6f}",
                )
                st.markdown(f":green[⚡ Runtime: {t_mc_sec:.4f}s ({inputs.mc_nb_paths} paths)]")
                mc_error = pricer_mc.last_std_error if pricer_mc.last_std_error > 0 else abs(mc_price) / np.sqrt(inputs.mc_nb_paths)
                st.caption(f"Error interval: +/- {mc_error:,.4f}")

        except Exception as e:
            with col3:
                st.error(f"Error: {str(e)}")
            mc_price = float("nan")
    else:
        with col3:
            st.error("MonteCarloPricer not available")
        mc_price = float("nan")

    # PRICING METHODS COMPARISON
    st.divider()

    # Create summary metrics (only if calculated)
    if st.session_state.calculate:
        prices_list = [tree_price_val, mc_price]
        if inputs.option_class.lower() == "european" and not np.isnan(bs_val):
            prices_list.append(bs_val)
        valid_prices = [p for p in prices_list if not (isinstance(p, float) and np.isnan(p))]

        if valid_prices:
            price_min = min(valid_prices)
            price_max = max(valid_prices)
            price_gap = tree_price_val - mc_price
            avg_price = sum(valid_prices) / len(valid_prices)

            # Display summary metrics
            summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
            with summary_col1:
                st.metric("Lowest Price", f"{price_min:,.6f}")
            with summary_col2:
                st.metric("Highest Price", f"{price_max:,.6f}")
            with summary_col3:
                st.metric("Price Gap", f"{price_gap:,.6f}")
            with summary_col4:
                st.metric("Average Price", f"{avg_price:,.6f}")

    # EXERCISE FRONTIER (if American and MC available)
    if 'pricer_mc' in locals() and pricer_mc is not None and inputs.option_class.lower() == "american":
        st.divider()
        st.subheader("Optimal Exercise Boundary")

        try:
            times, frontier = pricer_mc.get_exercise_frontier(
                antithetic=inputs.mc_antithetic,
                regression_type=inputs.mc_regression_type,
                degree=inputs.mc_degree
            )

            fig_frontier, ax = plt.subplots(figsize=(10, 5))
            valid_idx = ~np.isnan(frontier)
            ax.plot(times[valid_idx], frontier[valid_idx], 'b-', linewidth=3, label='Exercise Boundary', marker='o', markersize=4)
            ax.axhline(y=inputs.K, color='r', linestyle='--', alpha=0.7, linewidth=2, label=f'Strike K={inputs.K:.2f}')
            ax.axhline(y=inputs.S0, color='g', linestyle=':', alpha=0.7, linewidth=2, label=f'Spot S0={inputs.S0:.2f}')

            ax.set_xlabel('Time to Maturity (years)', fontsize=11)
            ax.set_ylabel('Stock Price', fontsize=11)
            ax.set_title('Optimal Exercise Boundary', fontsize=13, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            st.pyplot(fig_frontier, clear_figure=True)

        except Exception as e:
            st.warning(f"Cannot compute frontier: {str(e)}")


# =====================================================
# TAB 2: GREEKS (Tree + BS + MC via finite differences)
# =====================================================
with tab2:
    st.subheader("📈 Sensitivity Analysis - Greeks")

    col_greeks_btn1, col_greeks_btn2 = st.columns([1, 5])
    with col_greeks_btn1:
        if st.button("Compute Greeks", use_container_width=True, key="greeks_btn"):
            st.session_state.compute_greeks = True

    st.divider()

    if not st.session_state.compute_greeks:
        st.info("👉 Click **Compute Greeks** button above to start")
    else:
        tree_greeks, g_tree = compute_tree_and_greeks(market, option, inputs)

        # Greeks from BS if European
        if inputs.option_class.lower() == "european":
            bs = BlackScholesPricer(
                inputs.S0,
                inputs.K,
                T,
                inputs.r,
                inputs.sigma,
                inputs.option_type,  # type: ignore
                dividend=inputs.dividend,
                dividend_date=inputs.dividend_date,
            )
            bs_market = Market(stock_price=inputs.S0, int_rate=inputs.r, sigma=inputs.sigma)
            def _bs_price_fn():
                bs.S, bs.sigma, bs.r = bs_market.stock_price, bs_market.sigma, bs_market.int_rate
                return bs.price()
            g_bs = GreeksCalculator(_bs_price_fn, bs_market).greeks()
            bs_delta = g_bs.get("delta", np.nan)
            bs_gamma = g_bs.get("gamma", np.nan)
            bs_theta = np.nan
            bs_vega = g_bs.get("vega", np.nan)
            bs_rho = g_bs.get("rho", np.nan)
        else:
            bs_delta = bs_gamma = bs_theta = bs_vega = bs_rho = np.nan

        # Greeks from MC via finite differences (méthode du prof - utils.py)
        # Use fewer paths for faster finite-difference greeks computation
        mc_greeks_dict: Dict[str, float] = {}
        if MonteCarloPricer is not None:
            with st.spinner("Computing MC Greeks via finite differences..."):
                try:
                    # Use min(nb_paths, 5000) for the FD bumps – much faster, still accurate
                    fd_paths = min(inputs.mc_nb_paths, 5000)
                    pricer_base = MonteCarloPricer(
                        market, option, inputs.pricing_date,
                        nb_steps=inputs.mc_nb_steps, nb_paths=fd_paths,
                        seed=inputs.mc_seed, batch_size=inputs.mc_batch_size,
                        rng_method=inputs.mc_rng_method,
                    )
                    mc_base = pricer_base.price(
                        antithetic=inputs.mc_antithetic,
                        method="vectoriel",
                        regression_type=inputs.mc_regression_type,
                        degree=inputs.mc_degree,
                    )
                    # Build a lightweight inputs copy with capped paths for FD
                    from dataclasses import replace
                    inputs_fd = replace(inputs, mc_nb_paths=fd_paths)
                    mc_greeks_dict = compute_mc_greeks(inputs_fd, mc_base)
                except Exception as e:
                    st.warning(f"MC Greeks computation failed: {e}")

        # Comparison table: Tree / BS / MC
        greek_data = {
            "Tree": [
                g_tree["delta"],
                g_tree["gamma"],
                g_tree["theta_day"],
                g_tree["vega"],
                g_tree["rho"],
            ],
            "Black-Scholes": [bs_delta, bs_gamma, bs_theta, bs_vega, bs_rho],
        }
        if mc_greeks_dict:
            greek_data["Monte Carlo (FD)"] = [
                mc_greeks_dict.get("delta", np.nan),
                mc_greeks_dict.get("gamma", np.nan),
                mc_greeks_dict.get("theta", np.nan),
                mc_greeks_dict.get("vega", np.nan),
                mc_greeks_dict.get("rho", np.nan),
            ]

        greek_df = pd.DataFrame(
            greek_data,
            index=[
                "Delta",
                "Gamma",
                "Theta (per day)",
                "Vega (per % vol)",
                "Rho (per % rate)",
            ],
        )

        col_g1, col_g2 = st.columns([1, 1])
        with col_g1:
            st.write("**Current Greeks:**")
            st.dataframe(greek_df.style.format("{:.6f}"))

        with col_g2:
            st.write("**Interpretation:**")
            st.write(f"- **Delta:** {g_tree['delta']:.4f} - Price sensitivity to spot")
            st.write(f"- **Gamma:** {g_tree['gamma']:.6f} - Delta sensitivity to spot")
            st.write(f"- **Theta:** {g_tree['theta_day']:.6f} (per day) - Time decay")
            st.write(f"- **Vega:** {g_tree['vega']:.6f} - Volatility sensitivity")
            st.write(f"- **Rho:** {g_tree['rho']:.6f} - Rate sensitivity")

        st.divider()

        # Greeks visualization - Greeks vs Spot (5 graphs)
        # OPTIMIZED: use BS analytical greeks for European (instant),
        #            for American: delta/gamma from tree (1 build each),
        #            vega/rho/theta on a coarser grid then interpolated.
        st.write("**Greeks Sensitivity to Stock Price Changes**")
        try:
            n_spot = 30
            spot_range = np.linspace(inputs.S0 * 0.7, inputs.S0 * 1.3, n_spot)

            if inputs.option_class.lower() == "european":
                # ---- FAST PATH: Black-Scholes analytical (no tree needed) ----
                deltas, gammas, vegas, thetas, rhos = [], [], [], [], []
                for spot in spot_range:
                    bs_temp = BlackScholesPricer(
                        spot, inputs.K, T, inputs.r, inputs.sigma,
                        inputs.option_type,
                        dividend=inputs.dividend,
                        dividend_date=inputs.dividend_date,
                    )
                    _m = Market(stock_price=spot, int_rate=inputs.r, sigma=inputs.sigma)
                    def _pfn(_bs=bs_temp, _m=_m):
                        _bs.S, _bs.sigma, _bs.r = _m.stock_price, _m.sigma, _m.int_rate
                        return _bs.price()
                    g = GreeksCalculator(_pfn, _m).greeks()
                    deltas.append(g["delta"])
                    gammas.append(g["gamma"])
                    vegas.append(g["vega"])
                    thetas.append(g.get("theta", 0.0))
                    rhos.append(g["rho"])
                method_label = "Black-Scholes (analytical)"
            else:
                # ---- AMERICAN: optimized tree approach ----
                # Delta & Gamma: 1 tree build per spot (fast: no bumps)
                # Vega, Rho, Theta: computed on fewer points then interpolated
                deltas, gammas = [], []

                progress_greeks = st.progress(0, text="Computing Delta & Gamma...")
                for i, spot in enumerate(spot_range):
                    market_temp = Market(
                        stock_price=spot, int_rate=inputs.r, sigma=inputs.sigma,
                        div=inputs.dividend, div_date=inputs.dividend_date,
                    )
                    tree_temp = TrinomialTree(
                        market_temp, N=inputs.N, pruning=inputs.pruning,
                        epsilon=inputs.epsilon, pricing_date=inputs.pricing_date,
                    )
                    _ = tree_temp.price(option, compute_greeks=True)
                    deltas.append(tree_temp.delta())
                    gammas.append(tree_temp.gamma())
                    progress_greeks.progress((i + 1) / n_spot,
                                              text=f"Delta/Gamma: spot {i+1}/{n_spot}")

                # Vega, Rho, Theta on coarser grid (8 points) → interpolate
                n_coarse = 8
                coarse_idx = np.linspace(0, n_spot - 1, n_coarse, dtype=int)
                coarse_spots = spot_range[coarse_idx]

                vegas_coarse, rhos_coarse, thetas_coarse = [], [], []
                progress_greeks.progress(0, text="Computing Vega, Rho, Theta...")
                for i, spot in enumerate(coarse_spots):
                    market_temp = Market(
                        stock_price=spot, int_rate=inputs.r, sigma=inputs.sigma,
                        div=inputs.dividend, div_date=inputs.dividend_date,
                    )
                    tree_temp = TrinomialTree(
                        market_temp, N=inputs.N, pruning=inputs.pruning,
                        epsilon=inputs.epsilon, pricing_date=inputs.pricing_date,
                    )
                    _ = tree_temp.price(option, compute_greeks=True)
                    vegas_coarse.append(tree_temp.vega(option))
                    rhos_coarse.append(tree_temp.rho(option))
                    theta_val = compute_theta(tree_temp, market_temp, option, inputs)
                    thetas_coarse.append(theta_val)
                    progress_greeks.progress((i + 1) / n_coarse,
                                              text=f"Vega/Rho/Theta: {i+1}/{n_coarse}")
                progress_greeks.empty()

                # Interpolate to full grid
                vegas = np.interp(spot_range, coarse_spots, vegas_coarse).tolist()
                rhos = np.interp(spot_range, coarse_spots, rhos_coarse).tolist()
                thetas = np.interp(spot_range, coarse_spots, thetas_coarse).tolist()
                method_label = "Trinomial Tree (interpolated for Vega/Rho/Theta)"

            fig, axes = plt.subplots(2, 3, figsize=(15, 10))

            # Delta
            axes[0, 0].plot(spot_range, deltas, 'b-', linewidth=2, label='Delta')
            axes[0, 0].axvline(inputs.S0, color='r', linestyle='--', alpha=0.5, label=f'Current: {inputs.S0:.2f}')
            axes[0, 0].set_xlabel('Spot Price', fontsize=10)
            axes[0, 0].set_ylabel('Delta', fontsize=10)
            axes[0, 0].set_title('Delta vs Spot', fontsize=11, fontweight='bold')
            axes[0, 0].legend(fontsize=9)
            axes[0, 0].grid(True, alpha=0.3)

            # Gamma
            axes[0, 1].plot(spot_range, gammas, 'g-', linewidth=2, label='Gamma')
            axes[0, 1].axvline(inputs.S0, color='r', linestyle='--', alpha=0.5)
            axes[0, 1].set_xlabel('Spot Price', fontsize=10)
            axes[0, 1].set_ylabel('Gamma', fontsize=10)
            axes[0, 1].set_title('Gamma vs Spot', fontsize=11, fontweight='bold')
            axes[0, 1].grid(True, alpha=0.3)

            # Vega
            axes[0, 2].plot(spot_range, vegas, 'purple', linewidth=2, label='Vega')
            axes[0, 2].axvline(inputs.S0, color='r', linestyle='--', alpha=0.5)
            axes[0, 2].set_xlabel('Spot Price', fontsize=10)
            axes[0, 2].set_ylabel('Vega', fontsize=10)
            axes[0, 2].set_title('Vega vs Spot', fontsize=11, fontweight='bold')
            axes[0, 2].grid(True, alpha=0.3)

            # Theta
            axes[1, 0].plot(spot_range, thetas, 'orange', linewidth=2, label='Theta/day')
            axes[1, 0].axvline(inputs.S0, color='r', linestyle='--', alpha=0.5)
            axes[1, 0].set_xlabel('Spot Price', fontsize=10)
            axes[1, 0].set_ylabel('Theta (per day)', fontsize=10)
            axes[1, 0].set_title('Theta vs Spot', fontsize=11, fontweight='bold')
            axes[1, 0].grid(True, alpha=0.3)

            # Rho
            axes[1, 1].plot(spot_range, rhos, 'brown', linewidth=2, label='Rho')
            axes[1, 1].axvline(inputs.S0, color='r', linestyle='--', alpha=0.5)
            axes[1, 1].set_xlabel('Spot Price', fontsize=10)
            axes[1, 1].set_ylabel('Rho', fontsize=10)
            axes[1, 1].set_title('Rho vs Spot', fontsize=11, fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3)

            # Hide unused subplot and add method label
            axes[1, 2].axis('off')
            axes[1, 2].text(0.5, 0.5, f'Method: {method_label}',
                            ha='center', va='center', fontsize=10,
                            style='italic', color='gray',
                            transform=axes[1, 2].transAxes)

            plt.tight_layout()
            st.pyplot(fig, clear_figure=True)
        except Exception as e:
            st.error(f"Cannot compute Greeks vs spot: {str(e)}")


# =====================================================
# TAB 3: PLOTS
# =====================================================
with tab3:
    st.subheader("📊 Analysis Charts")

    col_plots_btn1, col_plots_btn2 = st.columns([1, 5])
    with col_plots_btn1:
        if st.button("Generate Plots", use_container_width=True, key="plots_btn"):
            st.session_state.compute_plots = True
    with col_plots_btn2:
        st.caption("Click to generate analysis charts (takes a moment)")

    st.divider()

    if not st.session_state.compute_plots:
        st.info("👉 Click **Generate Plots** button above to start analysis")
    elif MonteCarloPricer is None:
        st.error("MonteCarloPricer is not available. Cannot generate plots.")
    else:
        # Build a fresh pricer for analysis
        _plot_market, _plot_option, _plot_T = build_market_option(inputs)
        _plot_pricer = MonteCarloPricer(
            _plot_market, _plot_option, inputs.pricing_date,
            nb_steps=inputs.mc_nb_steps, nb_paths=inputs.mc_nb_paths,
            seed=inputs.mc_seed, batch_size=inputs.mc_batch_size,
            rng_method=inputs.mc_rng_method,
        )

        # =============================================================
        # 1. LONGSTAFF-SCHWARTZ REGRESSION VISUALIZATION
        # =============================================================
        st.subheader("1. Longstaff-Schwartz Regression")
        st.caption("Continuation value regression at selected time steps (ITM paths only)")

        try:
            # Generate paths
            S_reg, dt_reg, T_reg = _plot_pricer._generate_price_paths(inputs.mc_antithetic)
            discount_reg = np.exp(-_plot_market.int_rate * dt_reg)

            # Backward recursion collecting regression data at selected steps
            cashflows_reg = _plot_option.payoff(S_reg[:, -1]).copy()
            nb_steps_reg = _plot_pricer.nb_steps
            # Collect data at evenly spaced steps
            reg_steps = sorted(set([
                max(1, nb_steps_reg // 4),
                max(1, nb_steps_reg // 2),
                max(1, 3 * nb_steps_reg // 4),
            ]))
            reg_data = {}  # step -> (X_itm, Y_itm, continuation_fitted, basis_name)

            for t in range(nb_steps_reg - 1, 0, -1):
                cashflows_reg = cashflows_reg * discount_reg
                S_t = S_reg[:, t]
                intrinsic_val = _plot_option.payoff(S_t)
                itm = intrinsic_val > 0
                if np.count_nonzero(itm) > 0:
                    X_itm, Y_itm = S_t[itm], cashflows_reg[itm]
                    if len(X_itm) >= inputs.mc_degree + 1:
                        basis = _plot_pricer.regressor._basis_builder.build(
                            X_itm, inputs.mc_regression_type, inputs.mc_degree
                        )
                        try:
                            coeffs, _, _, _ = np.linalg.lstsq(basis, Y_itm, rcond=None)
                            cont_fitted = basis @ coeffs
                        except np.linalg.LinAlgError:
                            coeffs = np.polyfit(X_itm, Y_itm, min(inputs.mc_degree, 2))
                            cont_fitted = np.polyval(coeffs, X_itm)

                        if t in reg_steps:
                            reg_data[t] = (X_itm.copy(), Y_itm.copy(), cont_fitted.copy())

                        # Update cashflows (exercise decision)
                        exercise = intrinsic_val[itm] > cont_fitted
                        cashflows_reg[itm] = np.where(exercise, intrinsic_val[itm], Y_itm)

            if reg_data:
                n_plots = len(reg_data)
                fig_reg, axes_reg = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
                if n_plots == 1:
                    axes_reg = [axes_reg]

                for idx, (step, (X_r, Y_r, C_r)) in enumerate(sorted(reg_data.items())):
                    ax = axes_reg[idx]
                    time_at_step = step * dt_reg
                    # Scatter: discounted cashflows vs spot
                    ax.scatter(X_r, Y_r, alpha=0.15, s=8, color='steelblue', label='Discounted CF')
                    # Intrinsic value curve
                    x_sorted = np.sort(X_r)
                    intrinsic_sorted = _plot_option.payoff(x_sorted)
                    ax.plot(x_sorted, intrinsic_sorted, 'g--', linewidth=2, label='Intrinsic Value')
                    # Regression curve (sorted for smooth line)
                    sort_idx = np.argsort(X_r)
                    ax.plot(X_r[sort_idx], C_r[sort_idx], 'r-', linewidth=2.5, label=f'Regression ({inputs.mc_regression_type})')
                    ax.set_xlabel('Stock Price (S)', fontsize=10)
                    ax.set_ylabel('Value', fontsize=10)
                    ax.set_title(f't = {time_at_step:.3f}y  (step {step}/{nb_steps_reg})',
                                 fontsize=11, fontweight='bold')
                    ax.legend(fontsize=8, loc='upper left')
                    ax.grid(True, alpha=0.3)

                fig_reg.suptitle(
                    f'Longstaff-Schwartz Regression  |  {inputs.mc_regression_type.capitalize()} basis (deg={inputs.mc_degree})  |  '
                    f'{inputs.mc_nb_paths} paths',
                    fontsize=13, fontweight='bold', y=1.02
                )
                plt.tight_layout()
                st.pyplot(fig_reg, clear_figure=True)
            else:
                st.warning("No ITM paths found for regression visualization. Try different parameters.")

        except Exception as e:
            st.error(f"Regression plot error: {str(e)}")

        st.divider()

        # =============================================================
        # 2. MC PRICE DISTRIBUTION (Terminal Payoffs)
        # =============================================================
        st.subheader("2. Monte Carlo Payoff Distribution")
        st.caption("Distribution of discounted terminal payoffs across simulated paths")

        try:
            S_dist, dt_dist, T_dist = _plot_pricer._generate_price_paths(inputs.mc_antithetic)
            terminal_payoffs = _plot_option.payoff(S_dist[:, -1]) * np.exp(-_plot_market.int_rate * T_dist)

            fig_dist, (ax_hist, ax_term) = plt.subplots(1, 2, figsize=(14, 5))

            # Histogram of discounted payoffs
            ax_hist.hist(terminal_payoffs, bins=60, color='steelblue', edgecolor='white',
                         alpha=0.8, density=True)
            mean_payoff = np.mean(terminal_payoffs)
            ax_hist.axvline(mean_payoff, color='red', linewidth=2, linestyle='--',
                            label=f'Mean = {mean_payoff:.4f}')
            # Show % of zero payoffs
            pct_zero = np.mean(terminal_payoffs == 0) * 100
            ax_hist.set_xlabel('Discounted Payoff', fontsize=11)
            ax_hist.set_ylabel('Density', fontsize=11)
            ax_hist.set_title(f'Payoff Distribution  |  {pct_zero:.1f}% OTM at maturity', fontsize=12, fontweight='bold')
            ax_hist.legend(fontsize=10)
            ax_hist.grid(True, alpha=0.3)

            # Terminal stock price distribution
            S_terminal = S_dist[:, -1]
            ax_term.hist(S_terminal, bins=60, color='seagreen', edgecolor='white',
                         alpha=0.8, density=True)
            ax_term.axvline(inputs.K, color='red', linewidth=2, linestyle='--',
                            label=f'Strike K={inputs.K:.2f}')
            ax_term.axvline(np.mean(S_terminal), color='orange', linewidth=2, linestyle='-.',
                            label=f'Mean S(T) = {np.mean(S_terminal):.2f}')
            ax_term.set_xlabel('Stock Price at Maturity', fontsize=11)
            ax_term.set_ylabel('Density', fontsize=11)
            ax_term.set_title('Terminal Price Distribution', fontsize=12, fontweight='bold')
            ax_term.legend(fontsize=10)
            ax_term.grid(True, alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig_dist, clear_figure=True)

        except Exception as e:
            st.error(f"Distribution plot error: {str(e)}")

        st.divider()

        # =============================================================
        # 3. CONVERGENCE ANALYSIS (Price vs Number of Paths)
        # =============================================================
        st.subheader("3. Convergence Analysis")
        st.caption("MC price convergence as the number of paths increases")

        try:
            path_counts = [100, 250, 500, 1000, 2000, 5000, 10000, 20000, 50000]
            path_counts = [p for p in path_counts if p <= inputs.mc_nb_paths * 2]
            if inputs.mc_nb_paths not in path_counts:
                path_counts.append(inputs.mc_nb_paths)
            path_counts = sorted(set(path_counts))

            conv_prices = []
            conv_errors = []
            conv_times = []

            progress_bar = st.progress(0, text="Computing convergence...")
            for i, n_p in enumerate(path_counts):
                t0_conv = time.perf_counter()
                conv_pricer = MonteCarloPricer(
                    _plot_market, _plot_option, inputs.pricing_date,
                    nb_steps=inputs.mc_nb_steps, nb_paths=n_p,
                    seed=inputs.mc_seed, rng_method=inputs.mc_rng_method,
                )
                p = conv_pricer.price(
                    antithetic=inputs.mc_antithetic, method="vectoriel",
                    regression_type=inputs.mc_regression_type, degree=inputs.mc_degree,
                )
                elapsed = time.perf_counter() - t0_conv
                conv_prices.append(p)
                conv_errors.append(conv_pricer.last_std_error)
                conv_times.append(elapsed)
                progress_bar.progress((i + 1) / len(path_counts),
                                      text=f"Computed {n_p} paths...")
            progress_bar.empty()

            conv_prices = np.array(conv_prices)
            conv_errors = np.array(conv_errors)

            fig_conv, (ax_price, ax_err) = plt.subplots(1, 2, figsize=(14, 5))

            # Price convergence
            ax_price.plot(path_counts, conv_prices, 'b-o', linewidth=2, markersize=5, label='MC Price')
            ax_price.fill_between(path_counts,
                                  conv_prices - 1.96 * conv_errors,
                                  conv_prices + 1.96 * conv_errors,
                                  alpha=0.2, color='blue', label='95% CI')
            # BS reference (if European)
            if inputs.option_class.lower() == "european":
                try:
                    bs_ref = bs_price_cached(
                        inputs.S0, inputs.K, _plot_T, inputs.r, inputs.sigma,
                        inputs.option_type, inputs.dividend, inputs.dividend_date)
                    ax_price.axhline(bs_ref, color='green', linestyle='--', linewidth=2,
                                     label=f'BS = {bs_ref:.4f}')
                except Exception:
                    pass
            ax_price.set_xlabel('Number of Paths', fontsize=11)
            ax_price.set_ylabel('Option Price', fontsize=11)
            ax_price.set_title('Price Convergence', fontsize=12, fontweight='bold')
            ax_price.set_xscale('log')
            ax_price.legend(fontsize=9)
            ax_price.grid(True, alpha=0.3)

            # Standard error convergence
            ax_err.plot(path_counts, conv_errors, 'r-s', linewidth=2, markersize=5, label='Std Error')
            # Theoretical 1/sqrt(N) line
            if conv_errors[0] > 0:
                theoretical = conv_errors[0] * np.sqrt(path_counts[0]) / np.sqrt(path_counts)
                ax_err.plot(path_counts, theoretical, 'k--', linewidth=1.5, alpha=0.6,
                            label=r'$O(1/\sqrt{N})$ theoretical')
            ax_err.set_xlabel('Number of Paths', fontsize=11)
            ax_err.set_ylabel('Standard Error', fontsize=11)
            ax_err.set_title('Standard Error Convergence', fontsize=12, fontweight='bold')
            ax_err.set_xscale('log')
            ax_err.set_yscale('log')
            ax_err.legend(fontsize=9)
            ax_err.grid(True, alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig_conv, clear_figure=True)

        except Exception as e:
            st.error(f"Convergence plot error: {str(e)}")

        st.divider()

        # =============================================================
        # 4. PRICE vs STRIKE CURVE
        # =============================================================
        st.subheader("4. Price vs Strike")
        st.caption("Option price across different strike levels (Tree, BS, MC)")

        try:
            n_strikes = 20
            K_min = inputs.S0 * 0.7
            K_max = inputs.S0 * 1.3
            strikes = np.linspace(K_min, K_max, n_strikes)

            tree_curve = []
            bs_curve = []
            mc_curve = []

            progress_bar2 = st.progress(0, text="Computing price curve...")
            for i, K_val in enumerate(strikes):
                # Tree price
                try:
                    opt_k = Option(K_val, inputs.maturity, inputs.option_type, inputs.option_class)
                    t_tree = TrinomialTree(
                        _plot_market, N=inputs.N, pruning=inputs.pruning,
                        epsilon=inputs.epsilon, pricing_date=inputs.pricing_date)
                    tree_curve.append(float(t_tree.price(opt_k)))
                except Exception:
                    tree_curve.append(np.nan)

                # BS price (European only)
                if inputs.option_class.lower() == "european":
                    try:
                        bs_curve.append(bs_price_cached(
                            inputs.S0, K_val, _plot_T, inputs.r, inputs.sigma,
                            inputs.option_type, inputs.dividend, inputs.dividend_date))
                    except Exception:
                        bs_curve.append(np.nan)
                else:
                    bs_curve.append(np.nan)

                # MC price
                try:
                    opt_k_mc = Option(K_val, inputs.maturity, inputs.option_type, inputs.option_class)
                    mc_k = MonteCarloPricer(
                        _plot_market, opt_k_mc, inputs.pricing_date,
                        nb_steps=inputs.mc_nb_steps, nb_paths=min(inputs.mc_nb_paths, 5000),
                        seed=inputs.mc_seed, rng_method=inputs.mc_rng_method)
                    mc_curve.append(mc_k.price(
                        antithetic=inputs.mc_antithetic, method="vectoriel",
                        regression_type=inputs.mc_regression_type, degree=inputs.mc_degree))
                except Exception:
                    mc_curve.append(np.nan)

                progress_bar2.progress((i + 1) / n_strikes,
                                       text=f"Strike {i+1}/{n_strikes}...")
            progress_bar2.empty()

            fig_curve, ax_curve = plt.subplots(figsize=(10, 5))
            ax_curve.plot(strikes, tree_curve, 'b-o', linewidth=2, markersize=4, label='Trinomial Tree')
            if inputs.option_class.lower() == "european":
                ax_curve.plot(strikes, bs_curve, 'g--', linewidth=2, label='Black-Scholes')
            ax_curve.plot(strikes, mc_curve, 'r-s', linewidth=2, markersize=4, label='Monte Carlo LSM')
            ax_curve.axvline(inputs.K, color='gray', linestyle=':', alpha=0.7, label=f'Current K={inputs.K:.2f}')
            ax_curve.axvline(inputs.S0, color='orange', linestyle=':', alpha=0.7, label=f'Spot S0={inputs.S0:.2f}')

            # Intrinsic value reference
            if inputs.option_type == "call":
                intrinsic_ref = np.maximum(inputs.S0 - strikes, 0)
            else:
                intrinsic_ref = np.maximum(strikes - inputs.S0, 0)
            ax_curve.plot(strikes, intrinsic_ref, 'k--', alpha=0.4, linewidth=1, label='Intrinsic Value')

            ax_curve.set_xlabel('Strike Price (K)', fontsize=11)
            ax_curve.set_ylabel('Option Price', fontsize=11)
            ax_curve.set_title(
                f'{inputs.option_type.capitalize()} {inputs.option_class.capitalize()} - Price vs Strike',
                fontsize=13, fontweight='bold')
            ax_curve.legend(fontsize=9)
            ax_curve.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig_curve, clear_figure=True)

        except Exception as e:
            st.error(f"Price curve error: {str(e)}")

        st.divider()

        # =============================================================
        # 5. RUNTIME BENCHMARKING
        # =============================================================
        st.subheader("5. Runtime Benchmarking")
        st.caption("Computation time comparison across methods and path counts")

        try:
            bench_paths = [500, 1000, 2000, 5000, 10000, 20000]
            bench_paths = [p for p in bench_paths if p <= inputs.mc_nb_paths * 2]
            if not bench_paths:
                bench_paths = [500, 1000, 2000, 5000]

            bench_mc_times = []
            bench_tree_steps = [50, 100, 200, 500]
            bench_tree_times = []

            # MC runtime vs paths
            progress_bar3 = st.progress(0, text="Benchmarking MC...")
            total_bench = len(bench_paths) + len(bench_tree_steps)
            for i, n_p in enumerate(bench_paths):
                t0 = time.perf_counter()
                bp = MonteCarloPricer(
                    _plot_market, _plot_option, inputs.pricing_date,
                    nb_steps=inputs.mc_nb_steps, nb_paths=n_p,
                    seed=inputs.mc_seed, rng_method=inputs.mc_rng_method)
                bp.price(antithetic=inputs.mc_antithetic, method="vectoriel",
                         regression_type=inputs.mc_regression_type, degree=inputs.mc_degree)
                bench_mc_times.append(time.perf_counter() - t0)
                progress_bar3.progress((i + 1) / total_bench, text=f"MC {n_p} paths...")

            # Tree runtime vs steps
            for i, n_s in enumerate(bench_tree_steps):
                t0 = time.perf_counter()
                bt = TrinomialTree(
                    _plot_market, N=n_s, pruning=inputs.pruning,
                    epsilon=inputs.epsilon, pricing_date=inputs.pricing_date)
                bt.price(_plot_option)
                bench_tree_times.append(time.perf_counter() - t0)
                progress_bar3.progress((len(bench_paths) + i + 1) / total_bench,
                                       text=f"Tree N={n_s}...")
            progress_bar3.empty()

            fig_bench, (ax_mc_t, ax_tree_t) = plt.subplots(1, 2, figsize=(14, 5))

            ax_mc_t.bar(range(len(bench_paths)), bench_mc_times, color='steelblue',
                        edgecolor='white', alpha=0.85)
            ax_mc_t.set_xticks(range(len(bench_paths)))
            ax_mc_t.set_xticklabels([str(p) for p in bench_paths], fontsize=9)
            ax_mc_t.set_xlabel('Number of Paths', fontsize=11)
            ax_mc_t.set_ylabel('Runtime (seconds)', fontsize=11)
            ax_mc_t.set_title('Monte Carlo Runtime', fontsize=12, fontweight='bold')
            ax_mc_t.grid(True, alpha=0.3, axis='y')
            for i, v in enumerate(bench_mc_times):
                ax_mc_t.text(i, v + max(bench_mc_times) * 0.02, f'{v:.3f}s',
                             ha='center', fontsize=8, fontweight='bold')

            ax_tree_t.bar(range(len(bench_tree_steps)), bench_tree_times, color='seagreen',
                          edgecolor='white', alpha=0.85)
            ax_tree_t.set_xticks(range(len(bench_tree_steps)))
            ax_tree_t.set_xticklabels([str(s) for s in bench_tree_steps], fontsize=9)
            ax_tree_t.set_xlabel('Number of Steps (N)', fontsize=11)
            ax_tree_t.set_ylabel('Runtime (seconds)', fontsize=11)
            ax_tree_t.set_title('Trinomial Tree Runtime', fontsize=12, fontweight='bold')
            ax_tree_t.grid(True, alpha=0.3, axis='y')
            for i, v in enumerate(bench_tree_times):
                ax_tree_t.text(i, v + max(bench_tree_times) * 0.02, f'{v:.3f}s',
                               ha='center', fontsize=8, fontweight='bold')

            plt.tight_layout()
            st.pyplot(fig_bench, clear_figure=True)

        except Exception as e:
            st.error(f"Benchmark error: {str(e)}")


# =====================================================
# TAB 4: TREE
# =====================================================
with tab4:
    st.subheader("🌳 Trinomial Tree Visualization")
    
    st.write("**Display Settings**")
    col_tree1, col_tree2 = st.columns([1, 1])
    
    with col_tree1:
        max_depth: int = st.slider("Tree depth", min_value=10, max_value=1000, value=100, step=10)
        proba_min: float = st.number_input("Min probability", min_value=0.0, value=1e-6, format="%0.1e")
    
    with col_tree2:
        percentile_clip: int = st.slider("Clip %", 0, 20, 0)
        edge_alpha: float = st.slider("Edge alpha", 0.0, 1.0, 0.35, 0.05)
        linewidth: float = st.slider("Edge width", 0.1, 2.0, 0.4, 0.1)
    
    st.divider()

    try:
        tree = TrinomialTree(
            market,
            N=inputs.N,
            pruning=inputs.pruning,
            epsilon=inputs.epsilon,
            pricing_date=inputs.pricing_date,
        )
        _ = tree.price(option, build_tree=True)
        tree.plot_tree(
            max_depth=max_depth,
            proba_min=proba_min,
            percentile_clip=float(percentile_clip),
            edge_alpha=edge_alpha,
            linewidth=linewidth,
        )
        fig = plt.gcf()
        st.pyplot(fig, clear_figure=True)
        st.caption(f"Trinomial tree with N={inputs.N} steps | Pruning: {inputs.pruning} | epsilon={inputs.epsilon:.1e}")
    except Exception as e:
        st.error(f"Cannot display tree: {str(e)}")


# =====================================================
# TAB 5: MULTI-SEED ANALYSIS
# =====================================================
with tab5:
    st.subheader("🔄 Multi-Seed Analysis (Distribution des prix)")
    st.write("""
    Lance le pricing MC vectoriel pour chaque seed de 1 à N.
    Permet d'observer la **distribution des prix** et de calculer
    la **moyenne**, l'**écart-type** et l'**intervalle de confiance** sur les seeds.
    """)

    col_ms1, col_ms2 = st.columns([1, 1])
    with col_ms1:
        seed_min: int = st.number_input("Seed min", min_value=1, value=1, step=1, key="seed_min")
        seed_max: int = st.number_input("Seed max", min_value=2, value=100, step=10, key="seed_max")
    with col_ms2:
        st.caption(f"Seeds: {seed_min} to {seed_max} ({seed_max - seed_min + 1} runs)")
        st.caption(f"Method: vectoriel | Paths: {inputs.mc_nb_paths} | Steps: {inputs.mc_nb_steps}")
        st.caption(f"RNG: {inputs.mc_rng_method} | Regression: {inputs.mc_regression_type} deg={inputs.mc_degree}")

    if st.button("Run Multi-Seed Analysis", use_container_width=True, key="multi_seed_btn"):
        st.session_state.run_multi_seed = True

    st.divider()

    if not st.session_state.run_multi_seed:
        st.info("👉 Click **Run Multi-Seed Analysis** to start")
    elif MonteCarloPricer is None:
        st.error("MonteCarloPricer not available")
    else:
        with st.spinner(f"Running MC pricing for seeds {seed_min} to {seed_max}..."):
            df_seeds = run_multi_seed_analysis(inputs, seed_min, seed_max)

        # Statistics
        avg_price = df_seeds["Price"].mean()
        std_price = df_seeds["Price"].std(ddof=1)
        min_price = df_seeds["Price"].min()
        max_price = df_seeds["Price"].max()
        n_seeds = len(df_seeds)
        std_of_mean = std_price / np.sqrt(n_seeds)
        ci_95_low = avg_price - 1.96 * std_of_mean
        ci_95_high = avg_price + 1.96 * std_of_mean
        avg_runtime = df_seeds["Runtime"].mean()
        total_runtime = df_seeds["Runtime"].sum()

        # Display stats
        st.write("### Results")
        scol1, scol2, scol3, scol4 = st.columns(4)
        with scol1:
            st.metric("Average Price", f"{avg_price:.6f}")
        with scol2:
            st.metric("Std Dev (prices)", f"{std_price:.6f}")
        with scol3:
            st.metric("Std of Mean", f"{std_of_mean:.6f}")
        with scol4:
            st.metric("95% CI", f"[{ci_95_low:.4f}, {ci_95_high:.4f}]")

        scol5, scol6, scol7, scol8 = st.columns(4)
        with scol5:
            st.metric("Min Price", f"{min_price:.6f}")
        with scol6:
            st.metric("Max Price", f"{max_price:.6f}")
        with scol7:
            st.metric("Avg Runtime", f"{avg_runtime:.4f}s")
        with scol8:
            st.metric("Total Runtime", f"{total_runtime:.2f}s")

        st.divider()

        # Plots
        fig_ms, axes_ms = plt.subplots(1, 3, figsize=(18, 5))

        # 1. Histogram of prices
        axes_ms[0].hist(df_seeds["Price"], bins=30, color='steelblue', edgecolor='white', alpha=0.8)
        axes_ms[0].axvline(avg_price, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_price:.4f}')
        axes_ms[0].axvline(ci_95_low, color='orange', linestyle=':', linewidth=1.5, label=f'95% CI')
        axes_ms[0].axvline(ci_95_high, color='orange', linestyle=':', linewidth=1.5)
        axes_ms[0].set_xlabel('Price', fontsize=11)
        axes_ms[0].set_ylabel('Frequency', fontsize=11)
        axes_ms[0].set_title('Distribution of MC Prices across Seeds', fontsize=12, fontweight='bold')
        axes_ms[0].legend(fontsize=9)
        axes_ms[0].grid(True, alpha=0.3)

        # 2. Price vs Seed
        axes_ms[1].plot(df_seeds["Seed"], df_seeds["Price"], 'o-', markersize=3, linewidth=0.8, color='steelblue')
        axes_ms[1].axhline(avg_price, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_price:.4f}')
        axes_ms[1].fill_between(df_seeds["Seed"], ci_95_low, ci_95_high, alpha=0.15, color='orange', label='95% CI of mean')
        axes_ms[1].set_xlabel('Seed', fontsize=11)
        axes_ms[1].set_ylabel('Price', fontsize=11)
        axes_ms[1].set_title('Price vs Seed', fontsize=12, fontweight='bold')
        axes_ms[1].legend(fontsize=9)
        axes_ms[1].grid(True, alpha=0.3)

        # 3. Cumulative average
        cum_avg = df_seeds["Price"].expanding().mean()
        cum_std = df_seeds["Price"].expanding().std()
        axes_ms[2].plot(df_seeds["Seed"], cum_avg, 'b-', linewidth=2, label='Cumulative mean')
        axes_ms[2].fill_between(
            df_seeds["Seed"],
            cum_avg - 1.96 * cum_std / np.sqrt(np.arange(1, n_seeds + 1)),
            cum_avg + 1.96 * cum_std / np.sqrt(np.arange(1, n_seeds + 1)),
            alpha=0.2, color='blue', label='95% CI'
        )
        axes_ms[2].set_xlabel('Number of Seeds', fontsize=11)
        axes_ms[2].set_ylabel('Cumulative Average Price', fontsize=11)
        axes_ms[2].set_title('Convergence of Average Price', fontsize=12, fontweight='bold')
        axes_ms[2].legend(fontsize=9)
        axes_ms[2].grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig_ms, clear_figure=True)

        # Raw data table
        with st.expander("📋 Raw Data", expanded=False):
            st.dataframe(df_seeds.style.format({
                "Price": "{:.6f}",
                "StdError": "{:.6f}",
                "Runtime": "{:.4f}",
            }))


# =====================================================
# Footer
# =====================================================
st.divider()
st.markdown("""
### Notes
- **Dividend Handling:** Discrete dividends applied at the specified date
- **Black-Scholes:** Displayed only for **European** options
- **Greeks** (MC): Delta/Gamma/Vega via finite differences 
- **Monte Carlo LSM:** Longstaff-Schwartz method with polynomial regression, supports antithetic variates

**Authors:** Issam Fradi & Yassine Mannai
""")
