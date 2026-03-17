# MC-LONGSTAFF-SCHWARTZ-PRICING

Bibliothèque et application pour pricer des options américaines avec dividendes discrets en utilisant trois approches:
1. **Méthode Longstaff-Schwartz (Monte Carlo)** - Pricing par régression polynomiale, support complet des options américaines
2. **Arbre trinomial** - Approche discrète
3. **Black-Scholes** - Solution analytique pour options européennes

Une application Streamlit est fournie pour l'exploration interactive (prix, grecs, convergence, visualisations).

## ⚙️ Prérequis
- Python ≥ 3.12 (voir `pyproject.toml`)
- macOS, Linux ou Windows

Optionnel (installé automatiquement avec l'extra `[dev]`) :
- streamlit, numpy, pandas, matplotlib, scipy, pytest, black, mypy

## 📦 Installation

Installation utilisateur (bibliothèque uniquement):
```bash
pip install .
```

Installation de développement (bibliothèque + dépendances utiles à l'app/aux tests):
```bash
pip install -e .[dev]
```

Vérification rapide des tests:
```bash
pytest -q
```

## 🚀 Démarrer l'application Streamlit

Après installation (`[dev]`), lancez l'interface interactive:

```bash
python -m streamlit run streamlit_app.py
```

**Fonctionnalités UI:**
- Sélection du modèle de pricing: **Monte Carlo LSM**, Arbre trinomial, ou Black-Scholes
- Paramètres de marché: S₀, K, r, σ, maturité, type (call/put), style (européen/américain)
- Dividende discret: montant + date de paiement
- **Pour Monte Carlo:**
  - Nombre de simulations (chemins) et steps temporels
  - Type de régression: Quadratic, Laguerre, Hermite, Legendre, Chebyshev
  - Activation/désactivation variantes antithétiques
  - Calcul de la frontière d'exercice optimale
- Courbes de convergence et graphiques prix vs. strike

**Notes sur l'interface:**
- Black–Scholes disponible uniquement pour options européennes
- Base de temps: ACT/365F
- Affichage des grecs (delta, gamma, vega, theta, rho) selon le type d'option

## ✅ Tests

Tests unitaires dans `tests/` (pytest):

```bash
pytest -v
```

Couverture:
- **test_model_mc.py**: Tests complets Monte Carlo (European/American, tous types régression)
- **test_trinomial_tree.py**: Tests arbre trinomial
- **test_node.py**: Tests nœuds (internals)
- **test_funcs.py**: Tests utilitaires

## 🗂️ Structure du projet

```
MC-Longstaff-Schwartz/
├── src/
│   └── pricing/                        # Package Python principal
│       ├── __init__.py                 # Re-exports pour compatibilité
│       ├── parameters/
│       │   ├── market.py               # Paramètres de marché (avec validation)
│       │   └── option.py               # Définition de l'option (avec validation)
│       ├── brownian_motion/
│       │   └── brownian_motion.py       # Génération de chemins browniens
│       ├── models/
│       │   ├── model.py                # Classe abstraite Model
│       │   ├── black_scholes/
│       │   │   └── blackscholes.py     # Pricer Black-Scholes analytique
│       │   ├── lsm/
│       │   │   ├── model_mc.py         # Classe façade MonteCarloPricer
│       │   │   ├── pricing_mixins.py   # Mixins (européen, américain, batch, etc.)
│       │   │   ├── regression.py       # Bases polynomiales et régresseur LS
│       │   │   ├── path_generator.py   # Génération de chemins GBM
│       │   │   └── static_pricing.py   # Fonctions statiques de pricing
│       │   └── tree/
│       │       ├── trinomial_tree.py   # Pricer arbre trinomial
│       │       └── node.py            # Nœud de l'arbre
│       ├── greeks/
│       │   └── greeks.py               # GreeksCalculator (bump-and-reprice)
│       └── others/
│           └── funcs.py                # Utilitaires arbre trinomial
├── tests/                              # Tests unitaires (pytest, 126 tests)
│   ├── test_model_mc.py
│   ├── test_trinomial_tree.py
│   ├── test_node.py
│   └── test_funcs.py
├── rapport/                            # Rapport LaTeX du projet
├── streamlit_app.py                    # Application interactive Streamlit
├── pyproject.toml                      # Configuration du package
├── pytest.ini                          # Configuration pytest
└── README.md
```

## 📖 Références scientifiques

Le projet s'appuie sur:
- **Longstaff, F. A., & Schwartz, E. S. (2001).** "Valuing American Options by Simulation: A Simple Least-Squares Approach." *The Journal of Finance*, 46(1), 113–147.
  - Algorithme de base pour pricing via régression des fonctions de continuation
  - Support options avec dividendes discrets

## 👥 Auteurs

Issam Fradi — Yassine Mannai

Projet réalisé dans le cadre du cours **Pricing Options Américaines (MSc 272)**, Université Paris-Dauphine, 2025-26.
