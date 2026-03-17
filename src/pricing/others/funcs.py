# Fonctions utilitaires pour le pricing par arbre trinomial
import numpy as np
from typing import Generator, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..models.tree.node import Node
    from ..models.tree.trinomial_tree import TrinomialTree


def iter_column(node: "Node") -> Generator:
    """Iterate over all nodes in a column (vertical chain)."""
    if node is None:
        return
    # Yield le noeud de départ puis parcours vertical
    yield node
    yield from _traverse_direction(node.up, "up")
    yield from _traverse_direction(node.down, "down")


def _traverse_direction(start: "Node", direction: str) -> Generator:
    """Parcourt les noeuds dans une direction (up ou down)."""
    current = start
    while current:
        yield current
        # Passage au noeud suivant dans la direction
        current = current.up if direction == "up" else current.down


def compute_variance(S: float, r: float, delta_t: float, sigma: float) -> float:
    """
    Compute the variance of the stock price at the next time step.
    Var(S_{t+dt}) = S^2 * exp(2*r*dt) * (exp(sigma^2*dt) - 1)
    """
    return S**2 * np.exp(2 * r * delta_t) * (np.exp(sigma**2 * delta_t) - 1)


def compute_forward(S: float, r: float, delta_t: float) -> float:
    """
    Compute the forward price of the stock.
    F = S * exp(r * dt)
    """
    return S * np.exp(r * delta_t)


# Calcul des probabilités trinomiales à partir des moments
def compute_probabilities(
    esperance: float,
    forward: float,
    variance: float,
    alpha: float,
    dividend: bool = False
) -> Tuple[float, float, float]:
    """Compute trinomial probabilities (p_down, p_up, p_mid)."""
    u = alpha
    d = 1.0 / alpha
    # Prix de référence et second moment
    S_mid = esperance if not dividend else forward
    E_S2 = variance + esperance**2
    if S_mid <= 0:
        return (1/3, 1/3, 1/3)
    # Calcul des moments normalisés et résolution
    m1 = esperance / S_mid
    m2 = E_S2 / (S_mid**2)
    return _solve_proba_system(u, d, m1, m2)


def _solve_proba_system(
    u: float, d: float, m1: float, m2: float
) -> Tuple[float, float, float]:
    """Résout le système linéaire pour les probabilités trinomiales."""
    # Coefficients du système 2x2
    a11, a12 = u - 1, d - 1
    a21, a22 = u**2 - 1, d**2 - 1
    b1, b2 = m1 - 1, m2 - 1
    det = a11 * a22 - a12 * a21
    if abs(det) < 1e-12:
        # Fallback : probabilités uniformes
        return (1/3, 1/3, 1/3)
    # Résolution par la règle de Cramer
    p_up = (b1 * a22 - b2 * a12) / det
    p_down = (a11 * b2 - a21 * b1) / det
    p_mid = 1.0 - p_up - p_down
    return (p_down, p_up, p_mid)


def probas_valid(tree: "TrinomialTree") -> bool:
    """
    Validate that probabilities are within [0, 1] bounds.
    Raises ValueError if any probability is out of bounds.
    """
    eps = 1e-9
    for p, name in [(tree.p_up, "p_up"), (tree.p_mid, "p_mid"), (tree.p_down, "p_down")]:
        if p < -eps or p > 1 + eps:
            raise ValueError(f"Probability {name} = {p:.6f} is out of bounds [0, 1]")
    return True
