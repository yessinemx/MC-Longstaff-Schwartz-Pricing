import numpy as np
from enum import Enum


class RegressionType(Enum):
    """Type of polynomial basis for Longstaff-Schwartz regression."""
    QUADRATIC = "quadratic"
    LAGUERRE = "laguerre"
    HERMITE = "hermite"
    LEGENDRE = "legendre"
    CHEBYSHEV = "chebyshev"


class BasisBuilder:
    """Construction des matrices de fonctions de base pour la régression LS."""

    def build(self, X: np.ndarray, regression_type: str, degree: int) -> np.ndarray:
        """Build the basis function matrix for regression."""
        reg = regression_type.lower()
        builders = {
            "quadratic": self._polynomial,
            "laguerre": self._laguerre,
            "hermite": self._hermite,
            "legendre": self._legendre,
            "chebyshev": self._chebyshev,
        }
        builder = builders.get(reg, self._polynomial)
        return builder(X, degree)

    def _polynomial(self, X: np.ndarray, degree: int) -> np.ndarray:
        """Standard polynomial basis: 1, x, x^2, ..."""
        n = len(X)
        basis = np.zeros((n, degree + 1))
        for i in range(degree + 1):
            basis[:, i] = X ** i
        return basis

    def _laguerre(self, X: np.ndarray, degree: int) -> np.ndarray:
        """Laguerre polynomial basis."""
        n = len(X)
        basis = np.zeros((n, degree + 1))
        basis[:, 0] = 1.0
        if degree >= 1:
            basis[:, 1] = -X + 1
        if degree >= 2:
            basis[:, 2] = 0.5 * (X**2 - 4*X + 2)
        if degree >= 3:
            basis[:, 3] = (1/6) * (-X**3 + 9*X**2 - 18*X + 6)
        for i in range(4, degree + 1):
            basis[:, i] = ((2*(i-1) + 1 - X) * basis[:, i-1] - (i-1) * basis[:, i-2]) / i
        return basis

    def _hermite(self, X: np.ndarray, degree: int) -> np.ndarray:
        """Hermite polynomial basis (probabilist's)."""
        n = len(X)
        basis = np.zeros((n, degree + 1))
        basis[:, 0] = 1.0
        if degree >= 1:
            basis[:, 1] = X
        if degree >= 2:
            basis[:, 2] = X**2 - 1
        if degree >= 3:
            basis[:, 3] = X**3 - 3*X
        for i in range(4, degree + 1):
            basis[:, i] = X * basis[:, i-1] - (i-1) * basis[:, i-2]
        return basis

    def _legendre(self, X: np.ndarray, degree: int) -> np.ndarray:
        """Legendre polynomial basis with normalization."""
        n = len(X)
        basis = np.zeros((n, degree + 1))
        X_norm = 2 * (X - X.min()) / (X.max() - X.min() + 1e-10) - 1
        basis[:, 0] = 1.0
        if degree >= 1:
            basis[:, 1] = X_norm
        if degree >= 2:
            basis[:, 2] = 0.5 * (3*X_norm**2 - 1)
        if degree >= 3:
            basis[:, 3] = 0.5 * (5*X_norm**3 - 3*X_norm)
        for i in range(4, degree + 1):
            basis[:, i] = ((2*(i-1) + 1) * X_norm * basis[:, i-1] - (i-1) * basis[:, i-2]) / i
        return basis

    def _chebyshev(self, X: np.ndarray, degree: int) -> np.ndarray:
        """Chebyshev polynomial basis (first kind)."""
        n = len(X)
        basis = np.zeros((n, degree + 1))
        X_norm = 2 * (X - X.min()) / (X.max() - X.min() + 1e-10) - 1
        basis[:, 0] = 1.0
        if degree >= 1:
            basis[:, 1] = X_norm
        if degree >= 2:
            basis[:, 2] = 2*X_norm**2 - 1
        if degree >= 3:
            basis[:, 3] = 4*X_norm**3 - 3*X_norm
        for i in range(4, degree + 1):
            basis[:, i] = 2 * X_norm * basis[:, i-1] - basis[:, i-2]
        return basis


class LSRegressor:
    """Régression Longstaff-Schwartz : construit la base et ajuste les moindres carrés."""

    def __init__(self) -> None:
        self._basis_builder = BasisBuilder()

    def compute_continuation_value(
        self, X: np.ndarray, Y: np.ndarray,
        regression_type: str, degree: int
    ) -> np.ndarray:
        """Compute continuation value using polynomial regression."""
        if len(X) < degree + 1:
            return Y
        basis = self._basis_builder.build(X, regression_type, degree)
        return self._fit(basis, X, Y, degree)

    def _fit(
        self, basis: np.ndarray, X: np.ndarray,
        Y: np.ndarray, degree: int
    ) -> np.ndarray:
        """Fit least-squares regression and return continuation values."""
        try:
            coeffs, _, _, _ = np.linalg.lstsq(basis, Y, rcond=None)
            return basis @ coeffs
        except np.linalg.LinAlgError:
            coeffs = np.polyfit(X, Y, min(degree, 2))
            return np.polyval(coeffs, X)

    @staticmethod
    def static_fit(basis: np.ndarray, X: np.ndarray, Y: np.ndarray, degree: int) -> np.ndarray:
        """Fit least-squares pour le pricing statique."""
        try:
            coeffs, _, _, _ = np.linalg.lstsq(basis, Y, rcond=None)
            return basis @ coeffs
        except np.linalg.LinAlgError:
            coeffs = np.polyfit(X, Y, min(degree, 2))
            return np.polyval(coeffs, X)
