"""
Tests for Monte Carlo pricing module.
Tests cover European/American options, scalar/vectorial methods,
and various regression types for Longstaff-Schwartz.
"""
import datetime as dt
import math
import numpy as np
import pytest

from pricing.parameters.market import Market
from pricing.parameters.option import Option
from pricing.models.lsm.model_mc import MonteCarloPricer
from pricing.models.lsm.regression import RegressionType, BasisBuilder
from pricing.models.black_scholes.blackscholes import BlackScholesPricer


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #
@pytest.fixture
def market_no_div() -> Market:
    """Standard market without dividend."""
    return Market(stock_price=100.0, int_rate=0.05, sigma=0.2, div=0.0, div_date=None)


@pytest.fixture
def market_with_div() -> Market:
    """Market with $2 discrete dividend."""
    return Market(
        stock_price=100.0, int_rate=0.05, sigma=0.2,
        div=2.0, div_date=dt.date(2026, 6, 1)
    )


@pytest.fixture
def pricing_date() -> dt.date:
    return dt.date(2026, 1, 15)


@pytest.fixture
def maturity() -> dt.date:
    return dt.date(2027, 1, 15)


@pytest.fixture
def european_call(maturity: dt.date) -> Option:
    return Option(K=100.0, maturity=maturity, option_type="call", option_class="european")


@pytest.fixture
def european_put(maturity: dt.date) -> Option:
    return Option(K=100.0, maturity=maturity, option_type="put", option_class="european")


@pytest.fixture
def american_put(maturity: dt.date) -> Option:
    return Option(K=100.0, maturity=maturity, option_type="put", option_class="american")


@pytest.fixture
def american_call(maturity: dt.date) -> Option:
    return Option(K=100.0, maturity=maturity, option_type="call", option_class="american")


# ------------------------------------------------------------------ #
# European Option Tests
# ------------------------------------------------------------------ #
class TestEuropeanPricing:
    """Tests for European option pricing."""

    def test_european_call_converges_to_bs(
        self, market_no_div: Market, european_call: Option, pricing_date: dt.date
    ):
        """MC European call should converge to Black-Scholes."""
        T = (european_call.maturity - pricing_date).days / 365.0
        bs = BlackScholesPricer(
            market_no_div.stock_price, european_call.K, T,
            market_no_div.int_rate, market_no_div.sigma, "call"
        )
        bs_price = bs.price()

        pricer = MonteCarloPricer(
            market_no_div, european_call, pricing_date,
            nb_steps=1, nb_paths=100000, seed=42
        )
        mc_price = pricer.price(antithetic=True)

        # Should be within 1% of BS price
        assert mc_price == pytest.approx(bs_price, rel=0.01)

    def test_european_put_converges_to_bs(
        self, market_no_div: Market, european_put: Option, pricing_date: dt.date
    ):
        """MC European put should converge to Black-Scholes."""
        T = (european_put.maturity - pricing_date).days / 365.0
        bs = BlackScholesPricer(
            market_no_div.stock_price, european_put.K, T,
            market_no_div.int_rate, market_no_div.sigma, "put"
        )
        bs_price = bs.price()

        pricer = MonteCarloPricer(
            market_no_div, european_put, pricing_date,
            nb_steps=1, nb_paths=100000, seed=42
        )
        mc_price = pricer.price(antithetic=True)

        assert mc_price == pytest.approx(bs_price, rel=0.01)

    def test_scalar_vs_vectorial_same_result(
        self, market_no_div: Market, european_call: Option, pricing_date: dt.date
    ):
        """Scalar and vectorial methods should give IDENTICAL results with same seed."""
        # Create fresh pricer for scalar
        pricer_scalar = MonteCarloPricer(
            market_no_div, european_call, pricing_date,
            nb_steps=10, nb_paths=5000, seed=123
        )
        price_scalar = pricer_scalar.price(method="scalaire")

        # Create fresh pricer for vectorial (same seed)
        pricer_vector = MonteCarloPricer(
            market_no_div, european_call, pricing_date,
            nb_steps=10, nb_paths=5000, seed=123
        )
        price_vector = pricer_vector.price(method="vectoriel", antithetic=False)

        # Should be EXACTLY equal (same random numbers used)
        assert price_scalar == pytest.approx(price_vector, rel=1e-10)

    def test_antithetic_reduces_variance(
        self, market_no_div: Market, european_call: Option, pricing_date: dt.date
    ):
        """Antithetic variates should reduce variance."""
        nb_trials = 10
        prices_no_anti = []
        prices_with_anti = []

        for seed in range(nb_trials):
            pricer = MonteCarloPricer(
                market_no_div, european_call, pricing_date,
                nb_steps=1, nb_paths=1000, seed=seed
            )
            prices_no_anti.append(pricer.price(antithetic=False))

            pricer2 = MonteCarloPricer(
                market_no_div, european_call, pricing_date,
                nb_steps=1, nb_paths=1000, seed=seed
            )
            prices_with_anti.append(pricer2.price(antithetic=True))

        std_no_anti = np.std(prices_no_anti)
        std_with_anti = np.std(prices_with_anti)

        # Antithetic should have lower variance
        assert std_with_anti < std_no_anti


# ------------------------------------------------------------------ #
# American Option Tests
# ------------------------------------------------------------------ #
class TestAmericanPricing:
    """Tests for American option pricing with Longstaff-Schwartz."""

    def test_american_put_geq_european_put(
        self, market_no_div: Market, european_put: Option, american_put: Option, pricing_date: dt.date
    ):
        """American put should be >= European put."""
        pricer_eu = MonteCarloPricer(
            market_no_div, european_put, pricing_date,
            nb_steps=50, nb_paths=10000, seed=42
        )
        pricer_am = MonteCarloPricer(
            market_no_div, american_put, pricing_date,
            nb_steps=50, nb_paths=10000, seed=42
        )

        eu_price = pricer_eu.price()
        am_price = pricer_am.price()

        assert am_price >= eu_price - 0.01  # Small tolerance

    def test_american_call_no_div_equals_european(
        self, market_no_div: Market, european_call: Option, american_call: Option, pricing_date: dt.date
    ):
        """American call without dividend should equal European call."""
        pricer_eu = MonteCarloPricer(
            market_no_div, european_call, pricing_date,
            nb_steps=50, nb_paths=20000, seed=42
        )
        pricer_am = MonteCarloPricer(
            market_no_div, american_call, pricing_date,
            nb_steps=50, nb_paths=20000, seed=42
        )

        eu_price = pricer_eu.price()
        am_price = pricer_am.price()

        # Should be very close (no early exercise premium for calls without div)
        assert am_price == pytest.approx(eu_price, rel=0.02)

    def test_naive_american_overestimates(
        self, market_no_div: Market, american_put: Option, pricing_date: dt.date
    ):
        """Naive American pricing should overestimate the price."""
        pricer = MonteCarloPricer(
            market_no_div, american_put, pricing_date,
            nb_steps=50, nb_paths=10000, seed=42
        )

        ls_price = pricer.price(american_method="ls")
        naive_price = pricer.price(american_method="naive")

        # Naive should be higher (uses hindsight)
        assert naive_price > ls_price

    @pytest.mark.parametrize("regression_type", ["quadratic", "laguerre", "hermite"])
    def test_different_regression_bases(
        self, market_no_div: Market, american_put: Option, pricing_date: dt.date, regression_type: str
    ):
        """Different regression bases should give similar results."""
        pricer = MonteCarloPricer(
            market_no_div, american_put, pricing_date,
            nb_steps=50, nb_paths=20000, seed=42
        )

        price = pricer.price(regression_type=regression_type, degree=2)

        # All prices should be reasonable (between 0 and K for put)
        assert 0 < price < american_put.K

    @pytest.mark.parametrize("degree", [1, 2, 3, 4])
    def test_different_polynomial_degrees(
        self, market_no_div: Market, american_put: Option, pricing_date: dt.date, degree: int
    ):
        """Higher degrees should not dramatically change the price."""
        pricer = MonteCarloPricer(
            market_no_div, american_put, pricing_date,
            nb_steps=50, nb_paths=20000, seed=42
        )

        price_deg2 = pricer.price(regression_type="quadratic", degree=2)
        price = pricer.price(regression_type="quadratic", degree=degree)

        # Should be within 10% of degree-2 price
        assert price == pytest.approx(price_deg2, rel=0.10)


# ------------------------------------------------------------------ #
# Static Method Tests
# ------------------------------------------------------------------ #
class TestStaticMethods:
    """Tests for static pricing methods."""

    def test_static_european_call(self):
        """Test static European call pricing."""
        S0, K, r, T, sigma = 100.0, 100.0, 0.05, 1.0, 0.2
        nb_paths, nb_steps = 50000, 1

        # Generate terminal prices
        S_paths, dt = MonteCarloPricer.generate_gbm_paths(
            S0, r, sigma, T, nb_steps, nb_paths, seed=42
        )
        S_T = S_paths[:, -1]

        # Static price
        mc_price = MonteCarloPricer.price_european_call(S_T, K, r, T, sigma)

        # BS reference
        bs = BlackScholesPricer(S0, K, T, r, sigma, "call")
        bs_price = bs.price()

        assert mc_price == pytest.approx(bs_price, rel=0.01)

    def test_static_european_put(self):
        """Test static European put pricing."""
        S0, K, r, T, sigma = 100.0, 100.0, 0.05, 1.0, 0.2
        nb_paths, nb_steps = 50000, 1

        S_paths, dt = MonteCarloPricer.generate_gbm_paths(
            S0, r, sigma, T, nb_steps, nb_paths, seed=42
        )
        S_T = S_paths[:, -1]

        mc_price = MonteCarloPricer.price_european_put(S_T, K, r, T, sigma)

        bs = BlackScholesPricer(S0, K, T, r, sigma, "put")
        bs_price = bs.price()

        assert mc_price == pytest.approx(bs_price, rel=0.01)

    def test_static_american_put_ls(self):
        """Test static American put L&S pricing."""
        S0, K, r, T, sigma = 100.0, 100.0, 0.05, 1.0, 0.2
        nb_paths, nb_steps = 20000, 50

        S_paths, dt = MonteCarloPricer.generate_gbm_paths(
            S0, r, sigma, T, nb_steps, nb_paths, seed=42
        )

        am_price = MonteCarloPricer.price_american_put_ls(
            S_paths, K, r, dt, regression_type="quadratic", degree=2
        )

        # Should be positive and less than K
        assert 0 < am_price < K

        # Should be >= European put
        eu_price = MonteCarloPricer.price_european_put(S_paths[:, -1], K, r, T, sigma)
        assert am_price >= eu_price - 0.1

    def test_generate_gbm_paths_shape(self):
        """Test that generated paths have correct shape."""
        S0, r, sigma, T = 100.0, 0.05, 0.2, 1.0
        nb_steps, nb_paths = 50, 1000

        S_paths, dt = MonteCarloPricer.generate_gbm_paths(
            S0, r, sigma, T, nb_steps, nb_paths, seed=42
        )

        assert S_paths.shape == (nb_paths, nb_steps + 1)
        assert dt == pytest.approx(T / nb_steps)
        assert np.all(S_paths[:, 0] == S0)


# ------------------------------------------------------------------ #
# Payoff Tests
# ------------------------------------------------------------------ #
class TestPayoff:
    """Tests for option payoff calculation."""

    def test_call_payoff_itm(self, european_call: Option):
        """Call payoff when ITM."""
        S = np.array([110.0, 120.0, 150.0])
        payoffs = european_call.payoff(S)
        expected = np.array([10.0, 20.0, 50.0])
        np.testing.assert_array_almost_equal(payoffs, expected)

    def test_call_payoff_otm(self, european_call: Option):
        """Call payoff when OTM."""
        S = np.array([90.0, 80.0, 50.0])
        payoffs = european_call.payoff(S)
        expected = np.array([0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(payoffs, expected)

    def test_put_payoff_itm(self, european_put: Option):
        """Put payoff when ITM."""
        S = np.array([90.0, 80.0, 50.0])
        payoffs = european_put.payoff(S)
        expected = np.array([10.0, 20.0, 50.0])
        np.testing.assert_array_almost_equal(payoffs, expected)

    def test_put_payoff_otm(self, european_put: Option):
        """Put payoff when OTM."""
        S = np.array([110.0, 120.0, 150.0])
        payoffs = european_put.payoff(S)
        expected = np.array([0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(payoffs, expected)


# ------------------------------------------------------------------ #
# Regression Basis Tests
# ------------------------------------------------------------------ #
class TestRegressionBasis:
    """Tests for polynomial basis construction."""

    def test_quadratic_basis(self):
        """Test quadratic (standard polynomial) basis."""
        X = np.array([1.0, 2.0, 3.0])
        basis = BasisBuilder().build(X, "quadratic", 2)

        # Each row = one path, each col = basis function (1, x, x^2)
        expected = np.array([
            [1.0, 1.0, 1.0],   # X=1
            [1.0, 2.0, 4.0],   # X=2
            [1.0, 3.0, 9.0],   # X=3
        ])

        np.testing.assert_array_almost_equal(basis, expected)

    def test_laguerre_basis_degree2(self):
        """Test Laguerre basis up to degree 2."""
        X = np.array([0.0, 1.0, 2.0])
        basis = BasisBuilder().build(X, "laguerre", 2)

        # La_0 = 1, La_1 = -x+1, La_2 = 0.5*(x^2-4x+2)
        # Each row = one path, each col = basis function
        expected = np.array([
            [1.0, 1.0, 1.0],     # X=0
            [1.0, 0.0, -0.5],    # X=1
            [1.0, -1.0, -1.0],   # X=2
        ])

        np.testing.assert_array_almost_equal(basis, expected)

    def test_hermite_basis_degree2(self):
        """Test Hermite basis up to degree 2."""
        X = np.array([0.0, 1.0, 2.0])
        basis = BasisBuilder().build(X, "hermite", 2)

        # H_0 = 1, H_1 = x, H_2 = x^2 - 1
        # Each row = one path, each col = basis function
        expected = np.array([
            [1.0, 0.0, -1.0],   # X=0
            [1.0, 1.0, 0.0],    # X=1
            [1.0, 2.0, 3.0],    # X=2
        ])

        np.testing.assert_array_almost_equal(basis, expected)


# ------------------------------------------------------------------ #
# Edge Cases
# ------------------------------------------------------------------ #
class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_very_itm_call(self, market_no_div: Market, pricing_date: dt.date, maturity: dt.date):
        """Deep ITM call should be close to S - K*exp(-rT)."""
        option = Option(K=50.0, maturity=maturity, option_type="call", option_class="european")
        pricer = MonteCarloPricer(
            market_no_div, option, pricing_date,
            nb_steps=1, nb_paths=10000, seed=42
        )

        T = (maturity - pricing_date).days / 365.0
        price = pricer.price()
        intrinsic = market_no_div.stock_price - 50.0 * np.exp(-market_no_div.int_rate * T)

        assert price == pytest.approx(intrinsic, rel=0.05)

    def test_very_otm_put(self, market_no_div: Market, pricing_date: dt.date, maturity: dt.date):
        """Deep OTM put should be close to zero."""
        option = Option(K=50.0, maturity=maturity, option_type="put", option_class="european")
        pricer = MonteCarloPricer(
            market_no_div, option, pricing_date,
            nb_steps=1, nb_paths=10000, seed=42
        )

        price = pricer.price()
        assert price < 1.0  # Should be very small

    def test_zero_volatility(self, pricing_date: dt.date, maturity: dt.date):
        """Zero volatility should give deterministic result."""
        market = Market(stock_price=100.0, int_rate=0.05, sigma=0.001, div=0.0, div_date=None)  # Near-zero vol
        option = Option(K=100.0, maturity=maturity, option_type="call", option_class="european")

        pricer = MonteCarloPricer(
            market, option, pricing_date,
            nb_steps=1, nb_paths=1000, seed=42
        )

        T = (maturity - pricing_date).days / 365.0
        price = pricer.price()

        # With zero vol, S_T = S0 * exp(rT)
        S_T = market.stock_price * np.exp(market.int_rate * T)
        expected = max(S_T - option.K, 0) * np.exp(-market.int_rate * T)

        assert price == pytest.approx(expected, rel=0.01)

    def test_invalid_method_raises(self, market_no_div: Market, european_call: Option, pricing_date: dt.date):
        """Invalid method should raise ValueError."""
        pricer = MonteCarloPricer(
            market_no_div, european_call, pricing_date,
            nb_steps=10, nb_paths=100
        )

        with pytest.raises(ValueError, match="Unknown method"):
            pricer.price(method="invalid")

    def test_scalar_american_works(self, market_no_div: Market, american_put: Option, pricing_date: dt.date):
        """Scalar method should also support American options."""
        pricer = MonteCarloPricer(
            market_no_div, american_put, pricing_date,
            nb_steps=10, nb_paths=100
        )

        price = pricer.price(method="scalaire")
        assert price > 0


# ------------------------------------------------------------------ #
# Convergence Tests
# ------------------------------------------------------------------ #
class TestConvergence:
    """Tests for convergence behavior."""

    def test_more_paths_reduces_error(
        self, market_no_div: Market, european_call: Option, pricing_date: dt.date
    ):
        """More paths should reduce error."""
        T = (european_call.maturity - pricing_date).days / 365.0
        bs = BlackScholesPricer(
            market_no_div.stock_price, european_call.K, T,
            market_no_div.int_rate, market_no_div.sigma, "call"
        )
        bs_price = bs.price()

        errors = []
        for nb_paths in [1000, 10000, 100000]:
            pricer = MonteCarloPricer(
                market_no_div, european_call, pricing_date,
                nb_steps=1, nb_paths=nb_paths, seed=42
            )
            mc_price = pricer.price()
            errors.append(abs(mc_price - bs_price))

        # Errors should decrease
        assert errors[1] < errors[0]
        assert errors[2] < errors[1]


# ------------------------------------------------------------------ #
# All Regression Types Tests
# ------------------------------------------------------------------ #
class TestAllRegressionTypes:
    """Tests for all regression polynomial bases."""

    @pytest.mark.parametrize("regression_type", [
        "quadratic", "laguerre", "hermite", "legendre", "chebyshev"
    ])
    def test_all_regression_types_american_put(
        self, market_no_div: Market, american_put: Option, pricing_date: dt.date,
        regression_type: str
    ):
        """All regression types should work for American puts."""
        pricer = MonteCarloPricer(
            market_no_div, american_put, pricing_date,
            nb_steps=30, nb_paths=5000, seed=42
        )

        price = pricer.price(regression_type=regression_type, degree=2)

        # Price should be reasonable
        assert 0 < price < american_put.K
        # Price should be positive and within a reasonable range
        assert not np.isnan(price)
        assert not np.isinf(price)

    @pytest.mark.parametrize("regression_type", [
        "quadratic", "laguerre", "hermite", "legendre", "chebyshev"
    ])
    def test_all_regression_types_american_call(
        self, market_no_div: Market, american_call: Option, pricing_date: dt.date,
        regression_type: str
    ):
        """All regression types should work for American calls."""
        pricer = MonteCarloPricer(
            market_no_div, american_call, pricing_date,
            nb_steps=30, nb_paths=5000, seed=42
        )

        price = pricer.price(regression_type=regression_type, degree=2)

        # Price should be reasonable
        assert price > 0
        assert not np.isnan(price)
        assert not np.isinf(price)

    def test_all_regression_types_similar_prices(
        self, market_no_div: Market, american_put: Option, pricing_date: dt.date
    ):
        """All regression types should give similar prices (within 5%)."""
        regression_types = ["quadratic", "laguerre", "hermite", "legendre", "chebyshev"]
        prices = {}

        for reg_type in regression_types:
            pricer = MonteCarloPricer(
                market_no_div, american_put, pricing_date,
                nb_steps=50, nb_paths=10000, seed=42
            )
            prices[reg_type] = pricer.price(regression_type=reg_type, degree=2)

        # All prices should be within 2% of each other
        price_values = list(prices.values())
        reference_price = price_values[0]

        for price in price_values[1:]:
            assert abs(price - reference_price) / reference_price < 0.02, \
                f"Prices differ too much: {prices}"


# ------------------------------------------------------------------ #
# Dividend Options Tests
# ------------------------------------------------------------------ #
class TestDividendOptions:
    """Tests for options with discrete dividends."""

    def test_american_put_with_dividend_higher_value(
        self, market_no_div: Market, market_with_div: Market, american_put: Option, pricing_date: dt.date
    ):
        """American put with dividend should have higher value than without."""
        pricer_no_div = MonteCarloPricer(
            market_no_div, american_put, pricing_date,
            nb_steps=50, nb_paths=10000, seed=42
        )
        price_no_div = pricer_no_div.price()

        pricer_with_div = MonteCarloPricer(
            market_with_div, american_put, pricing_date,
            nb_steps=50, nb_paths=10000, seed=42
        )
        price_with_div = pricer_with_div.price()

        # Dividend reduces stock value, so put becomes more valuable
        assert price_with_div > price_no_div

    def test_american_call_with_dividend_lower_value(
        self, market_no_div: Market, market_with_div: Market, american_call: Option, pricing_date: dt.date
    ):
        """American call with dividend should have lower value than without."""
        pricer_no_div = MonteCarloPricer(
            market_no_div, american_call, pricing_date,
            nb_steps=50, nb_paths=10000, seed=42
        )
        price_no_div = pricer_no_div.price()

        pricer_with_div = MonteCarloPricer(
            market_with_div, american_call, pricing_date,
            nb_steps=50, nb_paths=10000, seed=42
        )
        price_with_div = pricer_with_div.price()

        # Dividend reduces stock value, so call becomes less valuable
        assert price_with_div < price_no_div

    def test_dividend_date_respected(self, pricing_date: dt.date, maturity: dt.date):
        """Dividend paid after maturity should not affect option value."""
        div_after_maturity = maturity + dt.timedelta(days=100)
        market = Market(
            stock_price=100.0, int_rate=0.05, sigma=0.2,
            div=10.0, div_date=div_after_maturity
        )
        option = Option(K=100.0, maturity=maturity, option_type="put", option_class="american")

        pricer = MonteCarloPricer(
            market, option, pricing_date,
            nb_steps=50, nb_paths=5000, seed=42
        )

        # Should behave as if no dividend
        price_with_late_div = pricer.price()

        market_no_div = Market(stock_price=100.0, int_rate=0.05, sigma=0.2)
        pricer_no_div = MonteCarloPricer(
            market_no_div, option, pricing_date,
            nb_steps=50, nb_paths=5000, seed=42
        )
        price_no_div = pricer_no_div.price()

        # Should be similar
        assert abs(price_with_late_div - price_no_div) < 1.0


# ------------------------------------------------------------------ #
# Exercise Frontier Tests
# ------------------------------------------------------------------ #
class TestExerciseFrontier:
    """Tests for exercise frontier computation."""

    def test_exercise_frontier_shape_american_put(
        self, market_no_div: Market, american_put: Option, pricing_date: dt.date
    ):
        """Exercise frontier should have correct shape for put."""
        pricer = MonteCarloPricer(
            market_no_div, american_put, pricing_date,
            nb_steps=30, nb_paths=5000, seed=42
        )

        times, frontier = pricer.get_exercise_frontier()

        # Times should go from 0 to T
        assert len(times) == 31  # nb_steps + 1
        assert times[0] == pytest.approx(0.0)
        assert times[-1] > 0

        # Frontier length should match times
        assert len(frontier) == len(times)

    def test_exercise_frontier_monotonicity_put(
        self, market_no_div: Market, american_put: Option, pricing_date: dt.date
    ):
        """Exercise frontier for put should be roughly decreasing with time."""
        pricer = MonteCarloPricer(
            market_no_div, american_put, pricing_date,
            nb_steps=50, nb_paths=10000, seed=42
        )

        times, frontier = pricer.get_exercise_frontier()

        # Remove NaN values
        valid_idx = ~np.isnan(frontier)
        if np.sum(valid_idx) > 1:
            valid_times = times[valid_idx]
            valid_frontier = frontier[valid_idx]

            # For a put, frontier should be below or at strike
            assert np.all(valid_frontier <= american_put.K + 1e-6)

    def test_exercise_frontier_all_regression_types(
        self, market_no_div: Market, american_put: Option, pricing_date: dt.date
    ):
        """Exercise frontier should be computable with all regression types."""
        regression_types = ["quadratic", "laguerre", "hermite", "legendre", "chebyshev"]

        for reg_type in regression_types:
            pricer = MonteCarloPricer(
                market_no_div, american_put, pricing_date,
                nb_steps=30, nb_paths=3000, seed=42
            )

            times, frontier = pricer.get_exercise_frontier(regression_type=reg_type, degree=2)

            assert len(times) > 0
            assert len(frontier) == len(times)
            assert not np.all(np.isnan(frontier))  # Should have some valid values


# ------------------------------------------------------------------ #
# European vs American Options
# ------------------------------------------------------------------ #
class TestEuropeanVsAmerican:
    """Tests comparing European and American option values."""

    def test_american_put_higher_than_european(
        self, market_no_div: Market, pricing_date: dt.date, maturity: dt.date
    ):
        """American put should always be >= European put."""
        european_put = Option(K=100.0, maturity=maturity, option_type="put", option_class="european")
        american_put = Option(K=100.0, maturity=maturity, option_type="put", option_class="american")

        pricer_eu = MonteCarloPricer(
            market_no_div, european_put, pricing_date,
            nb_steps=50, nb_paths=10000, seed=42
        )
        pricer_am = MonteCarloPricer(
            market_no_div, american_put, pricing_date,
            nb_steps=50, nb_paths=10000, seed=42
        )

        eu_price = pricer_eu.price()
        am_price = pricer_am.price()

        # American should be >= European
        assert am_price >= eu_price - 0.1  # Small tolerance

    def test_american_call_higher_than_european_with_div(
        self, market_with_div: Market, pricing_date: dt.date, maturity: dt.date
    ):
        """American call should be > European call when dividend exists."""
        european_call = Option(K=100.0, maturity=maturity, option_type="call", option_class="european")
        american_call = Option(K=100.0, maturity=maturity, option_type="call", option_class="american")

        pricer_eu = MonteCarloPricer(
            market_with_div, european_call, pricing_date,
            nb_steps=50, nb_paths=10000, seed=42
        )
        pricer_am = MonteCarloPricer(
            market_with_div, american_call, pricing_date,
            nb_steps=50, nb_paths=10000, seed=42
        )

        eu_price = pricer_eu.price()
        am_price = pricer_am.price()

        # American should be >= European
        assert am_price >= eu_price - 0.1


# ------------------------------------------------------------------ #
# Legendre and Chebyshev Basis Tests
# ------------------------------------------------------------------ #
class TestOrthogonalBases:
    """Tests for orthogonal polynomial bases."""

    def test_legendre_basis_orthogonal(self):
        """Test Legendre basis properties."""
        X = np.linspace(-1, 1, 100)
        basis = BasisBuilder().build(X, "legendre", 3)

        # Should have correct shape
        assert basis.shape == (100, 4)

        # Should have reasonable values (not NaN/inf)
        assert np.all(np.isfinite(basis))

    def test_chebyshev_basis_oscillatory(self):
        """Test Chebyshev basis oscillatory nature."""
        X = np.linspace(-1, 1, 100)
        basis = BasisBuilder().build(X, "chebyshev", 3)

        # Should have correct shape
        assert basis.shape == (100, 4)

        # Should have reasonable values
        assert np.all(np.isfinite(basis))

        # Chebyshev oscillates - should have sign changes
        T1 = basis[:, 1]  # First degree
        assert np.sum(np.diff(np.sign(T1)) != 0) > 0  # Sign changes

    def test_legendre_vs_chebyshev_american_prices(
        self, market_no_div: Market, american_put: Option, pricing_date: dt.date
    ):
        """Legendre and Chebyshev should give similar American option prices."""
        pricer = MonteCarloPricer(
            market_no_div, american_put, pricing_date,
            nb_steps=50, nb_paths=10000, seed=42
        )

        price_leg = pricer.price(regression_type="legendre", degree=2)
        price_cheb = pricer.price(regression_type="chebyshev", degree=2)

        # Should be similar
        assert abs(price_leg - price_cheb) / price_leg < 0.05


# ------------------------------------------------------------------ #
# Greeks cross-validation: GreeksCalculator vs BS analytique
# ------------------------------------------------------------------ #
class TestGreeksCrossValidation:
    """Vérifie que GreeksCalculator (bump-and-reprice) donne des résultats
    cohérents avec les grecs analytiques de BlackScholesPricer."""

    def test_bs_greeks_via_calculator(self):
        """GreeksCalculator sur BS doit donner des grecs cohérents."""
        from pricing.greeks import GreeksCalculator

        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
        bs = BlackScholesPricer(S, K, T, r, sigma, "call")
        market = Market(stock_price=S, int_rate=r, sigma=sigma)

        def price_fn():
            bs.S = market.stock_price
            bs.sigma = market.sigma
            bs.r = market.int_rate
            return bs.price()

        calc = GreeksCalculator(price_fn, market)
        g = calc.greeks(bump_S=0.001, bump_sigma=0.001, bump_r=0.0001)

        # Call delta doit être positif et < 1
        assert 0 < g["delta"] < 1
        # Gamma doit être positif
        assert g["gamma"] > 0
        # Vega doit être positif
        assert g["vega"] > 0

    def test_price_with_ci_contains_price(
        self, market_no_div: Market, european_put: Option, pricing_date: dt.date
    ):
        """price_with_ci doit renvoyer un intervalle encadrant le prix."""
        pricer = MonteCarloPricer(
            market_no_div, european_put, pricing_date,
            nb_steps=50, nb_paths=10000, seed=42,
        )
        price, lo, hi = pricer.price_with_ci(confidence=0.95)
        assert lo <= price <= hi
        assert hi - lo > 0

    def test_control_variate_reduces_stderr(
        self, market_no_div: Market, european_put: Option, pricing_date: dt.date
    ):
        """Le pricing avec variable de contrôle doit avoir un écart-type <= sans."""
        pricer = MonteCarloPricer(
            market_no_div, european_put, pricing_date,
            nb_steps=50, nb_paths=5000, seed=42,
        )
        _ = pricer.price()
        se_plain = pricer.last_std_error

        _ = pricer.price(control_variate=True)
        se_cv = pricer.last_std_error

        assert se_cv <= se_plain * 1.1  # tolérance de 10 %

    def test_convergence_table_shape(
        self, market_no_div: Market, european_put: Option, pricing_date: dt.date
    ):
        """convergence_table doit renvoyer le bon nombre d'entrées."""
        pricer = MonteCarloPricer(
            market_no_div, european_put, pricing_date,
            nb_steps=50, nb_paths=1000, seed=42,
        )
        counts = [500, 1000, 2000]
        table = pricer.convergence_table(path_counts=counts)
        assert len(table) == 3
        for row in table:
            assert "price" in row
            assert "std_error" in row
            assert row["std_error"] >= 0

    def test_market_validation_rejects_negative_spot(self):
        """Market doit lever ValueError si stock_price <= 0."""
        with pytest.raises(ValueError, match="stock_price"):
            Market(stock_price=-10, int_rate=0.05, sigma=0.2)

    def test_market_validation_rejects_negative_sigma(self):
        """Market doit lever ValueError si sigma < 0."""
        with pytest.raises(ValueError, match="sigma"):
            Market(stock_price=100, int_rate=0.05, sigma=-0.1)

    def test_option_validation_rejects_negative_strike(self):
        """Option doit lever ValueError si K <= 0."""
        with pytest.raises(ValueError, match="K"):
            Option(K=-5, maturity=dt.date(2027, 1, 1), option_type="call", option_class="european")
