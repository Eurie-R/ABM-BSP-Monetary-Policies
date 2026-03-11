"""
These tests validate:
  1. Bank agent initializes without errors
  2. Rate transmission direction is correct (RRP up → lending rate up, etc.)
  3. Credit supply contracts when RRP rises
  4. Model runs for N steps without crashing
  5. Data collector captures expected fields
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.bsp_model import BSPModel
from agents.bank import BankAgent
from config.parameters import RRP_RATE_DEFAULT, RRP_RATE_NEUTRAL


# =============================================================================
# Helper
# =============================================================================

def make_model(rrp_rate: float = RRP_RATE_DEFAULT) -> BSPModel:
    """Convenience factory — returns a fresh model at the given RRP rate."""
    return BSPModel(rrp_rate=rrp_rate, num_banks=3, seed=42)


def get_banks(model: BSPModel) -> list:
    """Return all BankAgent instances from a model (Mesa 3 compatible)."""
    return list(model.agents_by_type[BankAgent])


# =============================================================================
# Test 1: Initialization
# =============================================================================

def test_model_initializes():
    """Model should create with the correct number of bank agents."""
    model = make_model()
    banks = get_banks(model)
    assert len(banks) == 3, f"Expected 3 banks, got {len(banks)}"
    print("✓ test_model_initializes passed")


def test_bank_has_expected_attributes():
    """All required attributes must exist on a fresh bank agent."""
    model = make_model()
    bank = get_banks(model)[0]
    required = ["lending_rate", "deposit_rate", "reserve_ratio", "credit_supply",
                "deposits", "liquidity"]
    for attr in required:
        assert hasattr(bank, attr), f"BankAgent missing attribute: {attr}"
    print("✓ test_bank_has_expected_attributes passed")


# =============================================================================
# Test 2: Rate transmission direction
# =============================================================================

def test_lending_rate_above_rrp():
    """Lending rate must always exceed the RRP rate (banks earn a spread)."""
    for rrp in [0.02, 0.04, 0.065, 0.08]:
        model = make_model(rrp_rate=rrp)
        bank = get_banks(model)[0]
        assert bank.lending_rate > rrp, (
            f"At RRP={rrp:.1%}, lending_rate={bank.lending_rate:.4%} should exceed RRP"
        )
    print("✓ test_lending_rate_above_rrp passed")


def test_deposit_rate_below_rrp():
    """Deposit rate must always be below the RRP rate (net interest margin)."""
    for rrp in [0.03, 0.05, 0.065]:
        model = make_model(rrp_rate=rrp)
        bank = get_banks(model)[0]
        assert bank.deposit_rate < rrp, (
            f"At RRP={rrp:.1%}, deposit_rate={bank.deposit_rate:.4%} should be below RRP"
        )
    print("✓ test_deposit_rate_below_rrp passed")


def test_higher_rrp_increases_lending_rate():
    """Lending rate at high RRP must exceed lending rate at low RRP."""
    model_low = make_model(rrp_rate=0.03)
    model_high = make_model(rrp_rate=0.08)

    bank_low = get_banks(model_low)[0]
    bank_high = get_banks(model_high)[0]

    assert bank_high.lending_rate > bank_low.lending_rate, (
        "Higher RRP should produce higher lending rate"
    )
    print("✓ test_higher_rrp_increases_lending_rate passed")


# =============================================================================
# Test 3: Credit supply contraction
# =============================================================================

def test_higher_rrp_contracts_credit():
    """
    A rate hike should reduce total credit supply.
    Economic logic: higher RRP → banks hold more reserves → less to lend.
    """
    model_low = make_model(rrp_rate=0.03)
    model_high = make_model(rrp_rate=0.08)

    credit_low = sum(b.credit_supply for b in get_banks(model_low))
    credit_high = sum(b.credit_supply for b in get_banks(model_high))

    assert credit_high < credit_low, (
        f"Higher RRP should reduce credit supply. "
        f"Got low={credit_low:,.0f}, high={credit_high:,.0f}"
    )
    print(f"✓ test_higher_rrp_contracts_credit passed "
          f"(PHP {credit_low:,.0f} → PHP {credit_high:,.0f})")


def test_reserve_ratio_rises_with_rrp():
    """Reserve ratio must be higher when RRP is above neutral."""
    model_neutral = make_model(rrp_rate=RRP_RATE_NEUTRAL)
    model_high = make_model(rrp_rate=0.08)

    bank_n = get_banks(model_neutral)[0]
    bank_h = get_banks(model_high)[0]

    assert bank_h.reserve_ratio > bank_n.reserve_ratio, (
        "Reserve ratio should rise when RRP exceeds neutral"
    )
    print("✓ test_reserve_ratio_rises_with_rrp passed")


# =============================================================================
# Test 4: Model step execution
# =============================================================================

def test_model_runs_n_steps():
    """Model must run 10 steps without raising an exception."""
    model = make_model()
    try:
        for _ in range(10):
            model.step()
        print(f"✓ test_model_runs_n_steps passed (step_count={model.step_count})")
    except Exception as e:
        assert False, f"Model crashed during step: {e}"


def test_policy_shock_propagates():
    """
    After a rate hike mid-simulation, lending rate on the next step
    must reflect the new RRP rate.
    """
    model = make_model(rrp_rate=0.04)
    model.step()

    bank = get_banks(model)[0]
    rate_before = bank.lending_rate

    # Simulate a BSP hike
    model.set_rrp_rate(0.075)
    model.step()

    rate_after = bank.lending_rate
    assert rate_after > rate_before, (
        f"Lending rate should rise after hike. Before={rate_before:.4%}, After={rate_after:.4%}"
    )
    print(f"✓ test_policy_shock_propagates passed "
          f"({rate_before:.2%} → {rate_after:.2%})")


# =============================================================================
# Test 5: Data collection
# =============================================================================

def test_datacollector_captures_fields():
    """Data collector must capture all expected model-level fields."""
    model = make_model()
    model.step()
    model.step()

    df = model.datacollector.get_model_vars_dataframe()
    expected_cols = ["RRP_Rate", "Avg_Lending_Rate", "Avg_Deposit_Rate",
                     "Avg_Reserve_Ratio", "Total_Credit_Supply"]

    for col in expected_cols:
        assert col in df.columns, f"Missing column in datacollector: {col}"

    assert len(df) == 2, f"Expected 2 rows (2 steps), got {len(df)}"
    print(f"✓ test_datacollector_captures_fields passed\n{df.to_string()}")


# =============================================================================
# Run all tests
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("BSP ABM — Session 1 Smoke Tests")
    print("=" * 60)

    test_model_initializes()
    test_bank_has_expected_attributes()
    test_lending_rate_above_rrp()
    test_deposit_rate_below_rrp()
    test_higher_rrp_increases_lending_rate()
    test_higher_rrp_contracts_credit()
    test_reserve_ratio_rises_with_rrp()
    test_model_runs_n_steps()
    test_policy_shock_propagates()
    test_datacollector_captures_fields()

    print("=" * 60)
    print("All tests passed ✓")
    print("=" * 60)


# =============================================================================
# HOUSEHOLD TESTS (Session 2)
# =============================================================================

from agents.household import HouseholdAgent


def get_households(model: BSPModel) -> list:
    """Return all HouseholdAgent instances from a model."""
    return list(model.agents_by_type[HouseholdAgent])


def make_full_model(rrp_rate: float = RRP_RATE_DEFAULT) -> BSPModel:
    """Factory with both banks and households."""
    return BSPModel(rrp_rate=rrp_rate, num_banks=3, num_households=10, num_firms=0, seed=42)


def test_households_created():
    """Model should contain the correct number of household agents."""
    model = make_full_model()
    hh = get_households(model)
    assert len(hh) == 10, f"Expected 10 households, got {len(hh)}"
    print("OK  test_households_created")


def test_household_has_expected_attributes():
    """All required attributes must exist on a fresh household agent."""
    model = make_full_model()
    hh = get_households(model)[0]
    required = ["income", "savings_rate", "savings", "debt",
                "new_borrowing", "debt_repayment", "consumption"]
    for attr in required:
        assert hasattr(hh, attr), f"HouseholdAgent missing attribute: {attr}"
    print("OK  test_household_has_expected_attributes")


def test_income_is_positive():
    """All households must have positive income."""
    model = make_full_model()
    for hh in get_households(model):
        assert hh.income > 0, f"Household {hh.unique_id} has non-positive income"
    print("OK  test_income_is_positive")


def test_consumption_respects_floor():
    """Consumption must never fall below the income floor."""
    from config.parameters import HOUSEHOLD_CONSUMPTION_FLOOR
    model = make_full_model(rrp_rate=0.09)  # high rate -- stress test
    model.step()
    for hh in get_households(model):
        floor = hh.income * HOUSEHOLD_CONSUMPTION_FLOOR
        assert hh.consumption >= floor - 0.01, (  # small float tolerance
            f"Household {hh.unique_id} consumption PHP {hh.consumption:,.0f} "
            f"below floor PHP {floor:,.0f}"
        )
    print("OK  test_consumption_respects_floor")


def test_higher_rate_reduces_borrowing():
    """
    Households should borrow less when lending rate is higher.
    Economic logic: credit demand channel -- higher cost = less demand.
    """
    model_low = make_full_model(rrp_rate=0.03)
    model_high = make_full_model(rrp_rate=0.09)

    borrow_low = sum(h.new_borrowing for h in get_households(model_low))
    borrow_high = sum(h.new_borrowing for h in get_households(model_high))

    assert borrow_high < borrow_low, (
        f"Higher rate should reduce borrowing. "
        f"Low={borrow_low:,.0f}, High={borrow_high:,.0f}"
    )
    print(f"OK  test_higher_rate_reduces_borrowing "
          f"(PHP {borrow_low:,.0f} -> PHP {borrow_high:,.0f})")


def test_higher_deposit_rate_raises_savings():
    """
    Households should save more when deposit rate is higher.
    Economic logic: substitution effect -- higher return rewards patience.
    """
    model_low = make_full_model(rrp_rate=0.02)
    model_high = make_full_model(rrp_rate=0.08)

    savings_low = sum(h.savings_rate for h in get_households(model_low))
    savings_high = sum(h.savings_rate for h in get_households(model_high))

    assert savings_high > savings_low, (
        "Higher deposit rate should raise savings rate"
    )
    print(f"OK  test_higher_deposit_rate_raises_savings "
          f"(avg {savings_low/10:.2%} -> {savings_high/10:.2%})")


def test_debt_ceiling_respected():
    """
    No household should exceed the debt-to-income ceiling.
    """
    from config.parameters import HOUSEHOLD_DEBT_INCOME_RATIO_MAX
    model = make_full_model(rrp_rate=0.02)  # low rate -- max borrowing incentive
    for _ in range(12):  # run 12 steps to let debt accumulate
        model.step()
    for hh in get_households(model):
        ratio = hh.debt / hh.income
        assert ratio <= HOUSEHOLD_DEBT_INCOME_RATIO_MAX + 0.01, (
            f"Household {hh.unique_id} debt ratio {ratio:.2%} exceeds ceiling"
        )
    print("OK  test_debt_ceiling_respected")


def test_rate_hike_reduces_consumption():
    """
    After a rate hike, total household consumption should fall.
    This is the core demand-side transmission effect.
    """
    model = make_full_model(rrp_rate=0.04)
    model.step()
    consumption_before = model.total_consumption

    model.set_rrp_rate(0.08)
    model.step()
    consumption_after = model.total_consumption

    assert consumption_after < consumption_before, (
        f"Rate hike should reduce consumption. "
        f"Before=PHP {consumption_before:,.0f}, After=PHP {consumption_after:,.0f}"
    )
    print(f"OK  test_rate_hike_reduces_consumption "
          f"(PHP {consumption_before:,.0f} -> PHP {consumption_after:,.0f})")


def test_household_model_reporters():
    """DataCollector must capture all household aggregate fields."""
    model = make_full_model()
    model.step()
    df = model.datacollector.get_model_vars_dataframe()
    hh_cols = ["Avg_Savings_Rate", "Total_Consumption",
               "Total_HH_Debt", "Total_New_Borrowing"]
    for col in hh_cols:
        assert col in df.columns, f"Missing household reporter: {col}"
    print(f"OK  test_household_model_reporters\n{df[hh_cols].to_string()}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Session 2 -- Household Agent Tests")
    print("=" * 60)
    test_households_created()
    test_household_has_expected_attributes()
    test_income_is_positive()
    test_consumption_respects_floor()
    test_higher_rate_reduces_borrowing()
    test_higher_deposit_rate_raises_savings()
    test_debt_ceiling_respected()
    test_rate_hike_reduces_consumption()
    test_household_model_reporters()
    print("=" * 60)
    print("All household tests passed OK")
    print("=" * 60)


# =============================================================================
# INVESTMENT FIRM TESTS (Session 3)
# =============================================================================

from agents.investment_firm import InvestmentFirmAgent


def get_firms(model: BSPModel) -> list:
    """Return all InvestmentFirmAgent instances from a model."""
    return list(model.agents_by_type[InvestmentFirmAgent])


def make_full_model_s3(rrp_rate: float = RRP_RATE_DEFAULT) -> BSPModel:
    """Factory with banks, households, and firms."""
    return BSPModel(rrp_rate=rrp_rate, num_banks=3, num_households=10,
                    num_firms=10, seed=42)


def test_firms_created():
    """Model should contain the correct number of firm agents."""
    model = make_full_model_s3()
    firms = get_firms(model)
    assert len(firms) == 10, f"Expected 10 firms, got {len(firms)}"
    print("OK  test_firms_created")


def test_firm_has_expected_attributes():
    """All required attributes must exist on a fresh firm agent."""
    model = make_full_model_s3()
    firm = get_firms(model)[0]
    required = ["capital", "expected_roi", "hurdle_rate", "is_investing",
                "capex", "workers", "new_hires", "debt"]
    for attr in required:
        assert hasattr(firm, attr), f"InvestmentFirmAgent missing: {attr}"
    print("OK  test_firm_has_expected_attributes")


def test_hurdle_rate_above_lending_rate():
    """Hurdle rate must always exceed the lending rate (firms need a return premium)."""
    model = make_full_model_s3()
    lending = model.avg_lending_rate
    for firm in get_firms(model):
        assert firm.hurdle_rate > lending, (
            f"Firm {firm.unique_id} hurdle {firm.hurdle_rate:.2%} "
            f"should exceed lending {lending:.2%}"
        )
    print("OK  test_hurdle_rate_above_lending_rate")


def test_high_rate_stops_investment():
    """
    At very high lending rates, most/all firms should stop investing.
    Economic logic: hurdle rate exceeds every firm's expected ROI.
    """
    model = make_full_model_s3(rrp_rate=0.12)  # extreme hike
    model.step()
    pct = model.pct_firms_investing
    assert pct < 0.5, (
        f"At 12% RRP, most firms should stop investing. Got {pct:.1%} still investing"
    )
    print(f"OK  test_high_rate_stops_investment ({pct:.1%} investing at 12% RRP)")


def test_low_rate_enables_investment():
    """
    At low lending rates, most firms should invest.
    Economic logic: hurdle rate is low, most ROIs clear it.
    """
    model = make_full_model_s3(rrp_rate=0.02)
    model.step()
    pct = model.pct_firms_investing
    assert pct > 0.5, (
        f"At 2% RRP, most firms should invest. Got {pct:.1%}"
    )
    print(f"OK  test_low_rate_enables_investment ({pct:.1%} investing at 2% RRP)")


def test_rate_hike_reduces_capex():
    """
    Total capex should fall after a rate hike.
    Economic logic: investment channel -- higher cost of capital kills projects.
    """
    model_low = make_full_model_s3(rrp_rate=0.03)
    model_high = make_full_model_s3(rrp_rate=0.09)

    model_low.step()
    model_high.step()

    capex_low = model_low.total_capex
    capex_high = model_high.total_capex

    assert capex_high < capex_low, (
        f"Higher rate should reduce capex. "
        f"Low=PHP {capex_low:,.0f}, High=PHP {capex_high:,.0f}"
    )
    print(f"OK  test_rate_hike_reduces_capex "
          f"(PHP {capex_low:,.0f} -> PHP {capex_high:,.0f})")


def test_capex_respects_debt_ceiling():
    """
    Firms must not breach their debt-to-capital ceiling over time.
    """
    from config.parameters import FIRM_DEBT_CAPITAL_RATIO_MAX
    model = make_full_model_s3(rrp_rate=0.02)  # low rate -- max borrowing incentive
    for _ in range(12):
        model.step()
    for firm in get_firms(model):
        ratio = firm.debt / firm.capital if firm.capital > 0 else 0
        assert ratio <= FIRM_DEBT_CAPITAL_RATIO_MAX + 0.01, (
            f"Firm {firm.unique_id} debt ratio {ratio:.2%} exceeds ceiling"
        )
    print("OK  test_capex_respects_debt_ceiling")


def test_gdp_proxy_contracts_on_hike():
    """
    GDP proxy (consumption + capex) should fall after a rate hike.
    This is the core macro-level transmission test.
    """
    model = make_full_model_s3(rrp_rate=0.04)
    model.step()
    gdp_before = model.gdp_proxy

    model.set_rrp_rate(0.09)
    model.step()
    gdp_after = model.gdp_proxy

    assert gdp_after < gdp_before, (
        f"Rate hike should reduce GDP proxy. "
        f"Before=PHP {gdp_before:,.0f}, After=PHP {gdp_after:,.0f}"
    )
    print(f"OK  test_gdp_proxy_contracts_on_hike "
          f"(PHP {gdp_before:,.0f} -> PHP {gdp_after:,.0f})")


def test_firm_reporters_in_datacollector():
    """DataCollector must capture all firm and GDP aggregate fields."""
    model = make_full_model_s3()
    model.step()
    df = model.datacollector.get_model_vars_dataframe()
    firm_cols = ["Total_Capex", "Total_Workers",
                 "Pct_Firms_Investing", "Total_Firm_Debt", "GDP_Proxy"]
    for col in firm_cols:
        assert col in df.columns, f"Missing firm reporter: {col}"
    print(f"OK  test_firm_reporters_in_datacollector")
    print(df[firm_cols].to_string())


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Session 3 -- Investment Firm Tests")
    print("=" * 60)
    test_firms_created()
    test_firm_has_expected_attributes()
    test_hurdle_rate_above_lending_rate()
    test_high_rate_stops_investment()
    test_low_rate_enables_investment()
    test_rate_hike_reduces_capex()
    test_capex_respects_debt_ceiling()
    test_gdp_proxy_contracts_on_hike()
    test_firm_reporters_in_datacollector()
    print("=" * 60)
    print("All firm tests passed OK")
    print("=" * 60)