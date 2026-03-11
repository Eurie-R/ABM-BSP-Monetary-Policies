"""
Investment Firm Agent for the BSP Monetary Policy ABM.

Economic Role:
--------------
Firms are the THIRD transmission node of BSP monetary policy.
They borrow to fund capital expenditure (capex) and hiring. Their
core decision rule is the hurdle rate comparison:

    Invest if: expected_ROI > lending_rate + hurdle_spread
    Cut capex if: expected_ROI <= lending_rate + hurdle_spread

When BSP hikes rates:
  1. Lending rate rises (via BankAgent repricing)
  2. Hurdle rate rises with it
  3. Fewer investment projects clear the hurdle
  4. Capex contracts, hiring freezes or reverses
  5. GDP proxy (total investment) falls

Heterogeneity:
--------------
Firms differ in two key ways:
  - Capital base  -- large vs small enterprise
  - Expected ROI  -- optimistic vs conservative growth expectations

This means a rate hike hits small firms (lower capital buffer) and
conservative firms (lower ROI) hardest -- consistent with real-world
evidence from BSP business surveys.

Transmission chain:
    BSP RRP up
      -> Bank lending rate up
        -> Firm hurdle rate rises
          -> Capex cut, hiring frozen
            -> Total investment falls (GDP proxy contracts)

Assumptions (flag for future revision):
-----------------------------------------
- ROI is fixed (no demand feedback loop yet -- firms don't react to
  falling household consumption)
- All investment is debt-financed (no retained earnings)
- Labor market is simplified -- workers = f(capex), no wage dynamics
- No firm exit or entry modeled at this stage
"""

from mesa import Agent
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from model.bsp_model import BSPModel

from config.parameters import (
    FIRM_CAPITAL_MEAN,
    FIRM_CAPITAL_STD,
    FIRM_CAPITAL_MIN,
    FIRM_HURDLE_RATE_SPREAD,
    FIRM_ROI_MEAN,
    FIRM_ROI_STD,
    FIRM_CAPEX_BASE_RATE,
    FIRM_CAPEX_SENSITIVITY,
    FIRM_CAPEX_MIN_RATE,
    FIRM_HIRING_CAPEX_RATIO,
    FIRM_INITIAL_WORKERS,
    FIRM_DEBT_CAPITAL_RATIO_MAX,
    FIRM_DEBT_REPAY_RATE,
    FIRM_DEPRECIATION_RATE,
    RRP_RATE_NEUTRAL,
)


class InvestmentFirmAgent(Agent):
    """
    Represents a Philippine investment firm (SME or large corporation)
    that makes capex and hiring decisions based on borrowing costs.

    Heterogeneity is introduced via randomized capital base and expected
    ROI at initialization -- giving each firm a different sensitivity
    to interest rate changes.

    Attributes
    ----------
    capital : float
        Firm's asset base in PHP.
    expected_roi : float
        Firm's internal expected return on investment (fixed, decimal).
    hurdle_rate : float
        Minimum acceptable ROI = lending_rate + hurdle_spread.
    is_investing : bool
        True if expected_roi clears the hurdle rate this step.
    capex : float
        Capital expenditure this step (PHP).
    workers : float
        Current headcount (fractional for modeling convenience).
    new_hires : float
        Net hiring this step (positive = hiring, negative = layoffs).
    debt : float
        Outstanding firm debt (PHP).
    debt_repayment : float
        Debt repaid this step (PHP).
    new_borrowing : float
        New debt taken to fund capex (PHP).
    depreciation : float
        Capital depreciation this step (PHP).
    """

    def __init__(self, model: "BSPModel") -> None:
        """
        Initialize an InvestmentFirmAgent with randomized capital and ROI.

        Parameters
        ----------
        model : BSPModel
            The BSP ABM model instance.
        """
        super().__init__(model)

        # Heterogeneous capital base
        self.capital: float = max(
            FIRM_CAPITAL_MIN,
            self.model.random.gauss(FIRM_CAPITAL_MEAN, FIRM_CAPITAL_STD)
        )

        # Heterogeneous ROI expectation -- fixed per firm (their business model)
        self.expected_roi: float = max(
            0.01,
            self.model.random.gauss(FIRM_ROI_MEAN, FIRM_ROI_STD)
        )

        # Debt must be set before computing capex (ceiling check)
        self.debt: float = self.capital * FIRM_DEBT_CAPITAL_RATIO_MAX * 0.4

        # Initialize at current lending rate
        lending_rate: float = model.avg_lending_rate
        self.hurdle_rate: float = self._compute_hurdle_rate(lending_rate)
        self.is_investing: bool = self._check_investment_decision()
        self.capex: float = self._compute_capex(lending_rate)
        self.new_borrowing: float = self._compute_new_borrowing()
        self.debt_repayment: float = self._compute_debt_repayment()

        self.depreciation: float = round(self.capital * FIRM_DEPRECIATION_RATE, 2)

        # Workforce
        self.workers: float = float(FIRM_INITIAL_WORKERS)
        self.new_hires: float = self._compute_hiring()

    # -------------------------------------------------------------------------
    # Core investment mechanics
    # -------------------------------------------------------------------------

    def _compute_hurdle_rate(self, lending_rate: float) -> float:
        """
        Compute the firm's hurdle rate (minimum ROI required to invest).

        Formula:
            hurdle_rate = lending_rate + hurdle_spread

        The spread reflects that firms need a return premium above their
        cost of debt to justify the risk of investing.

        Parameters
        ----------
        lending_rate : float
            Current average bank lending rate (decimal).

        Returns
        -------
        float
            Hurdle rate (decimal).
        """
        return round(lending_rate + FIRM_HURDLE_RATE_SPREAD, 6)

    def _check_investment_decision(self) -> bool:
        """
        Determine if the firm proceeds with investment this step.

        Rule: Invest if expected_ROI > hurdle_rate.

        Returns
        -------
        bool
            True if the firm invests, False if it holds back.
        """
        return self.expected_roi > self.hurdle_rate

    def _compute_capex(self, lending_rate: float) -> float:
        """
        Compute capital expenditure this step.

        If investing:
            capex = capital * base_rate * max(0, 1 - sensitivity * excess_cost)
            where excess_cost = max(0, lending_rate - neutral_rate)

        If not investing:
            capex = capital * min_rate  (maintenance capex only)

        The sensitivity parameter means each 1% the lending rate is above
        neutral reduces capex by FIRM_CAPEX_SENSITIVITY percent.
        Also enforces the debt ceiling -- no capex that would breach it.

        Parameters
        ----------
        lending_rate : float
            Current average bank lending rate (decimal).

        Returns
        -------
        float
            Capex amount this step (PHP).
        """
        if not self.is_investing:
            return round(self.capital * FIRM_CAPEX_MIN_RATE, 2)

        excess_cost: float = max(0.0, lending_rate - RRP_RATE_NEUTRAL)
        scale: float = max(0.0, 1.0 - FIRM_CAPEX_SENSITIVITY * excess_cost)
        desired: float = self.capital * FIRM_CAPEX_BASE_RATE * scale

        # Enforce debt ceiling
        debt_headroom: float = (
            self.capital * FIRM_DEBT_CAPITAL_RATIO_MAX
        ) - self.debt
        return round(max(self.capital * FIRM_CAPEX_MIN_RATE,
                         min(desired, max(0.0, debt_headroom))), 2)

    def _compute_new_borrowing(self) -> float:
        """
        New borrowing equals capex -- all investment is debt-financed.

        Returns
        -------
        float
            New debt taken this step (PHP).
        """
        return self.capex

    def _compute_debt_repayment(self) -> float:
        """
        Fixed fraction of debt stock is repaid each step.

        Returns
        -------
        float
            Debt repaid this step (PHP).
        """
        return round(self.debt * FIRM_DEBT_REPAY_RATE, 2)

    def _compute_hiring(self) -> float:
        """
        Compute net hiring as a function of capex.

        Formula:
            new_hires = (capex - last_capex_equivalent) * hiring_ratio

        Simplified here as: if capex is at base rate, hiring is 0.
        Above base = hiring, below base = layoffs.

        Returns
        -------
        float
            Net new hires this step (can be negative = layoffs).
        """
        base_capex: float = self.capital * FIRM_CAPEX_BASE_RATE
        capex_delta: float = self.capex - base_capex
        return round(capex_delta * FIRM_HIRING_CAPEX_RATIO, 4)

    # -------------------------------------------------------------------------
    # Mesa step
    # -------------------------------------------------------------------------

    def step(self) -> None:
        """
        Execute one time step of firm behavior.

        Order of operations (economically motivated):
          1. Read lending rate from model
          2. Recompute hurdle rate
          3. Make investment decision (go/no-go)
          4. Compute capex based on decision and rate level
          5. Compute debt repayment (existing obligations first)
          6. Update debt stock
          7. Update capital (capex adds to asset base)
          8. Compute hiring from capex change
          9. Update workforce
        """
        lending_rate: float = self.model.avg_lending_rate

        # 1-3: Decision
        self.hurdle_rate = self._compute_hurdle_rate(lending_rate)
        self.is_investing = self._check_investment_decision()

        # 4: Capex
        self.capex = self._compute_capex(lending_rate)
        self.new_borrowing = self._compute_new_borrowing()

        # 5-6: Debt management
        self.debt_repayment = self._compute_debt_repayment()
        self.debt = round(
            max(0.0, self.debt - self.debt_repayment + self.new_borrowing), 2
        )

        # 7: Capital grows with investment, shrinks with depreciation
        self.depreciation = round(self.capital * FIRM_DEPRECIATION_RATE, 2)
        self.capital = round(max(FIRM_CAPITAL_MIN, self.capital + self.capex - self.depreciation), 2)

        # 8-9: Workforce
        self.new_hires = self._compute_hiring()
        self.workers = round(max(0.0, self.workers + self.new_hires), 4)

    # -------------------------------------------------------------------------
    # Reporting helpers
    # -------------------------------------------------------------------------

    def get_state(self) -> dict:
        """
        Return a snapshot of the firm's current state.

        Returns
        -------
        dict
            Key metrics for this firm agent.
        """
        return {
            "agent_id": self.unique_id,
            "capital": self.capital,
            "expected_roi": self.expected_roi,
            "hurdle_rate": self.hurdle_rate,
            "is_investing": self.is_investing,
            "capex": self.capex,
            "workers": self.workers,
            "new_hires": self.new_hires,
            "debt": self.debt,
            "depreciation": self.depreciation,
            "debt_to_capital": self.debt / self.capital if self.capital > 0 else 0.0,
        }

    def __repr__(self) -> str:
        return (
            f"InvestmentFirmAgent(id={self.unique_id}, "
            f"capital=PHP {self.capital:,.0f}, "
            f"roi={self.expected_roi:.1%}, "
            f"hurdle={self.hurdle_rate:.1%}, "
            f"investing={self.is_investing}, "
            f"capex=PHP {self.capex:,.0f}, "
            f"workers={self.workers:.1f})"
        )