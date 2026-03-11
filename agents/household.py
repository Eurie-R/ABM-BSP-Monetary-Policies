"""
Household Agent for the BSP Monetary Policy ABM.

Economic Role:
--------------
Households are the SECOND transmission node of BSP monetary policy.
They observe bank lending and deposit rates and respond by adjusting:

  1. Savings rate    -- rises when deposit rate rises (substitution effect)
  2. New borrowing   -- falls when lending rate rises (credit demand channel)
  3. Debt stock      -- accumulates from borrowing, decays via repayment
  4. Consumption     -- residual after savings and debt service

The aggregate effect: a rate hike compresses household consumption,
which is the primary driver of domestic demand in the Philippines
(household consumption ~70% of GDP).

Transmission chain:
    BSP RRP up
      -> Bank lending rate up, deposit rate up
        -> Households save more, borrow less
          -> Consumption falls
            -> Aggregate demand contracts (deflationary pressure)

Assumptions (flag for future revision):
-----------------------------------------
- Income is fixed each step (no unemployment channel yet)
- All households borrow from a single representative bank rate
- No wealth effects (assets not modeled yet)
- Debt default not modeled at this stage
- Remittance income not included (relevant for PH -- future enhancement)
"""

from mesa import Agent
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from model.bsp_model import BSPModel

from config.parameters import (
    HOUSEHOLD_INCOME_MEAN,
    HOUSEHOLD_INCOME_STD,
    HOUSEHOLD_SAVINGS_RATE_BASE,
    HOUSEHOLD_SAVINGS_RATE_MIN,
    HOUSEHOLD_SAVINGS_RATE_MAX,
    HOUSEHOLD_SAVINGS_SENSITIVITY,
    HOUSEHOLD_DEBT_INCOME_RATIO_MAX,
    HOUSEHOLD_BORROW_SENSITIVITY,
    HOUSEHOLD_DEBT_REPAY_RATE,
    HOUSEHOLD_BORROW_BASE_RATE,
    HOUSEHOLD_CONSUMPTION_FLOOR,
    RRP_RATE_NEUTRAL,
)


class HouseholdAgent(Agent):
    """
    Represents a Philippine household that responds to interest rate signals
    from the banking sector.

    Heterogeneity is introduced via randomized income at initialization,
    giving each household a different baseline for savings and borrowing
    decisions.

    Attributes
    ----------
    income : float
        Monthly income in PHP (fixed, drawn from normal distribution).
    savings_rate : float
        Fraction of income saved each step (adjusts with deposit rate).
    savings : float
        Accumulated savings stock (PHP).
    debt : float
        Outstanding debt stock (PHP).
    new_borrowing : float
        New loans taken this step (PHP).
    debt_repayment : float
        Debt repaid this step (PHP).
    consumption : float
        Consumption spending this step (PHP).
    """

    def __init__(self, model: "BSPModel") -> None:
        """
        Initialize a HouseholdAgent with randomized income and
        baseline financial state at the model's starting RRP rate.

        Parameters
        ----------
        model : BSPModel
            The BSP ABM model instance.
        """
        super().__init__(model)

        # Heterogeneous income -- drawn once at initialization
        self.income: float = max(
            5_000.0,  # floor at PHP 5,000 -- no negative incomes
            self.model.random.gauss(HOUSEHOLD_INCOME_MEAN, HOUSEHOLD_INCOME_STD)
        )

        # Initialize all behavioral variables at current rates
        lending_rate: float = model.avg_lending_rate
        deposit_rate: float = model.avg_deposit_rate

        # debt must be set BEFORE _compute_new_borrowing (ceiling check uses it)
        self.debt: float = self.income * HOUSEHOLD_DEBT_INCOME_RATIO_MAX * 0.5

        self.savings_rate: float = self._compute_savings_rate(deposit_rate)
        self.new_borrowing: float = self._compute_new_borrowing(lending_rate)
        self.debt_repayment: float = self._compute_debt_repayment()
        self.savings: float = self.income * self.savings_rate * 6  # ~6 months buffer
        self.consumption: float = self._compute_consumption()

    # -------------------------------------------------------------------------
    # Core behavioral mechanics
    # -------------------------------------------------------------------------

    def _compute_savings_rate(self, deposit_rate: float) -> float:
        """
        Compute the household's savings rate as a function of the deposit rate.

        Formula:
            savings_rate = base_rate + sensitivity * deposit_rate

        Higher deposit rates reward saving, pulling income away from consumption.
        Clamped between min and max to reflect real-world behavioral bounds.

        Parameters
        ----------
        deposit_rate : float
            Current average bank deposit rate (decimal).

        Returns
        -------
        float
            Savings rate (decimal, e.g., 0.15 = 15% of income saved).
        """
        raw: float = HOUSEHOLD_SAVINGS_RATE_BASE + HOUSEHOLD_SAVINGS_SENSITIVITY * deposit_rate
        return round(max(HOUSEHOLD_SAVINGS_RATE_MIN, min(HOUSEHOLD_SAVINGS_RATE_MAX, raw)), 6)

    def _compute_new_borrowing(self, lending_rate: float) -> float:
        """
        Compute new borrowing this step as a function of the lending rate.

        Formula:
            excess_rate  = max(0, lending_rate - neutral_rate)
            reduction    = borrow_sensitivity * excess_rate
            new_borrowing = income * borrow_base_rate * max(0, 1 - reduction)

        At the neutral rate, households borrow their baseline amount.
        For each 1% the lending rate exceeds neutral, borrowing is cut by
        HOUSEHOLD_BORROW_SENSITIVITY (default 80%).

        Also enforces the debt-to-income ceiling -- no new borrowing if
        already at max leverage.

        Parameters
        ----------
        lending_rate : float
            Current average bank lending rate (decimal).

        Returns
        -------
        float
            New borrowing amount this step (PHP), floored at 0.
        """
        # Debt ceiling check -- no new borrowing if already at limit
        if self.debt >= self.income * HOUSEHOLD_DEBT_INCOME_RATIO_MAX:
            return 0.0

        excess_rate: float = max(0.0, lending_rate - RRP_RATE_NEUTRAL)
        reduction: float = HOUSEHOLD_BORROW_SENSITIVITY * excess_rate
        desired: float = self.income * HOUSEHOLD_BORROW_BASE_RATE * max(0.0, 1.0 - reduction)

        # Don't borrow past the ceiling
        headroom: float = (self.income * HOUSEHOLD_DEBT_INCOME_RATIO_MAX) - self.debt
        return round(max(0.0, min(desired, headroom)), 2)

    def _compute_debt_repayment(self) -> float:
        """
        Compute debt repayment this step.

        Formula:
            repayment = debt_stock * repay_rate

        A fixed fraction of the debt stock is repaid each step,
        approximating an amortizing loan structure.

        Returns
        -------
        float
            Repayment amount (PHP).
        """
        return round(self.debt * HOUSEHOLD_DEBT_REPAY_RATE, 2)

    def _compute_consumption(self) -> float:
        """
        Compute consumption as the residual of income after saving and debt service.

        Formula:
            consumption = income - savings_amount - debt_repayment + new_borrowing

        Enforces a consumption floor of HOUSEHOLD_CONSUMPTION_FLOOR * income
        to reflect subsistence spending (food, shelter, utilities).

        Returns
        -------
        float
            Consumption spending this step (PHP).
        """
        savings_amount: float = self.income * self.savings_rate
        raw: float = self.income - savings_amount - self.debt_repayment + self.new_borrowing
        floor: float = self.income * HOUSEHOLD_CONSUMPTION_FLOOR
        return round(max(floor, raw), 2)

    # -------------------------------------------------------------------------
    # Mesa step
    # -------------------------------------------------------------------------

    def step(self) -> None:
        """
        Execute one time step of household behavior.

        Each step, the household reads aggregate bank rates from the model
        and updates its financial decisions in this order:

          1. Update savings rate (responds to deposit rate)
          2. Compute debt repayment (fixed fraction of debt stock)
          3. Compute new borrowing (responds to lending rate)
          4. Update debt stock (+ new borrowing - repayment)
          5. Update savings stock (+ this period's savings)
          6. Compute consumption (residual)

        The order matters: debt repayment is determined BEFORE new
        borrowing, reflecting that existing obligations are met first.
        """
        lending_rate: float = self.model.avg_lending_rate
        deposit_rate: float = self.model.avg_deposit_rate

        # 1. Savings decision
        self.savings_rate = self._compute_savings_rate(deposit_rate)

        # 2. Service existing debt first
        self.debt_repayment = self._compute_debt_repayment()

        # 3. New borrowing decision (after checking debt ceiling)
        self.new_borrowing = self._compute_new_borrowing(lending_rate)

        # 4. Update debt stock
        self.debt = round(
            max(0.0, self.debt - self.debt_repayment + self.new_borrowing), 2
        )

        # 5. Accumulate savings
        self.savings = round(self.savings + self.income * self.savings_rate, 2)

        # 6. Consumption is the residual
        self.consumption = self._compute_consumption()

    # -------------------------------------------------------------------------
    # Reporting helpers
    # -------------------------------------------------------------------------

    def get_state(self) -> dict:
        """
        Return a snapshot of the household's current financial state.

        Returns
        -------
        dict
            Key metrics for this household agent.
        """
        return {
            "agent_id": self.unique_id,
            "income": self.income,
            "savings_rate": self.savings_rate,
            "savings": self.savings,
            "debt": self.debt,
            "new_borrowing": self.new_borrowing,
            "debt_repayment": self.debt_repayment,
            "consumption": self.consumption,
            "debt_to_income": self.debt / self.income if self.income > 0 else 0.0,
        }

    def __repr__(self) -> str:
        return (
            f"HouseholdAgent(id={self.unique_id}, "
            f"income=PHP {self.income:,.0f}, "
            f"savings={self.savings_rate:.1%}, "
            f"debt=PHP {self.debt:,.0f}, "
            f"consumption=PHP {self.consumption:,.0f})"
        )