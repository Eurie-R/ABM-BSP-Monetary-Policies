"""
Bank Agent for the BSP Monetary Policy ABM.

Economic Role:
--------------
Banks are the PRIMARY transmission node of BSP monetary policy.
When the BSP changes the RRP rate:
  1. Lending rate reprices (RRP + spread)           → credit becomes cheaper/costlier
  2. Deposit rate adjusts (RRP - deposit spread)    → savings incentive changes
  3. Reserve ratio adjusts (parking at BSP more     → loanable funds contract
     attractive when RRP is high)
  4. Credit supply updates (deposits × (1 - reserve_ratio) × multiplier)

Assumptions (flag for future revision):
----------------------------------------
- All banks are homogeneous in type (universal/commercial).
- Spreads are deterministic functions of RRP, not market-determined yet.
- No interbank lending market modeled at this stage.
- Deposits are exogenous (not yet coupled to household savings decisions).
"""

from mesa import Agent
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from model.bsp_model import BSPModel
from config.parameters import (
    BANK_CREDIT_SPREAD,
    BANK_SPREAD_MIN,
    BANK_SPREAD_COMPRESSION,
    BANK_DEPOSIT_SPREAD,
    BANK_RESERVE_RATIO_BASE,
    BANK_RESERVE_SENSITIVITY,
    BANK_CREDIT_MULTIPLIER,
    BANK_INITIAL_DEPOSITS,
    BANK_INITIAL_CAPITAL,
    BANK_LIQUIDITY_BUFFER,
    RRP_RATE_NEUTRAL,
)


class BankAgent(Agent):
    """
    Represents a commercial/universal bank in the Philippine financial system.

    The bank observes the BSP RRP rate from the model and adjusts its
    lending rate, deposit rate, reserve holdings, and credit supply each step.

    Attributes
    ----------
    unique_id : int
        Mesa-required agent identifier.
    model : Model
        Reference to the BSP ABM model (used to read current RRP rate).
    lending_rate : float
        Interest rate charged on new loans (decimal, e.g., 0.10 = 10%).
    deposit_rate : float
        Interest rate offered on deposits (decimal).
    reserve_ratio : float
        Fraction of deposits held as reserves (statutory + voluntary).
    credit_supply : float
        Total loanable funds available to households and firms (PHP).
    deposits : float
        Total deposit liabilities (PHP).
    capital : float
        Bank equity buffer (PHP).
    liquidity : float
        Liquid assets above reserve requirement (PHP).
    """

    def __init__(self, model: "BSPModel") -> None:
        """
        Initialize a BankAgent with baseline values derived from the
        initial RRP rate in the model.

        Parameters
        ----------
        model : BSPModel
            The BSP ABM model instance. unique_id is auto-assigned by Mesa 3.
        """
        super().__init__(model)

        # Balance sheet
        self.deposits: float = BANK_INITIAL_DEPOSITS
        self.capital: float = BANK_INITIAL_CAPITAL

        # Rate-setting — initialize at whatever the model's starting RRP is
        self.lending_rate: float = self._compute_lending_rate(model.rrp_rate)
        self.deposit_rate: float = self._compute_deposit_rate(model.rrp_rate)

        # Reserve behavior
        self.reserve_ratio: float = self._compute_reserve_ratio(model.rrp_rate)

        # Credit supply derived from the above
        self.credit_supply: float = self._compute_credit_supply()

        # Liquidity: deposits not tied up in reserves, above the safety buffer
        self.liquidity: float = self._compute_liquidity()

    # -------------------------------------------------------------------------
    # Core transmission mechanics
    # -------------------------------------------------------------------------

    def _compute_lending_rate(self, rrp_rate: float) -> float:
        """
        Compute the bank's lending rate as a function of the RRP rate.

        Formula:
            rate_above_neutral = max(0, RRP - neutral_rate)
            spread = max(0, base_spread - compression × rate_above_neutral)
            lending_rate = RRP + spread

        Calibration (Session 5):
            The spread COMPRESSES as RRP rises above neutral. This reflects
            real BSP data: spread was 3.57% at RRP=2% (May 2022) but only
            1.30% at RRP=6.5% (Oct 2023). Banks absorb part of the hike
            rather than passing it fully to borrowers — doing so would kill
            credit demand. This produces a pass-through ratio of ~42%,
            consistent with the observed 40.5% (172bps / 425bps).

        Parameters
        ----------
        rrp_rate : float
            Current BSP RRP rate (decimal).

        Returns
        -------
        float
            Bank lending rate (decimal).
        """
        rate_above_neutral: float = max(0.0, rrp_rate - RRP_RATE_NEUTRAL)
        spread: float = max(BANK_SPREAD_MIN, BANK_CREDIT_SPREAD - BANK_SPREAD_COMPRESSION * rate_above_neutral)
        return round(rrp_rate + spread, 6)

    def _compute_deposit_rate(self, rrp_rate: float) -> float:
        """
        Compute the deposit rate offered to savers.

        Formula:
            deposit_rate = RRP - deposit_spread

        Banks pass through RRP increases to depositors, but with a lag
        and a persistent margin (net interest margin).

        Parameters
        ----------
        rrp_rate : float
            Current BSP RRP rate (decimal).

        Returns
        -------
        float
            Deposit interest rate (decimal), floored at 0.
        """
        return round(max(0.0, rrp_rate - BANK_DEPOSIT_SPREAD), 6)

    def _compute_reserve_ratio(self, rrp_rate: float) -> float:
        """
        Compute the effective reserve ratio (statutory + voluntary).

        Formula:
            reserve_ratio = base_RRR + reserve_sensitivity × max(0, RRP - neutral)

        When RRP is above neutral, parking reserves at BSP is more
        attractive, so banks voluntarily hold excess reserves.

        Parameters
        ----------
        rrp_rate : float
            Current BSP RRP rate (decimal).

        Returns
        -------
        float
            Reserve ratio (decimal, e.g., 0.12 = 12%).
        """
        excess: float = max(0.0, rrp_rate - RRP_RATE_NEUTRAL)
        return round(BANK_RESERVE_RATIO_BASE + BANK_RESERVE_SENSITIVITY * excess, 6)

    def _compute_credit_supply(self) -> float:
        """
        Compute total credit supply from loanable funds.

        Formula:
            loanable_funds = deposits × (1 - reserve_ratio)
            credit_supply  = loanable_funds × credit_multiplier

        The multiplier reflects the bank's ability to extend credit
        beyond its immediate deposit base (fractional reserve banking).

        Returns
        -------
        float
            Credit supply available to the economy (PHP).
        """
        loanable_funds: float = self.deposits * (1.0 - self.reserve_ratio)
        return round(loanable_funds * BANK_CREDIT_MULTIPLIER, 2)

    def _compute_liquidity(self) -> float:
        """
        Compute liquidity as deposits not absorbed by reserves,
        minus the mandatory safety buffer.

        Returns
        -------
        float
            Liquidity position (PHP). Negative signals stress.
        """
        free_reserves: float = self.deposits * (1.0 - self.reserve_ratio)
        buffer: float = self.deposits * BANK_LIQUIDITY_BUFFER
        return round(free_reserves - buffer, 2)

    # -------------------------------------------------------------------------
    # Mesa step method
    # -------------------------------------------------------------------------

    def step(self) -> None:
        """
        Execute one time step of bank behavior.

        Each step, the bank reads the current RRP rate from the model
        and updates all its derived variables accordingly.

        The order of updates matters:
          1. Rates reprice first (fastest to adjust in real markets)
          2. Reserve ratio adjusts (balance sheet decision)
          3. Credit supply recalculates (depends on new reserve ratio)
          4. Liquidity updates last (residual)
        """
        rrp: float = self.model.rrp_rate

        self.lending_rate = self._compute_lending_rate(rrp)
        self.deposit_rate = self._compute_deposit_rate(rrp)
        self.reserve_ratio = self._compute_reserve_ratio(rrp)
        self.credit_supply = self._compute_credit_supply()
        self.liquidity = self._compute_liquidity()

    # -------------------------------------------------------------------------
    # Reporting helpers
    # -------------------------------------------------------------------------

    def get_state(self) -> dict:
        """
        Return a snapshot of the bank's current state as a dictionary.
        Useful for data collection and debugging.

        Returns
        -------
        dict
            Key metrics for this bank agent.
        """
        return {
            "agent_id": self.unique_id,
            "lending_rate": self.lending_rate,
            "deposit_rate": self.deposit_rate,
            "reserve_ratio": self.reserve_ratio,
            "credit_supply": self.credit_supply,
            "deposits": self.deposits,
            "liquidity": self.liquidity,
        }

    def __repr__(self) -> str:
        return (
            f"BankAgent(id={self.unique_id}, "
            f"lending={self.lending_rate:.2%}, "
            f"deposit={self.deposit_rate:.2%}, "
            f"reserves={self.reserve_ratio:.2%}, "
            f"credit=PHP {self.credit_supply:,.0f})"
        )