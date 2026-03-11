"""
BSPModel: Core ABM simulating BSP monetary policy transmission.
The model includes three agent types:
- BankAgent: Adjusts lending/deposit rates and credit supply based on RRP.
- HouseholdAgent: Adjusts consumption/savings/borrowing based on bank rates.
- InvestmentFirmAgent: Adjusts capex/hiring based on bank rates.
"""

from mesa import Model
from mesa.datacollection import DataCollector

from agents.bank import BankAgent
from agents.household import HouseholdAgent
from agents.investment_firm import InvestmentFirmAgent
from config.parameters import (
    RRP_RATE_DEFAULT,
    NUM_BANKS,
    NUM_HOUSEHOLDS,
    NUM_FIRMS,
    RANDOM_SEED,
)


class BSPModel(Model):
    """
    Top-level ABM simulating BSP monetary policy transmission.

    Activation order each step:
      1. Banks     -- reprice rates based on RRP
      2. Households -- adjust savings/borrowing/consumption based on bank rates
      3. Firms     -- adjust capex/hiring based on bank rates

    GDP Proxy = Total Household Consumption + Total Firm Capex.
    This is a simplified demand-side GDP analog.

    Parameters
    ----------
    rrp_rate : float
        Initial BSP RRP rate (decimal).
    num_banks : int
        Number of bank agents.
    num_households : int
        Number of household agents.
    num_firms : int
        Number of investment firm agents.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        rrp_rate: float = RRP_RATE_DEFAULT,
        num_banks: int = NUM_BANKS,
        num_households: int = NUM_HOUSEHOLDS,
        num_firms: int = NUM_FIRMS,
        seed: int = RANDOM_SEED,
    ) -> None:
        super().__init__()
        self.random.seed(seed)

        self.rrp_rate: float = rrp_rate
        self.step_count: int = 0

        # agent creation order matters:
        # banks first (rate-setters), then households and firms (rate-takers)
        for _ in range(num_banks):
            BankAgent(self)
        for _ in range(num_households):
            HouseholdAgent(self)
        for _ in range(num_firms):
            InvestmentFirmAgent(self)

        self.datacollector = DataCollector(
            model_reporters={
                # policy
                "RRP_Rate":             lambda m: m.rrp_rate,
                # bank aggregates
                "Avg_Lending_Rate":     lambda m: m.avg_lending_rate,
                "Avg_Deposit_Rate":     lambda m: m.avg_deposit_rate,
                "Avg_Reserve_Ratio":    lambda m: m.avg_reserve_ratio,
                "Total_Credit_Supply":  lambda m: m.total_credit_supply,
                # household aggregates
                "Avg_Savings_Rate":     lambda m: m.avg_savings_rate,
                "Total_Consumption":    lambda m: m.total_consumption,
                "Total_HH_Debt":        lambda m: m.total_household_debt,
                "Total_New_Borrowing":  lambda m: m.total_new_borrowing,
                # firm aggregates
                "Total_Capex":          lambda m: m.total_capex,
                "Total_Workers":        lambda m: m.total_workers,
                "Pct_Firms_Investing":  lambda m: m.pct_firms_investing,
                "Total_Firm_Debt":      lambda m: m.total_firm_debt,
                # macro proxy
                "GDP_Proxy":            lambda m: m.gdp_proxy,
            },
            agent_reporters={
                "Type":         lambda a: type(a).__name__,
                "Lending_Rate": lambda a: getattr(a, "lending_rate", None),
                "Consumption":  lambda a: getattr(a, "consumption", None),
                "Capex":        lambda a: getattr(a, "capex", None),
                "Is_Investing": lambda a: getattr(a, "is_investing", None),
                "Workers":      lambda a: getattr(a, "workers", None),
                "Debt":         lambda a: getattr(a, "debt", None),
            },
        )

    # -------------------------------------------------------------------------
    # Model-level properties -- Bank aggregates
    # -------------------------------------------------------------------------

    @property
    def avg_lending_rate(self) -> float:
        """Average lending rate across all bank agents."""
        banks = list(self.agents_by_type.get(BankAgent, []))
        return sum(b.lending_rate for b in banks) / len(banks) if banks else 0.0

    @property
    def avg_deposit_rate(self) -> float:
        """Average deposit rate across all bank agents."""
        banks = list(self.agents_by_type.get(BankAgent, []))
        return sum(b.deposit_rate for b in banks) / len(banks) if banks else 0.0

    @property
    def avg_reserve_ratio(self) -> float:
        """Average reserve ratio across all bank agents."""
        banks = list(self.agents_by_type.get(BankAgent, []))
        return sum(b.reserve_ratio for b in banks) / len(banks) if banks else 0.0

    @property
    def total_credit_supply(self) -> float:
        """Total credit supply across all bank agents (PHP)."""
        return sum(b.credit_supply for b in self.agents_by_type.get(BankAgent, []))

    # -------------------------------------------------------------------------
    # Model-level properties -- Household aggregates
    # -------------------------------------------------------------------------

    @property
    def avg_savings_rate(self) -> float:
        """Average savings rate across all household agents."""
        hh = list(self.agents_by_type.get(HouseholdAgent, []))
        return sum(h.savings_rate for h in hh) / len(hh) if hh else 0.0

    @property
    def total_consumption(self) -> float:
        """Total consumption across all household agents (PHP)."""
        return sum(h.consumption for h in self.agents_by_type.get(HouseholdAgent, []))

    @property
    def total_household_debt(self) -> float:
        """Total outstanding household debt (PHP)."""
        return sum(h.debt for h in self.agents_by_type.get(HouseholdAgent, []))

    @property
    def total_new_borrowing(self) -> float:
        """Total new borrowing this step across all households (PHP)."""
        return sum(h.new_borrowing for h in self.agents_by_type.get(HouseholdAgent, []))

    # -------------------------------------------------------------------------
    # Model-level properties -- Firm aggregates
    # -------------------------------------------------------------------------

    @property
    def total_capex(self) -> float:
        """Total capital expenditure across all firms this step (PHP)."""
        return sum(f.capex for f in self.agents_by_type.get(InvestmentFirmAgent, []))

    @property
    def total_workers(self) -> float:
        """Total workforce across all firms."""
        return sum(f.workers for f in self.agents_by_type.get(InvestmentFirmAgent, []))

    @property
    def pct_firms_investing(self) -> float:
        """Fraction of firms whose ROI clears the hurdle rate (0-1)."""
        firms = list(self.agents_by_type.get(InvestmentFirmAgent, []))
        if not firms:
            return 0.0
        return sum(1 for f in firms if f.is_investing) / len(firms)

    @property
    def total_firm_debt(self) -> float:
        """Total outstanding firm debt (PHP)."""
        return sum(f.debt for f in self.agents_by_type.get(InvestmentFirmAgent, []))

    # -------------------------------------------------------------------------
    # Model-level properties -- Macro proxy
    # -------------------------------------------------------------------------

    @property
    def gdp_proxy(self) -> float:
        """
        Simplified GDP proxy = Total Consumption + Total Capex.

        This captures the two largest demand-side components:
        - C (household consumption)
        - I (business investment / capex)

        Government spending and net exports are excluded at this stage.
        """
        return self.total_consumption + self.total_capex

    # -------------------------------------------------------------------------
    # Policy interface
    # -------------------------------------------------------------------------

    def set_rrp_rate(self, new_rate: float) -> None:
        """
        Update the BSP RRP rate (simulates a policy decision).
        Call before model.step() to apply at the next step.

        Parameters
        ----------
        new_rate : float
            New RRP rate (decimal, e.g., 0.065 for 6.5%).
        """
        self.rrp_rate = round(new_rate, 6)

    # -------------------------------------------------------------------------
    # Mesa step -- ordered activation
    # -------------------------------------------------------------------------

    def step(self) -> None:
        """
        Advance the model by one time step.

        Activation order is economically motivated:
          1. Collect data snapshot (before any agent acts)
          2. Banks reprice based on new RRP
          3. Households respond to updated bank rates
          4. Firms respond to updated bank rates
        """
        self.datacollector.collect(self)

        for agent in self.agents_by_type.get(BankAgent, []):
            agent.step()

        for agent in self.agents_by_type.get(HouseholdAgent, []):
            agent.step()

        for agent in self.agents_by_type.get(InvestmentFirmAgent, []):
            agent.step()

        self.step_count += 1