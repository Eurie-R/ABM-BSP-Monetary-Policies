"""
Central configuration for the BSP Monetary Policy ABM.

All numeric constants live here — no magic numbers in agent or model code.
Every parameter includes its source and calibration rationale.

Calibration basis:
  - BSP published statistics & circulars (2022-2024)
  - PSA Family Income & Expenditure Survey (FIES) 2023
  - PSA National Accounts 2023
  - BSP Financial Stability Report 2023
"""

# =============================================================================
# POLICY PARAMETERS
# =============================================================================
# Source: BSP Monetary Board decisions 2022-2023
# BSP began hiking in May 2022 from 2.0%, peaked at 6.5% in Oct 2023 (off-cycle).
# Total tightening: 450 basis points over ~18 months.
# Long-run neutral estimated at 4.0% (BSP research, pre-pandemic baseline).

RRP_RATE_DEFAULT: float = 0.065       # 6.5%  — BSP peak rate, Oct 2023 off-cycle hike
RRP_RATE_NEUTRAL: float = 0.04        # 4.0%  — estimated long-run neutral rate
RRP_RATE_MIN: float = 0.02            # 2.0%  — pandemic-era floor (May 2022 starting point)
RRP_RATE_MAX: float = 0.10            # 10.0% — upper simulation bound

# =============================================================================
# BANK AGENT PARAMETERS
# =============================================================================
# Source: BSP bank lending and deposit rate statistics (Table 19 / IFS data)
# At peak RRP of 6.5% (2023), average lending rate was ~7.55-7.80%.
# Implied spread over RRP: ~1.1-1.3%. Calibrated to 1.2%.
# Pass-through ratio: lending rate rose ~172bps for 425bps RRP hike = ~40%.
# Source: Statista / BSP average lending rates 2022-2023.

# --- Lending Rate ---
BANK_CREDIT_SPREAD: float = 0.035
# 3.5% — base spread at low/neutral rates (May 2022: 5.57% lending - 2.0% RRP = 3.57%).
# Spread COMPRESSES above neutral — see BANK_SPREAD_COMPRESSION below.
# Source: BSP lending rate statistics May 2022.

BANK_SPREAD_MIN: float = 0.005         # 0.5% floor -- minimum margin to cover bank operating costs
BANK_SPREAD_COMPRESSION: float = 1.09
# Spread COMPRESSES as RRP rises above neutral (inverted from initial model).
# Real data: spread was 3.57% at RRP=2% but only 1.30% at RRP=6.5%.
# Banks absorb part of rate hikes rather than passing fully to borrowers.
# Formula: spread = max(0, base_spread - compression * max(0, RRP - neutral))
# Calibrated so: lending(2%)=5.50%≈BSP 5.57%, lending(6.25%)=7.30%≈BSP 7.29%
# Resulting pass-through: (7.30-5.50)/(6.25-2.0) = 42.3% ≈ actual 40.5%.

# --- Deposit Rate ---
BANK_DEPOSIT_SPREAD: float = 0.015
# 1.5% below RRP — banks maintain net interest margin on deposits.
# At 6.5% RRP, deposit rate was ~5.0%, consistent with BSP Table 19.

# --- Reserve Ratio ---
# Source: BSP Circular No. 1169 (May 2023) — RRR cut for U/KBs to 9.5%
# effective June 30, 2023. Previous RRR was 12% (pre-pandemic: 14%).
# Voluntary excess reserves rise when RRP is attractive (BSP parking).
BANK_RESERVE_RATIO_BASE: float = 0.095   # 9.5% — BSP Circular No. 1169, June 2023
BANK_RESERVE_SENSITIVITY: float = 0.3    # Voluntary reserves per 1% above neutral

# --- Credit Supply ---
# Conservative credit multiplier consistent with Philippine M3/deposit ratio.
BANK_CREDIT_MULTIPLIER: float = 1.8      # Conservative multiplier for U/KBs
BANK_INITIAL_DEPOSITS: float = 1_000_000 # PHP — base deposit stock per bank agent
BANK_INITIAL_CAPITAL: float = 150_000    # PHP — equity buffer

# --- Liquidity ---
BANK_LIQUIDITY_BUFFER: float = 0.05      # 5% minimum liquidity cushion above RRR

# --- Population ---
NUM_BANKS: int = 5

# =============================================================================
# HOUSEHOLD AGENT PARAMETERS
# =============================================================================
# Source: PSA Family Income & Expenditure Survey (FIES) 2023
# Average annual family income: PHP 353,230 -> PHP 29,436/month
# Average annual family expenditure: PHP 258,050 -> expenditure rate ~73%
# Regional spread: NCR PHP 42,793/mo vs BARMM PHP 17,240/mo (wide dispersion)
# Savings rate implied: ~27% of income is "saved" (not spent), but much of
# this goes to debt service, remittances, etc. — liquid savings rate ~10%.

NUM_HOUSEHOLDS: int = 50

# --- Income ---
HOUSEHOLD_INCOME_MEAN: float = 29_436.0
# PHP 29,436/month — PSA FIES 2023 (PHP 353,230 annual / 12)
# Previous value PHP 30,000 was a reasonable approximation; updated to exact figure.

HOUSEHOLD_INCOME_STD: float = 10_000.0
# Std dev PHP 10,000 — reflects regional income dispersion in FIES 2023
# (NCR-to-BARMM range spans ~PHP 25,000/month difference)

# --- Savings behavior ---
# Source: BSP Consumer Expectations Survey + FIES 2023
# Base savings rate ~10% reflects liquid savings (FIES expenditure rate ~73%,
# remaining ~27% covers debt, remittances, non-liquid savings).
HOUSEHOLD_SAVINGS_RATE_BASE: float = 0.10     # 10% base liquid savings rate
HOUSEHOLD_SAVINGS_RATE_MIN: float = 0.02      # 2% floor — subsistence buffer
HOUSEHOLD_SAVINGS_RATE_MAX: float = 0.40      # 40% ceiling — upper behavioral bound

HOUSEHOLD_SAVINGS_SENSITIVITY: float = 0.8
# Reduced from 1.5 to 0.8 — Philippine consumption proved resilient through
# the 2022-2023 hike cycle (HFCE grew 5.6% in 2023 despite 450bps of hikes).
# A sensitivity of 0.8 means each 1% deposit rate rise increases savings by 0.8pp.

# --- Borrowing behavior ---
HOUSEHOLD_DEBT_INCOME_RATIO_MAX: float = 0.30  # 30% DTI ceiling (BSP guideline)
HOUSEHOLD_BORROW_SENSITIVITY: float = 0.8      # 80% reduction in new borrowing per 1% excess
HOUSEHOLD_DEBT_REPAY_RATE: float = 0.05        # 5% of debt stock repaid each step
HOUSEHOLD_BORROW_BASE_RATE: float = 0.05       # 5% of income borrowed at neutral rate

# --- Consumption ---
# Source: PSA FIES 2023 — average expenditure rate 73% of income
# Floor raised from 50% to 70% to reflect actual Philippine spending behavior.
# Filipinos maintain high consumption even under financial stress (subsistence
# spending on food, transport, utilities dominates the budget).
HOUSEHOLD_CONSUMPTION_FLOOR: float = 0.70
# 70% floor — calibrated to PSA FIES 2023 expenditure rate of ~73%.
# Previous value 50% was too low; corrected in Session 5.

# =============================================================================
# INVESTMENT FIRM AGENT PARAMETERS
# =============================================================================
# Source: PSA National Accounts 2023, BSP Business Expectations Survey
# Gross Capital Formation (GCF): grew 5.4% full-year 2023 despite rate hikes.
# Q2 2023 (peak rates): GCF contracted only -0.04% — very mild contraction.
# Avg corporate ROI expectation: 12-18% (BSP Business Expectations Survey).

NUM_FIRMS: int = 20

# --- Firm size heterogeneity ---
FIRM_CAPITAL_MEAN: float = 5_000_000.0    # PHP — representative SME/corp mix
FIRM_CAPITAL_STD: float = 2_000_000.0     # PHP — reflects SME vs large corp spread
FIRM_CAPITAL_MIN: float = 500_000.0       # PHP — floor (small enterprise)

# --- Investment decision ---
FIRM_HURDLE_RATE_SPREAD: float = 0.05     # 5% buffer — calibrated so ~72% firms invest at peak rates
FIRM_ROI_MEAN: float = 0.14               # 14% — BSP Business Expectations Survey midpoint
FIRM_ROI_STD: float = 0.03               # 3% — heterogeneity across firms

# --- Capex behavior ---
FIRM_CAPEX_BASE_RATE: float = 0.08        # 8% of capital deployed as capex per step
FIRM_CAPEX_SENSITIVITY: float = 1.0
# Reduced from 2.0 to 1.0 — PSA data shows GCF contracted only -0.04% at
# peak rates. Philippine firms are less rate-sensitive than initially modeled,
# partly due to strong infrastructure pipeline and resilient domestic demand.

FIRM_CAPEX_MIN_RATE: float = 0.01         # 1% maintenance capex floor

# --- Capital depreciation (NEW — Session 5) ---
FIRM_DEPRECIATION_RATE: float = 0.02
# 2% per step capital depreciation — prevents unbounded capital accumulation.
# Approximates ~24% annual depreciation rate (consistent with PSA fixed asset
# consumption estimates for Philippine business sector).
# Without this, capital and capex grew indefinitely in Session 4 runs.

# --- Hiring behavior ---
FIRM_HIRING_CAPEX_RATIO: float = 0.004    # 0.4 workers per PHP 1,000 capex
FIRM_INITIAL_WORKERS: int = 10            # Starting headcount per firm

# --- Debt ---
FIRM_DEBT_CAPITAL_RATIO_MAX: float = 0.60  # 60% debt-to-capital ceiling
FIRM_DEBT_REPAY_RATE: float = 0.04         # 4% of debt stock repaid each step

# =============================================================================
# BENCHMARK SCENARIO — 2022-2023 BSP HIKE CYCLE
# =============================================================================
# Actual RRP rate path for validation (monthly steps):
# Source: BSP Monetary Board meeting decisions, May 2022 - Oct 2023
#
# Step 1  = May 2022  : 2.00% (baseline)
# Step 2  = Jun 2022  : 2.25% (+25bps)
# Step 3  = Jul 2022  : 2.50% (+25bps)  [estimated — no meeting, but interpolated]
# Step 4  = Aug 2022  : 3.25% (+75bps)
# Step 5  = Sep 2022  : 4.00% (+75bps)
# Step 6  = Oct 2022  : 4.00% (hold)
# Step 7  = Nov 2022  : 5.00% (+100bps)
# Step 8  = Dec 2022  : 5.00% (hold)
# Step 9  = Jan 2023  : 5.00% (hold)
# Step 10 = Feb 2023  : 5.50% (+50bps)
# Step 11 = Mar 2023  : 6.25% (+75bps)
# Step 12 = Apr 2023  : 6.25% (hold)
# Step 13 = May 2023  : 6.25% (hold)
# Step 14 = Jun 2023  : 6.25% (hold)
# Step 15 = Jul 2023  : 6.25% (hold)
# Step 16 = Aug 2023  : 6.25% (hold)
# Step 17 = Sep 2023  : 6.25% (hold)
# Step 18 = Oct 2023  : 6.50% (+25bps, off-cycle)
# Step 19-24          : 6.50% (hold)

BENCHMARK_RRP_PATH: list = [
    0.0200,  # Step 1  — May 2022 baseline
    0.0225,  # Step 2  — Jun 2022 +25bps
    0.0250,  # Step 3  — Jul 2022 interpolated
    0.0325,  # Step 4  — Aug 2022 +75bps
    0.0400,  # Step 5  — Sep 2022 +75bps
    0.0400,  # Step 6  — Oct 2022 hold
    0.0500,  # Step 7  — Nov 2022 +100bps
    0.0500,  # Step 8  — Dec 2022 hold
    0.0500,  # Step 9  — Jan 2023 hold
    0.0550,  # Step 10 — Feb 2023 +50bps
    0.0625,  # Step 11 — Mar 2023 +75bps
    0.0625,  # Step 12 — Apr 2023 hold
    0.0625,  # Step 13 — May 2023 hold
    0.0625,  # Step 14 — Jun 2023 hold
    0.0625,  # Step 15 — Jul 2023 hold
    0.0625,  # Step 16 — Aug 2023 hold
    0.0625,  # Step 17 — Sep 2023 hold
    0.0650,  # Step 18 — Oct 2023 +25bps off-cycle
    0.0650,  # Step 19 — Nov 2023 hold
    0.0650,  # Step 20 — Dec 2023 hold
    0.0650,  # Step 21 — Jan 2024 hold
    0.0650,  # Step 22 — Feb 2024 hold
    0.0650,  # Step 23 — Mar 2024 hold
    0.0650,  # Step 24 — Apr 2024 hold
]

# Real-world anchor points for validation chart
# Source: BSP lending rate statistics, PSA National Accounts 2023
BENCHMARK_REAL_DATA: dict = {
    "lending_rate_start":    0.0557,  # 5.57% avg lending rate May 2022 (BSP)
    "lending_rate_peak":     0.0780,  # 7.80% avg lending rate Apr 2023 (BSP/Statista)
    "lending_rate_change":   0.0172,  # +172bps actual lending rate rise
    "rrp_change":            0.0425,  # +425bps RRP change (May 2022 to Mar 2023)
    "pass_through_ratio":    0.405,   # 40.5% pass-through (172/425)
    "hfce_growth_2023":      0.056,   # +5.6% household consumption growth (PSA 2023)
    "gcf_growth_2023":       0.054,   # +5.4% gross capital formation growth (PSA 2023)
    "gcf_q2_2023":          -0.0004,  # -0.04% GCF contraction at peak rates (PSA Q2 2023)
}

# =============================================================================
# SIMULATION PARAMETERS
# =============================================================================

SIMULATION_STEPS: int = 24    # 24 monthly steps
RANDOM_SEED: int = 42