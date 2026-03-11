"""
Run the model on the ACTUAL 2022-2023 BSP hike cycle RRP path and
compare model outputs against real published data anchors from:
  - BSP lending/deposit rate statistics
  - PSA National Accounts 2023
  - PSA FIES 2023

Two outputs are produced:
  1. Validation summary table (printed to console)
  2. Validation chart — model output vs real data benchmarks

Benchmark data sources:
  - BSP average lending rate: 5.57% (May 2022) -> 7.80% (Apr 2023)
  - Pass-through ratio: 172bps lending / 425bps RRP = 40.5%
  - HFCE growth 2023: +5.6% (PSA National Accounts)
  - GCF growth 2023: +5.4% (PSA National Accounts)
  - GCF Q2 2023 at peak rates: -0.04% (PSA Q2 2023 GDP release)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np

from model.bsp_model import BSPModel
from config.parameters import (
    BENCHMARK_RRP_PATH,
    BENCHMARK_REAL_DATA,
    NUM_BANKS,
    NUM_HOUSEHOLDS,
    NUM_FIRMS,
    RANDOM_SEED,
)


# ---------------------------------------------------------------------------
# Run benchmark scenario
# ---------------------------------------------------------------------------

def run_benchmark() -> tuple[BSPModel, pd.DataFrame]:
    """
    Run the model on the actual 2022-2023 BSP RRP path.

    Returns
    -------
    tuple[BSPModel, pd.DataFrame]
        Completed model and its time series DataFrame.
    """
    model = BSPModel(
        rrp_rate=BENCHMARK_RRP_PATH[0],
        num_banks=NUM_BANKS,
        num_households=NUM_HOUSEHOLDS,
        num_firms=NUM_FIRMS,
        seed=RANDOM_SEED,
    )

    for step_idx, rrp in enumerate(BENCHMARK_RRP_PATH):
        model.set_rrp_rate(rrp)
        model.step()

    df = model.datacollector.get_model_vars_dataframe().copy()
    df.index = range(1, len(df) + 1)
    df.index.name = "Step"
    return model, df


# ---------------------------------------------------------------------------
# Validation summary table
# ---------------------------------------------------------------------------

def print_validation_summary(df: pd.DataFrame) -> dict:
    """
    Compare key model outputs against real-world benchmarks.
    Prints a formatted table and returns the comparison dict.

    Checks
    ------
    1. Lending rate at peak — model vs BSP published rate
    2. Pass-through ratio — model vs actual 40.5%
    3. Lending rate change magnitude — model vs +172bps actual
    4. Direction of consumption change at peak (should fall or slow)
    5. Direction of capex change at peak (should contract or slow)

    Parameters
    ----------
    df : pd.DataFrame
        Model time series from benchmark run.

    Returns
    -------
    dict
        Comparison results with pass/fail flags.
    """
    real = BENCHMARK_REAL_DATA

    # Model values at key points
    lending_start  = df.loc[1,  "Avg_Lending_Rate"]
    lending_peak   = df.loc[11, "Avg_Lending_Rate"]   # Step 11 = Mar 2023 = RRP 6.25%
    rrp_start      = df.loc[1,  "RRP_Rate"]
    rrp_peak       = df.loc[11, "RRP_Rate"]
    consumption_s1 = df.loc[1,  "Total_Consumption"]
    consumption_s7 = df.loc[7,  "Total_Consumption"]   # Step 7 = Nov 2022 = first big hike
    consumption_s11= df.loc[11, "Total_Consumption"]
    capex_s1       = df.loc[1,  "Total_Capex"]
    capex_s11      = df.loc[11, "Total_Capex"]
    capex_s7       = df.loc[7,  "Total_Capex"]

    model_lending_change = lending_peak - lending_start
    model_rrp_change     = rrp_peak - rrp_start
    model_passthrough    = model_lending_change / model_rrp_change if model_rrp_change else 0

    results = {
        "lending_rate_start": {
            "label":    "Lending rate at start (May 2022)",
            "model":    f"{lending_start:.2%}",
            "real":     f"{real['lending_rate_start']:.2%}",
            "pass":     abs(lending_start - real["lending_rate_start"]) < 0.02,
            "note":     "Tolerance ±2pp",
        },
        "lending_rate_peak": {
            "label":    "Lending rate near peak (Mar 2023)",
            "model":    f"{lending_peak:.2%}",
            "real":     f"{real['lending_rate_peak']:.2%}",
            "pass":     abs(lending_peak - real["lending_rate_peak"]) < 0.02,
            "note":     "Tolerance ±2pp",
        },
        "lending_rate_change": {
            "label":    "Lending rate rise (bps)",
            "model":    f"{model_lending_change * 10000:.0f}bps",
            "real":     f"{real['lending_rate_change'] * 10000:.0f}bps",
            "pass":     abs(model_lending_change - real["lending_rate_change"]) < 0.015,
            "note":     "Tolerance ±150bps",
        },
        "pass_through": {
            "label":    "RRP-to-lending pass-through ratio",
            "model":    f"{model_passthrough:.1%}",
            "real":     f"{real['pass_through_ratio']:.1%}",
            "pass":     abs(model_passthrough - real["pass_through_ratio"]) < 0.15,
            "note":     "Tolerance ±15pp",
        },
        "consumption_direction": {
            "label":    "Consumption falls after hike?",
            "model":    "YES" if consumption_s11 < consumption_s1 else "NO",
            "real":     "SLOWED (HFCE +5.6% vs prior trend)",
            "pass":     consumption_s11 <= consumption_s1 * 1.02,
            "note":     "Direction check (resilient PH consumption)",
        },
        "capex_direction": {
            "label":    "Capex contracts at peak?",
            "model":    "YES" if capex_s11 < capex_s7 else "NO",
            "real":     "YES (GCF -0.04% Q2 2023)",
            "pass":     capex_s11 <= capex_s7,
            "note":     "Direction check vs Q2 2023 GCF",
        },
        "firms_investing_peak": {
            "label":    "% firms investing at peak",
            "model":    f"{df.loc[11, 'Pct_Firms_Investing']:.0%}",
            "real":     "~60-70% (GCF still positive full-year 2023)",
            "pass":     0.4 <= df.loc[11, "Pct_Firms_Investing"] <= 0.85,
            "note":     "Range 40-85%",
        },
    }

    # Print table
    print("\n" + "=" * 78)
    print("BSP ABM — Session 5 Validation Summary")
    print("Benchmark: Actual 2022-2023 BSP Hike Cycle (May 2022 – Apr 2024)")
    print("=" * 78)
    print(f"{'Check':<42} {'Model':>10} {'Real':>14} {'Pass':>6}")
    print("-" * 78)
    for key, r in results.items():
        status = "✓" if r["pass"] else "✗"
        print(f"{r['label']:<42} {r['model']:>10} {r['real']:>14} {status:>6}")
    print("-" * 78)
    passed = sum(1 for r in results.values() if r["pass"])
    total  = len(results)
    print(f"Result: {passed}/{total} checks passed")
    print("=" * 78)

    return results


# ---------------------------------------------------------------------------
# Validation chart
# ---------------------------------------------------------------------------

def plot_validation(
    df: pd.DataFrame,
    results: dict,
    save_path: str | None = None,
    show: bool = False,
) -> None:
    """
    Produce a 4-panel validation chart comparing model output against
    real BSP/PSA data anchors.

    Panels
    ------
    1. RRP path — actual vs model input (should be identical)
    2. Lending rate — model vs BSP published rate anchors
    3. Consumption index — model trajectory (normalized to step 1 = 100)
    4. Capex/Investment index — model trajectory with GCF benchmark

    Parameters
    ----------
    df : pd.DataFrame
        Model time series.
    results : dict
        Validation results from print_validation_summary().
    save_path : str or None
        Path to save PNG.
    show : bool
        Display interactively.
    """
    real = BENCHMARK_REAL_DATA
    steps = list(df.index)

    COLORS = {
        "model":    "#1D3557",
        "real":     "#E63946",
        "band":     "#A8DADC",
        "rrp":      "#F4A261",
        "pass":     "#2D6A4F",
        "fail":     "#E63946",
    }

    fig = plt.figure(figsize=(15, 18))
    fig.patch.set_facecolor("#F8F9FA")
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.55, wspace=0.35,
                           top=0.92, bottom=0.05, left=0.09, right=0.95)

    # ── Panel 1 (top-left): RRP Path ─────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(steps, [r * 100 for r in BENCHMARK_RRP_PATH],
             color=COLORS["real"], linewidth=2.5, linestyle="--",
             label="Actual BSP RRP Path", zorder=3)
    ax1.plot(steps, df["RRP_Rate"] * 100,
             color=COLORS["model"], linewidth=1.5, alpha=0.6,
             label="Model RRP Input")
    ax1.set_title("① RRP Rate Path (Input)", fontsize=10,
                  fontweight="bold", loc="left", color="#1D3557")
    ax1.set_ylabel("Rate (%)")
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    ax1.legend(fontsize=8)
    _style_val_panel(ax1, steps)

    # ── Panel 2 (top-right): Lending Rate ────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(steps, df["Avg_Lending_Rate"] * 100,
             color=COLORS["model"], linewidth=2.5, label="Model Avg Lending Rate")

    # Real data anchors — two points from BSP published data
    ax2.scatter([1, 11], [real["lending_rate_start"] * 100,
                           real["lending_rate_peak"] * 100],
                color=COLORS["real"], s=100, zorder=5,
                label="BSP Published Data", marker="D")

    # Annotate anchors
    ax2.annotate(f"BSP: {real['lending_rate_start']:.2%}",
                 xy=(1, real["lending_rate_start"] * 100),
                 xytext=(2.5, real["lending_rate_start"] * 100 - 0.3),
                 fontsize=8, color=COLORS["real"],
                 arrowprops=dict(arrowstyle="->", color=COLORS["real"], lw=0.8))
    ax2.annotate(f"BSP: {real['lending_rate_peak']:.2%}",
                 xy=(11, real["lending_rate_peak"] * 100),
                 xytext=(12.5, real["lending_rate_peak"] * 100 - 0.3),
                 fontsize=8, color=COLORS["real"],
                 arrowprops=dict(arrowstyle="->", color=COLORS["real"], lw=0.8))

    # Pass-through annotation box
    pt_model = results["pass_through"]["model"]
    pt_real  = results["pass_through"]["real"]
    ax2.text(0.97, 0.05,
             f"Pass-through\nModel: {pt_model}\nActual: {pt_real}",
             transform=ax2.transAxes, fontsize=8, ha="right",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF3CD",
                       edgecolor="#F4A261", alpha=0.9))

    ax2.set_title("② Lending Rate — Model vs BSP Data", fontsize=10,
                  fontweight="bold", loc="left", color="#1D3557")
    ax2.set_ylabel("Rate (%)")
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    ax2.legend(fontsize=8)
    _style_val_panel(ax2, steps)

    # ── Panel 3 (mid-left): Consumption Index ────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    base_c = df.loc[1, "Total_Consumption"]
    c_index = (df["Total_Consumption"] / base_c * 100)
    ax3.plot(steps, c_index, color=COLORS["model"], linewidth=2.5,
             label="Model Consumption (indexed)")
    ax3.axhline(y=100, color="grey", linewidth=0.8, linestyle=":")

    # PSA HFCE growth benchmark: +5.6% over 2023 (12 steps)
    # Plot as a reference band — what "real" resilient consumption looks like
    real_end = 100 * (1 + real["hfce_growth_2023"])
    ax3.annotate("",
                 xy=(24, real_end), xytext=(1, 100),
                 arrowprops=dict(arrowstyle="->", color=COLORS["real"],
                                 lw=1.5, linestyle="dashed"))
    ax3.text(13, (100 + real_end) / 2 + 1,
             f"PSA HFCE +{real['hfce_growth_2023']:.1%}\n(2023 actual)",
             fontsize=8, color=COLORS["real"], ha="center")

    ax3.set_title("③ Consumption Index (Step 1 = 100)", fontsize=10,
                  fontweight="bold", loc="left", color="#1D3557")
    ax3.set_ylabel("Index (Step 1 = 100)")
    ax3.legend(fontsize=8)
    _style_val_panel(ax3, steps)

    # ── Panel 4 (mid-right): Capex Index ─────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    base_k = df.loc[1, "Total_Capex"]
    k_index = (df["Total_Capex"] / base_k * 100)
    ax4.plot(steps, k_index, color=COLORS["model"], linewidth=2.5,
             label="Model Total Capex (indexed)")
    ax4.axhline(y=100, color="grey", linewidth=0.8, linestyle=":")

    # GCF benchmark: near-flat at Q2 peak (step 11), then +5.4% full year
    ax4.scatter([11], [100 + real["gcf_q2_2023"] * 100],
                color=COLORS["real"], s=100, zorder=5, marker="D",
                label=f"PSA GCF Q2 2023 ({real['gcf_q2_2023']:.2%})")
    ax4.scatter([24], [100 + real["gcf_growth_2023"] * 100],
                color="#F4A261", s=100, zorder=5, marker="D",
                label=f"PSA GCF Full-Year 2023 (+{real['gcf_growth_2023']:.1%})")

    ax4.set_title("④ Capex Index vs PSA GCF Benchmarks", fontsize=10,
                  fontweight="bold", loc="left", color="#1D3557")
    ax4.set_ylabel("Index (Step 1 = 100)")
    ax4.legend(fontsize=8)
    _style_val_panel(ax4, steps)

    # ── Panel 5 (bottom-left): Validation Scorecard ──────────────────
    ax5 = fig.add_subplot(gs[2:, :])
    ax5.axis("off")

    scorecard_data = [
        [r["label"], r["model"], r["real"], "PASS ✓" if r["pass"] else "FAIL ✗", r["note"]]
        for r in results.values()
    ]
    col_labels = ["Validation Check", "Model Output", "Real Benchmark",
                  "Status", "Notes"]

    table = ax5.table(
        cellText=scorecard_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)

    # Style header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#1D3557")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Style rows — green for pass, red for fail
    for i, r in enumerate(results.values()):
        color = "#D4EDDA" if r["pass"] else "#F8D7DA"
        for j in range(len(col_labels)):
            table[i + 1, j].set_facecolor(color)

    ax5.set_title("Validation Scorecard — Model vs Real BSP/PSA Data",
                  fontsize=11, fontweight="bold", loc="left",
                  pad=12, color="#1D3557")

    # ── Overall title ─────────────────────────────────────────────────
    passed = sum(1 for r in results.values() if r["pass"])
    total  = len(results)
    fig.suptitle(
        f"BSP ABM — Session 5 Validation | Benchmark: 2022–2023 Hike Cycle "
        f"| Score: {passed}/{total} checks passed",
        fontsize=13, fontweight="bold", y=0.97, color="#1D3557"
    )
    fig.text(
        0.5, 0.01,
        "Sources: BSP lending rate statistics · PSA National Accounts 2023 · PSA FIES 2023",
        ha="center", fontsize=8, color="#6C757D", style="italic"
    )

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"Validation chart saved: {save_path}")

    if show:
        plt.show()

    plt.close(fig)


def _style_val_panel(ax: plt.Axes, steps: list) -> None:
    """Consistent styling for validation panels."""
    ax.set_facecolor("#FFFFFF")
    ax.set_xlabel("Step (months: May 2022 = 1)", fontsize=8)
    ax.set_xlim(steps[0], steps[-1])
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(4))
    # Month labels
    month_labels = {
        1: "May'22", 5: "Sep'22", 7: "Nov'22", 11: "Mar'23",
        18: "Oct'23", 24: "Apr'24"
    }
    ax.set_xticks(list(month_labels.keys()))
    ax.set_xticklabels(list(month_labels.values()), fontsize=7.5)


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------

def run_sensitivity_analysis(
    param: str,
    values: list,
    save_path: str | None = None,
    show: bool = False,
) -> pd.DataFrame:
    """
    Run the benchmark scenario across a range of values for one parameter
    and plot how GDP Proxy and Lending Rate respond.

    Parameters
    ----------
    param : str
        Parameter name in config.parameters (e.g. 'BANK_CREDIT_SPREAD').
    values : list
        List of values to test.
    save_path : str or None
        Optional save path for chart.
    show : bool
        Display interactively.

    Returns
    -------
    pd.DataFrame
        Summary table: one row per value tested.
    """
    import config.parameters as params_module

    original = getattr(params_module, param)
    records = []

    for val in values:
        setattr(params_module, param, val)
        model = BSPModel(
            rrp_rate=BENCHMARK_RRP_PATH[0],
            seed=RANDOM_SEED,
        )
        for rrp in BENCHMARK_RRP_PATH:
            model.set_rrp_rate(rrp)
            model.step()
        df = model.datacollector.get_model_vars_dataframe()
        records.append({
            param:              val,
            "lending_rate_end": df["Avg_Lending_Rate"].iloc[-1],
            "gdp_proxy_end":    df["GDP_Proxy"].iloc[-1],
            "gdp_proxy_min":    df["GDP_Proxy"].min(),
            "pct_investing_min":df["Pct_Firms_Investing"].min(),
            "consumption_end":  df["Total_Consumption"].iloc[-1],
        })

    # Restore original
    setattr(params_module, param, original)

    summary = pd.DataFrame(records)

    if save_path or show:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.patch.set_facecolor("#F8F9FA")
        for ax, col, label in zip(
            axes,
            ["lending_rate_end", "gdp_proxy_min", "pct_investing_min"],
            ["Lending Rate at End", "GDP Proxy (minimum)", "Min % Firms Investing"],
        ):
            ax.plot(values, summary[col], marker="o", color="#1D3557", linewidth=2)
            ax.axvline(x=original, color="#E63946", linestyle="--",
                       linewidth=1.2, label=f"Calibrated: {original}")
            ax.set_title(label, fontsize=10, fontweight="bold")
            ax.set_xlabel(param, fontsize=9)
            ax.legend(fontsize=8)
            ax.set_facecolor("#FFFFFF")
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.spines[["top", "right"]].set_visible(False)

        fig.suptitle(f"Sensitivity Analysis — {param}",
                     fontsize=12, fontweight="bold", color="#1D3557")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            print(f"Sensitivity chart saved: {save_path}")
        if show:
            plt.show()
        plt.close(fig)

    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running benchmark scenario (2022-2023 hike cycle)...")
    model, df = run_benchmark()

    results = print_validation_summary(df)

    plot_validation(
        df, results,
        save_path="/mnt/user-data/outputs/bsp_validation.png",
        show=False,
    )

    print("\nRunning sensitivity analysis on BANK_CREDIT_SPREAD...")
    sens = run_sensitivity_analysis(
        param="BANK_CREDIT_SPREAD",
        values=[0.006, 0.008, 0.010, 0.012, 0.015, 0.020, 0.025],
        save_path="/mnt/user-data/outputs/bsp_sensitivity_spread.png",
        show=False,
    )
    print(sens.to_string(index=False))