"""
Visualization module for the BSP Monetary Policy ABM.

Produces a multi-panel chart showing the full monetary transmission chain:
  Panel 1: BSP RRP Rate & Bank Rates
  Panel 2: Credit Supply & Reserve Ratio
  Panel 3: Household Consumption & Savings Rate
  Panel 4: Firm Capex & % Firms Investing
  Panel 5: GDP Proxy = Consumption + Capex

Usage:
    from analysis.charts import run_and_plot, compare_scenarios
    run_and_plot(scenario="hike_then_cut", save_path="chart.png", show=False)
"""

import sys
import os

# Ensure project root is on sys.path regardless of where this file is run from
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec

from model.bsp_model import BSPModel
from config.parameters import SIMULATION_STEPS

COLORS = {
    "rrp":         "#E63946",
    "lending":     "#F4A261",
    "deposit":     "#2A9D8F",
    "credit":      "#457B9D",
    "reserves":    "#A8DADC",
    "consumption": "#2D6A4F",
    "savings":     "#95D5B2",
    "capex":       "#6A0572",
    "investing":   "#C77DFF",
    "gdp":         "#1D3557",
    "shock":       "#E63946",
}


def build_scenario(scenario: str) -> BSPModel:
    """
    Run a named policy scenario and return the completed model.

    Scenarios
    ---------
    "hike"          : 4% -> 6.5% at step 6
    "cut"           : 6.5% -> 4% at step 6
    "hike_then_cut" : 4% -> 6.5% at step 6, -> 5% at step 18
    """
    scenarios = {
        "hike":          [(6,  0.065)],
        "cut":           [(6,  0.040)],
        "hike_then_cut": [(6,  0.065), (18, 0.050)],
    }
    start_rates = {
        "hike": 0.04, "cut": 0.065, "hike_then_cut": 0.04,
    }
    if scenario not in scenarios:
        raise ValueError(f"Unknown scenario '{scenario}'. "
                         f"Choose from: {list(scenarios.keys())}")

    model = BSPModel(rrp_rate=start_rates[scenario])
    for step in range(SIMULATION_STEPS):
        for shock_step, new_rate in scenarios[scenario]:
            if step == shock_step:
                model.set_rrp_rate(new_rate)
        model.step()
    return model


def get_dataframe(model: BSPModel) -> pd.DataFrame:
    """Extract model time series from DataCollector as a clean DataFrame."""
    df = model.datacollector.get_model_vars_dataframe().copy()
    df.index.name = "Step"
    df.index = df.index + 1
    return df


def _detect_shocks(df: pd.DataFrame) -> list:
    """Return step indices where RRP rate changed."""
    shocks = []
    rates = df["RRP_Rate"].tolist()
    for i in range(1, len(rates)):
        if abs(rates[i] - rates[i - 1]) > 1e-6:
            shocks.append(df.index[i])
    return shocks


def _add_shock_lines(ax: plt.Axes, shock_steps: list, df: pd.DataFrame) -> None:
    """Add vertical marker lines at policy shock points."""
    for step in shock_steps:
        ax.axvline(x=step, color=COLORS["shock"], linestyle=":",
                   linewidth=1.2, alpha=0.7, zorder=2)
        rate = df.loc[step, "RRP_Rate"] * 100
        ax.annotate(
            f"RRP→{rate:.1f}%",
            xy=(step, 1.0), xycoords=("data", "axes fraction"),
            xytext=(4, -14), textcoords="offset points",
            fontsize=7.5, color=COLORS["shock"],
        )


def _style_panel(ax: plt.Axes, title: str, ylabel: str, steps: list) -> None:
    """Apply consistent styling to a chart panel."""
    ax.set_facecolor("#FFFFFF")
    ax.set_title(title, fontsize=10, fontweight="bold",
                 loc="left", pad=6, color="#1D3557")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xlim(steps[0], steps[-1])
    ax.grid(True, linestyle="--", alpha=0.35, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=8)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(4))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))


def run_and_plot(
    scenario: str = "hike_then_cut",
    save_path: str | None = None,
    show: bool = False,
) -> pd.DataFrame:
    """
    Run a scenario and produce the full 5-panel transmission chain chart.

    Parameters
    ----------
    scenario : str
        "hike", "cut", or "hike_then_cut"
    save_path : str or None
        Path to save the PNG. If None, not saved to disk.
    show : bool
        Display chart interactively (requires display).

    Returns
    -------
    pd.DataFrame
        Model time series data.
    """
    model = build_scenario(scenario)
    df = get_dataframe(model)
    steps = df.index.tolist()
    shock_steps = _detect_shocks(df)

    scenario_titles = {
        "hike":          "Rate Hike: 4.0% → 6.5% at Step 6",
        "cut":           "Rate Cut: 6.5% → 4.0% at Step 6",
        "hike_then_cut": "Rate Hike (Step 6: 4%→6.5%) then Cut (Step 18: →5%)",
    }

    fig = plt.figure(figsize=(15, 22))
    fig.patch.set_facecolor("#F8F9FA")
    gs = gridspec.GridSpec(5, 1, figure=fig, hspace=0.55,
                           top=0.93, bottom=0.04, left=0.10, right=0.92)

    # ── Panel 1: Policy & Bank Rates ──────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(steps, df["RRP_Rate"] * 100, color=COLORS["rrp"],
             linewidth=2.5, label="BSP RRP Rate", zorder=3)
    ax1.plot(steps, df["Avg_Lending_Rate"] * 100, color=COLORS["lending"],
             linewidth=2, linestyle="--", label="Avg Lending Rate")
    ax1.plot(steps, df["Avg_Deposit_Rate"] * 100, color=COLORS["deposit"],
             linewidth=2, linestyle=":", label="Avg Deposit Rate")
    _add_shock_lines(ax1, shock_steps, df)
    _style_panel(ax1, "① Policy & Bank Rates", "Rate (%)", steps)
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    ax1.legend(loc="upper left", fontsize=9, framealpha=0.8)

    # ── Panel 2: Credit & Reserves ────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2b = ax2.twinx()
    ax2.plot(steps, df["Total_Credit_Supply"] / 1e6, color=COLORS["credit"],
             linewidth=2.5, label="Total Credit Supply")
    ax2b.plot(steps, df["Avg_Reserve_Ratio"] * 100, color=COLORS["reserves"],
              linewidth=2, linestyle="--", label="Avg Reserve Ratio")
    _add_shock_lines(ax2, shock_steps, df)
    _style_panel(ax2, "② Banking Channel — Credit Supply & Reserves",
                 "Credit Supply (PHP Millions)", steps)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("PHP %.2fM"))
    ax2b.set_ylabel("Reserve Ratio (%)", fontsize=9, color=COLORS["reserves"])
    ax2b.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f%%"))
    ax2b.tick_params(axis="y", colors=COLORS["reserves"])
    lines = ax2.get_legend_handles_labels()[0] + ax2b.get_legend_handles_labels()[0]
    labels = ax2.get_legend_handles_labels()[1] + ax2b.get_legend_handles_labels()[1]
    ax2.legend(lines, labels, loc="upper left", fontsize=9, framealpha=0.8)

    # ── Panel 3: Households ───────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    ax3b = ax3.twinx()
    ax3.plot(steps, df["Total_Consumption"] / 1e3, color=COLORS["consumption"],
             linewidth=2.5, label="Total Consumption")
    ax3b.plot(steps, df["Avg_Savings_Rate"] * 100, color=COLORS["savings"],
              linewidth=2, linestyle="--", label="Avg Savings Rate")
    _add_shock_lines(ax3, shock_steps, df)
    _style_panel(ax3, "③ Consumption Channel — Households",
                 "Consumption (PHP Thousands)", steps)
    ax3.yaxis.set_major_formatter(mticker.FormatStrFormatter("PHP %.0fK"))
    ax3b.set_ylabel("Savings Rate (%)", fontsize=9, color=COLORS["savings"])
    ax3b.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    ax3b.tick_params(axis="y", colors=COLORS["savings"])
    lines = ax3.get_legend_handles_labels()[0] + ax3b.get_legend_handles_labels()[0]
    labels = ax3.get_legend_handles_labels()[1] + ax3b.get_legend_handles_labels()[1]
    ax3.legend(lines, labels, loc="upper left", fontsize=9, framealpha=0.8)

    # ── Panel 4: Firms ────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[3])
    ax4b = ax4.twinx()
    ax4.plot(steps, df["Total_Capex"] / 1e6, color=COLORS["capex"],
             linewidth=2.5, label="Total Capex")
    ax4b.plot(steps, df["Pct_Firms_Investing"] * 100, color=COLORS["investing"],
              linewidth=2, linestyle="--", label="% Firms Investing")
    _add_shock_lines(ax4, shock_steps, df)
    _style_panel(ax4, "④ Investment Channel — Firms",
                 "Total Capex (PHP Millions)", steps)
    ax4.yaxis.set_major_formatter(mticker.FormatStrFormatter("PHP %.1fM"))
    ax4b.set_ylabel("% Firms Investing", fontsize=9, color=COLORS["investing"])
    ax4b.set_ylim(0, 110)
    ax4b.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax4b.tick_params(axis="y", colors=COLORS["investing"])
    lines = ax4.get_legend_handles_labels()[0] + ax4b.get_legend_handles_labels()[0]
    labels = ax4.get_legend_handles_labels()[1] + ax4b.get_legend_handles_labels()[1]
    ax4.legend(lines, labels, loc="upper left", fontsize=9, framealpha=0.8)

    # ── Panel 5: GDP Proxy ────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[4])
    ax5.fill_between(steps, df["Total_Capex"] / 1e6, 0,
                     alpha=0.4, color=COLORS["capex"], label="Capex (I)")
    ax5.fill_between(steps,
                     (df["Total_Consumption"] + df["Total_Capex"]) / 1e6,
                     df["Total_Capex"] / 1e6,
                     alpha=0.4, color=COLORS["consumption"], label="Consumption (C)")
    ax5.plot(steps, df["GDP_Proxy"] / 1e6, color=COLORS["gdp"],
             linewidth=2.5, label="GDP Proxy (C + I)")
    _add_shock_lines(ax5, shock_steps, df)
    _style_panel(ax5, "⑤ Macro Outcome — GDP Proxy (C + I)",
                 "PHP Millions", steps)
    ax5.yaxis.set_major_formatter(mticker.FormatStrFormatter("PHP %.1fM"))
    ax5.legend(loc="upper left", fontsize=9, framealpha=0.8)
    ax5.set_xlabel("Simulation Step (months)", fontsize=10)

    fig.suptitle(
        f"BSP Monetary Policy Transmission — {scenario_titles[scenario]}",
        fontsize=13, fontweight="bold", y=0.97, color="#1D3557"
    )
    fig.text(
        0.5, 0.005,
        "BSP ABM | 5 Banks · 50 Households · 20 Firms | GDP Proxy = C + I",
        ha="center", fontsize=8, color="#6C757D", style="italic"
    )

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"Saved: {save_path}")

    if show:
        plt.show()

    plt.close(fig)
    return df


def compare_scenarios(
    scenario_a: str = "hike",
    scenario_b: str = "cut",
    save_path: str | None = None,
    show: bool = False,
) -> None:
    """
    Overlay two scenarios on GDP Proxy, Consumption, and Capex panels.

    Useful for comparing tightening vs easing effects side by side.
    """
    df_a = get_dataframe(build_scenario(scenario_a))
    df_b = get_dataframe(build_scenario(scenario_b))
    steps = df_a.index.tolist()

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.patch.set_facecolor("#F8F9FA")

    label_a = scenario_a.replace("_", " ").title()
    label_b = scenario_b.replace("_", " ").title()

    metrics = [
        ("GDP_Proxy",         "GDP Proxy (C+I)",     1e6, "PHP Millions"),
        ("Total_Consumption", "Total Consumption",    1e3, "PHP Thousands"),
        ("Total_Capex",       "Total Capex",          1e6, "PHP Millions"),
    ]

    for ax, (col, title, scale, unit) in zip(axes, metrics):
        ax.plot(steps, df_a[col] / scale, color=COLORS["rrp"],
                linewidth=2, label=label_a)
        ax.plot(steps, df_b[col] / scale, color=COLORS["gdp"],
                linewidth=2, linestyle="--", label=label_b)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("Step (months)", fontsize=9)
        ax.set_ylabel(unit, fontsize=9)
        ax.legend(fontsize=9)
        ax.set_facecolor("#FFFFFF")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)

    fig.suptitle(
        f"Scenario Comparison — {label_a} vs {label_b}",
        fontsize=13, fontweight="bold", y=1.02, color="#1D3557"
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"Saved: {save_path}")

    if show:
        plt.show()

    plt.close(fig)