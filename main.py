"""
Entry point for the BSP Monetary Policy ABM.
"""

from model.bsp_model import BSPModel
from config.parameters import SIMULATION_STEPS
from analysis.charts import run_and_plot, compare_scenarios


def run_demo() -> None:
    """
    Demo scenario: Start at 4% RRP, hike to 6.5% at step 6,
    hold, then cut back to 5% at step 18.

    Output shows the full transmission chain:
    RRP -> Lending Rate -> Credit Supply -> Consumption -> Capex -> GDP Proxy
    """
    print("=" * 95)
    print("BSP Monetary Policy ABM -- Session 3 Demo (Full Transmission Chain)")
    print("Scenario: Rate hike (4% -> 6.5%) at step 6 | Rate cut (-> 5%) at step 18")
    print("=" * 95)
    print(
        f"{'Step':>4} | {'RRP':>5} | {'Lending':>7} | "
        f"{'Consumption':>13} | {'Capex':>13} | "
        f"{'FirmsInvesting':>14} | {'GDP Proxy':>13}"
    )
    print("-" * 95)

    model = BSPModel(rrp_rate=0.04)

    for step in range(SIMULATION_STEPS):
        if step == 6:
            model.set_rrp_rate(0.065)
            print("  >>> BSP HIKE: RRP -> 6.5%")
        if step == 18:
            model.set_rrp_rate(0.05)
            print("  >>> BSP CUT:  RRP -> 5.0%")

        model.step()

        df = model.datacollector.get_model_vars_dataframe()
        r = df.iloc[-1]

        print(
            f"{step + 1:>4} | "
            f"{r['RRP_Rate']:>5.2%} | "
            f"{r['Avg_Lending_Rate']:>7.2%} | "
            f"PHP {r['Total_Consumption']:>9,.0f} | "
            f"PHP {r['Total_Capex']:>9,.0f} | "
            f"{r['Pct_Firms_Investing']:>14.1%} | "
            f"PHP {r['GDP_Proxy']:>9,.0f}"
        )

    print("=" * 95)
    # Full 5-panel transmission chain
    run_and_plot(scenario="hike_then_cut", save_path="chart.png", show=True)

    # Side-by-side hike vs cut comparison
    compare_scenarios(scenario_a="hike", scenario_b="cut", save_path="comparison.png", show=True)


if __name__ == "__main__":
    run_demo()