"""
AI Risk & Reliability Platform - Enterprise Financial Simulator Module
File: financial_simulator.py
Description: Models enterprise ROI, risk exposure cost savings, cost of model failures,
             and payback periods resulting from automated data cleaning and risk mitigation.
"""

from typing import Dict, Any


class FinancialSimulator:
    """
    Financial business case and ROI calculation engine for AI platform deployment.
    """

    def __init__(self, 
                 annual_predictions: int = 100000,
                 cost_per_prediction_error: float = 250.0,
                 baseline_error_rate: float = 0.08,
                 post_cleaning_error_rate: float = 0.03,
                 platform_annual_cost: float = 15000.0):
        self.annual_predictions = annual_predictions
        self.cost_per_error = cost_per_prediction_error
        self.baseline_error_rate = baseline_error_rate
        self.post_cleaning_error_rate = post_cleaning_error_rate
        self.platform_cost = platform_annual_cost

    def simulate_roi(self) -> Dict[str, Any]:
        """Runs enterprise financial simulation and returns savings and ROI metrics."""
        baseline_errors = int(self.annual_predictions * self.baseline_error_rate)
        post_cleaning_errors = int(self.annual_predictions * self.post_cleaning_error_rate)
        prevented_errors = baseline_errors - post_cleaning_errors

        baseline_loss_cost = round(baseline_errors * self.cost_per_error, 2)
        post_cleaning_loss_cost = round(post_cleaning_errors * self.cost_per_error, 2)

        gross_annual_savings = round(prevented_errors * self.cost_per_error, 2)
        net_annual_benefit = round(gross_annual_savings - self.platform_cost, 2)

        # ROI = (Net Benefit / Platform Cost) * 100
        roi_percentage = round((net_annual_benefit / self.platform_cost) * 100, 2) if self.platform_cost > 0 else 0.0

        # Payback period in months
        monthly_savings = gross_annual_savings / 12.0
        payback_months = round(self.platform_cost / monthly_savings, 1) if monthly_savings > 0 else 0.0

        return {
            "annual_predictions": self.annual_predictions,
            "cost_per_error": self.cost_per_error,
            "baseline_annual_errors": baseline_errors,
            "post_cleaning_annual_errors": post_cleaning_errors,
            "prevented_errors": prevented_errors,
            "baseline_financial_loss": baseline_loss_cost,
            "post_cleaning_financial_loss": post_cleaning_loss_cost,
            "gross_annual_savings": gross_annual_savings,
            "platform_annual_cost": self.platform_cost,
            "net_annual_benefit": net_annual_benefit,
            "roi_percentage": roi_percentage,
            "payback_period_months": payback_months
        }


# Standalone Demonstration Execution
if __name__ == "__main__":
    print("=" * 60)
    print("  AI RISK PLATFORM — FINANCIAL SIMULATOR MODULE DEMO")
    print("=" * 60)

    simulator = FinancialSimulator(
        annual_predictions=100000,
        cost_per_prediction_error=350.0,
        baseline_error_rate=0.085,
        post_cleaning_error_rate=0.025,
        platform_annual_cost=25000.0
    )

    results = simulator.simulate_roi()

    print(f"Annual Prediction Volume: {results['annual_predictions']:,}")
    print(f"Cost per Model Error     : ${results['cost_per_error']:,.2f}")
    print(f"Prevented Failures/Yr   : {results['prevented_errors']:,} errors")
    print(f"Baseline Risk Exposure  : ${results['baseline_financial_loss']:,.2f}")
    print(f"Gross Annual Savings    : ${results['gross_annual_savings']:,.2f}")
    print(f"Net Financial Benefit   : ${results['net_annual_benefit']:,.2f}")
    print(f"Enterprise Platform ROI : {results['roi_percentage']}%")
    print(f"Payback Period          : {results['payback_period_months']} months")
    print("=" * 60)
