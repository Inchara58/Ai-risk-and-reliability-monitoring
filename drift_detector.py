"""
AI Risk & Reliability Platform - Population Stability & Data Drift Engine (Stage 4)
File: drift_detector.py
Description: Evaluates statistical distribution shifts between explicit baseline reference datasets
             (e.g., Training v1.0, Benchmark Baseline) and current production datasets (v2.0, Live Batch).
             Calculates Population Stability Index (PSI) and two-sample Kolmogorov-Smirnov (KS) tests.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any, List, Union, Optional


class DriftDetector:
    """
    Data and feature distribution drift detection engine calculating
    empirical PSI (Population Stability Index) and Kolmogorov-Smirnov tests.
    """

    def __init__(self, baseline_df: pd.DataFrame, current_df: pd.DataFrame,
                 baseline_label: str = "Baseline Dataset (v1.0 / Benchmark)",
                 current_label: str = "Current Evaluation Dataset (v2.0 / Production)"):
        self.baseline_df = baseline_df.copy()
        self.current_df = current_df.copy()
        self.baseline_label = baseline_label
        self.current_label = current_label

    @staticmethod
    def calculate_psi(baseline: np.ndarray, current: np.ndarray, num_bins: int = 10) -> float:
        """
        Calculates Population Stability Index (PSI) between two 1D numeric arrays.
        PSI Formula: sum((Actual% - Expected%) * ln(Actual% / Expected%))
        """
        baseline = baseline[~np.isnan(baseline)]
        current = current[~np.isnan(current)]

        if len(baseline) == 0 or len(current) == 0:
            return 0.0

        # Define bin edges based on baseline distribution
        percentiles = np.linspace(0, 100, num_bins + 1)
        bin_edges = np.percentile(baseline, percentiles)
        bin_edges = np.unique(bin_edges)  # Remove duplicates in case of repeated values

        if len(bin_edges) < 2:
            return 0.0

        # Bin counts
        baseline_counts, _ = np.histogram(baseline, bins=bin_edges)
        current_counts, _ = np.histogram(current, bins=bin_edges)

        # Proportions with Laplace smoothing epsilon to prevent log(0)
        eps = 1e-4
        baseline_props = (baseline_counts / max(1, len(baseline))) + eps
        current_props = (current_counts / max(1, len(current))) + eps

        # Normalize to sum to 1
        baseline_props /= np.sum(baseline_props)
        current_props /= np.sum(current_props)

        psi_value = float(np.sum((current_props - baseline_props) * np.log(current_props / baseline_props)))
        return round(max(0.0, psi_value), 4)

    def detect_drift(self) -> Dict[str, Any]:
        """
        Performs full empirical distribution drift detection across shared numeric features.
        Returns per-feature metrics, overall PSI score, stability rating, and statistical interpretations.
        """
        shared_cols = [
            col for col in self.baseline_df.columns
            if col in self.current_df.columns and pd.api.types.is_numeric_dtype(self.baseline_df[col])
        ]

        feature_drift = {}
        total_psi = 0.0
        drifted_features = []
        stable_features = []

        for col in shared_cols:
            base_arr = self.baseline_df[col].dropna().values.astype(float)
            curr_arr = self.current_df[col].dropna().values.astype(float)

            if len(base_arr) > 5 and len(curr_arr) > 5:
                psi_val = self.calculate_psi(base_arr, curr_arr)
                
                # Two-sample Kolmogorov-Smirnov test
                try:
                    ks_stat, p_value = stats.ks_2samp(base_arr, curr_arr)
                    ks_stat = float(ks_stat)
                    p_value = float(p_value)
                except Exception:
                    ks_stat, p_value = 0.0, 1.0

                # Statistical classification based on Basel / Credit Risk Standards:
                # PSI < 0.10: No significant shift (Stable)
                # 0.10 <= PSI < 0.25: Moderate shift (Monitor / Investigate)
                # PSI >= 0.25: Significant statistical shift (Action Required / Model Retrain)
                if psi_val >= 0.25:
                    status = "SEVERE_DRIFT"
                    severity = "CRITICAL"
                    interpretation = "Significant distribution shift detected. Model degradation likely; retraining recommended."
                    drifted_features.append(col)
                elif psi_val >= 0.10:
                    status = "MODERATE_DRIFT"
                    severity = "WARNING"
                    interpretation = "Moderate distribution divergence. Recommend monitoring feature in production pipeline."
                    drifted_features.append(col)
                else:
                    status = "STABLE"
                    severity = "NORMAL"
                    interpretation = "Feature distribution is statistically consistent with baseline reference."
                    stable_features.append(col)

                feature_drift[col] = {
                    "feature_name": col,
                    "psi_score": psi_val,
                    "ks_statistic": round(ks_stat, 4),
                    "p_value": round(p_value, 4),
                    "is_statistically_significant": p_value < 0.05,
                    "status": status,
                    "severity": severity,
                    "interpretation": interpretation,
                    "baseline_mean": round(float(np.mean(base_arr)), 3),
                    "current_mean": round(float(np.mean(curr_arr)), 3),
                    "baseline_std": round(float(np.std(base_arr)), 3),
                    "current_std": round(float(np.std(curr_arr)), 3)
                }
                total_psi += psi_val

        feature_count = len(shared_cols)
        avg_psi = round(total_psi / max(1, feature_count), 4)

        if avg_psi >= 0.25 or len(drifted_features) > (feature_count * 0.4):
            overall_status = "CRITICAL_DRIFT_ALERT"
            drift_score = max(0.0, round(100.0 - (avg_psi * 150.0), 1))
            summary_message = f"Critical population drift detected across {len(drifted_features)} features. Model recalibration required."
        elif avg_psi >= 0.10 or len(drifted_features) > 0:
            overall_status = "MODERATE_DRIFT_WARNING"
            drift_score = max(50.0, round(100.0 - (avg_psi * 120.0), 1))
            summary_message = f"Moderate distribution drift identified in {len(drifted_features)} feature(s). Ongoing surveillance advised."
        else:
            overall_status = "DATASET_STABLE"
            drift_score = round(max(85.0, 100.0 - (avg_psi * 50.0)), 1)
            summary_message = "All evaluated features show statistical alignment with reference baseline distribution."

        return {
            "baseline_label": self.baseline_label,
            "current_label": self.current_label,
            "baseline_rows": len(self.baseline_df),
            "current_rows": len(self.current_df),
            "overall_psi_score": avg_psi,
            "drift_reliability_score": drift_score,
            "overall_status": overall_status,
            "summary_message": summary_message,
            "total_features_evaluated": feature_count,
            "drifted_features_count": len(drifted_features),
            "stable_features_count": len(stable_features),
            "drifted_features": drifted_features,
            "feature_drift_details": feature_drift
        }

    def calculate_psi_dict(self) -> Dict[str, Any]:
        """Convenience method returning summary dict for risk engine integration."""
        res = self.detect_drift()
        return {
            "psi_score": res["overall_psi_score"],
            "drift_score": res["drift_reliability_score"],
            "status": res["overall_status"],
            "drifted_count": res["drifted_features_count"]
        }


# Standalone Demonstration Execution
if __name__ == "__main__":
    print("=" * 65)
    print("      AEGIS AI — DATA DRIFT DETECTOR MODULE DEMO")
    print("=" * 65)

    np.random.seed(42)
    # Baseline distribution
    base_data = pd.DataFrame({
        "Temperature": np.random.normal(70, 5, 500),
        "Vibration": np.random.normal(0.45, 0.05, 500),
        "Pressure": np.random.normal(100, 10, 500)
    })

    # Current production distribution with severe drift in Temperature and Pressure
    curr_data = pd.DataFrame({
        "Temperature": np.random.normal(82, 8, 500),   # Shifted (Drift)
        "Vibration": np.random.normal(0.46, 0.05, 500), # Stable
        "Pressure": np.random.normal(125, 15, 500)     # Shifted (Drift)
    })

    detector = DriftDetector(
        base_data, curr_data,
        baseline_label="Baseline Training Dataset (v1.0)",
        current_label="Production Batch (v2.0)"
    )
    res = detector.detect_drift()

    print(f"Baseline: {res['baseline_label']} ({res['baseline_rows']} rows)")
    print(f"Current : {res['current_label']} ({res['current_rows']} rows)")
    print(f"Overall PSI Score       : {res['overall_psi_score']}")
    print(f"Overall Status          : {res['overall_status']}")
    print(f"Drifted Features        : {res['drifted_features_count']} / {res['total_features_evaluated']}")
    print("\nFeature Details:")
    for feat, info in res['feature_drift_details'].items():
        print(f" - {feat:<12}: PSI={info['psi_score']:<6} | KS={info['ks_statistic']} (p={info['p_value']}) -> [{info['status']}]")
    print("=" * 65)
