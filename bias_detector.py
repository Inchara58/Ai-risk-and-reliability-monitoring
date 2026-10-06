"""
AI Risk & Reliability Platform - Demographic Bias & Fairness Detector Module (Stage 4)
File: bias_detector.py
Description: Evaluates demographic fairness, Disparate Impact (EEOC 80% rule),
             Demographic Parity Difference, and regulatory compliance.
             Strictly enforces non-fabrication: if no protected attribute is present,
             it reports fairness analysis as unavailable.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Union, Optional, List


# Known protected attribute candidate keywords
DEMOGRAPHIC_CANDIDATE_KEYWORDS = [
    "gender", "sex", "race", "ethnicity", "age", "age_group", 
    "marital_status", "nationality", "religion", "disability"
]


class BiasDetector:
    """
    Fairness and demographic bias auditing engine for tabular datasets and ML model predictions.
    Strictly empirical with explicit attribute validation.
    """

    def __init__(self, df: pd.DataFrame, 
                 protected_col: Optional[str] = None, 
                 target_col: Optional[str] = None, 
                 privileged_group: Optional[Any] = None, 
                 unprivileged_group: Optional[Any] = None, 
                 positive_outcome: Any = 1):
        self.df = df.copy()
        self.protected_col = protected_col
        self.target_col = target_col
        self.privileged_group = privileged_group
        self.unprivileged_group = unprivileged_group
        self.positive_outcome = positive_outcome

    @classmethod
    def detect_candidate_protected_attributes(cls, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Scans DataFrame columns to identify candidate demographic/protected attributes.
        Returns a list of candidate column descriptors with their unique categorical levels.
        """
        candidates = []
        for col in df.columns:
            col_lower = str(col).lower()
            if any(keyword in col_lower for keyword in DEMOGRAPHIC_CANDIDATE_KEYWORDS):
                unique_vals = [str(x) for x in df[col].dropna().unique() if str(x) != "nan"]
                if 2 <= len(unique_vals) <= 20:
                    candidates.append({
                        "column_name": col,
                        "unique_values": unique_vals,
                        "count": len(unique_vals),
                        "suggested_privileged": unique_vals[0] if len(unique_vals) > 0 else None,
                        "suggested_unprivileged": unique_vals[1] if len(unique_vals) > 1 else None
                    })
        return candidates

    def evaluate_bias(self) -> Dict[str, Any]:
        """
        Calculates Disparate Impact Ratio, Demographic Parity Difference, and Fairness Score.
        Returns an empirical assessment or a transparent unavailable status if no protected attributes exist.
        """
        # Step 1: Auto-detect if not explicitly provided
        if not self.protected_col:
            candidates = self.detect_candidate_protected_attributes(self.df)
            if candidates:
                self.protected_col = candidates[0]["column_name"]
                if self.privileged_group is None:
                    self.privileged_group = candidates[0]["suggested_privileged"]
                if self.unprivileged_group is None:
                    self.unprivileged_group = candidates[0]["suggested_unprivileged"]
            else:
                return {
                    "has_protected_attribute": False,
                    "fairness_evaluable": False,
                    "status": "UNAVAILABLE",
                    "disparate_impact_ratio": None,
                    "demographic_parity_diff": None,
                    "fairness_score": None,
                    "is_compliant": None,
                    "compliance_status": "N/A — No Protected Attribute",
                    "explanation": "Fairness analysis cannot be performed with the available attributes. No demographic or protected group features detected.",
                    "candidate_attributes": []
                }

        # Step 2: Validate column existence
        if self.protected_col not in self.df.columns:
            return {
                "has_protected_attribute": False,
                "fairness_evaluable": False,
                "status": "ERROR",
                "disparate_impact_ratio": None,
                "demographic_parity_diff": None,
                "fairness_score": None,
                "is_compliant": None,
                "compliance_status": "Column Not Found",
                "explanation": f"Protected attribute '{self.protected_col}' not found in dataset.",
                "candidate_attributes": self.detect_candidate_protected_attributes(self.df)
            }

        # Step 3: Validate or auto-detect target column
        if not self.target_col or self.target_col not in self.df.columns:
            # Fallback to last column or target column candidates
            num_cols = [c for c in self.df.columns if c != self.protected_col]
            if not num_cols:
                return {
                    "has_protected_attribute": True,
                    "fairness_evaluable": False,
                    "status": "ERROR",
                    "disparate_impact_ratio": None,
                    "demographic_parity_diff": None,
                    "fairness_score": None,
                    "is_compliant": None,
                    "compliance_status": "Target Column Missing",
                    "explanation": "No target outcome column available to evaluate demographic parity against.",
                    "candidate_attributes": self.detect_candidate_protected_attributes(self.df)
                }
            self.target_col = num_cols[-1]

        # Step 4: Resolve group values if unspecified
        unique_groups = [str(x) for x in self.df[self.protected_col].dropna().unique()]
        if len(unique_groups) < 2:
            return {
                "has_protected_attribute": True,
                "fairness_evaluable": False,
                "status": "INSUFFICIENT_VARIATION",
                "disparate_impact_ratio": None,
                "demographic_parity_diff": None,
                "fairness_score": None,
                "is_compliant": None,
                "compliance_status": "Single Group Only",
                "explanation": f"Protected attribute '{self.protected_col}' only has 1 unique category in dataset. At least 2 groups required for disparity analysis.",
                "candidate_attributes": self.detect_candidate_protected_attributes(self.df)
            }

        if self.privileged_group is None or str(self.privileged_group) not in unique_groups:
            self.privileged_group = unique_groups[0]
        if self.unprivileged_group is None or str(self.unprivileged_group) not in unique_groups:
            self.unprivileged_group = unique_groups[1] if len(unique_groups) > 1 else unique_groups[0]

        # Step 5: Filter groups and calculate positive selection rates
        prot_str = self.df[self.protected_col].astype(str)
        priv_mask = prot_str == str(self.privileged_group)
        unpriv_mask = prot_str == str(self.unprivileged_group)

        priv_df = self.df[priv_mask]
        unpriv_df = self.df[unpriv_mask]

        total_priv = len(priv_df)
        total_unpriv = len(unpriv_df)

        if total_priv == 0 or total_unpriv == 0:
            return {
                "has_protected_attribute": True,
                "fairness_evaluable": False,
                "status": "EMPTY_GROUP",
                "disparate_impact_ratio": None,
                "demographic_parity_diff": None,
                "fairness_score": None,
                "is_compliant": None,
                "compliance_status": "Empty Demographic Group",
                "explanation": "One or both demographic groups contain 0 sample records.",
                "candidate_attributes": self.detect_candidate_protected_attributes(self.df)
            }

        # Determine positive outcome logic
        target_series = self.df[self.target_col]
        if pd.api.types.is_numeric_dtype(target_series):
            pos_val = self.positive_outcome
            # If target has 0 and 1, ensure 1 is positive outcome
            uniq_target = set(target_series.dropna().unique())
            if uniq_target == {0, 1} or uniq_target == {0.0, 1.0}:
                pos_val = 1
            priv_pos = int((priv_df[self.target_col] == pos_val).sum())
            unpriv_pos = int((unpriv_df[self.target_col] == pos_val).sum())
        else:
            # Categorical target - take most frequent or specified
            mode_val = target_series.mode()[0] if not target_series.mode().empty else None
            pos_val = self.positive_outcome if self.positive_outcome in target_series.values else mode_val
            priv_pos = int((priv_df[self.target_col].astype(str) == str(pos_val)).sum())
            unpriv_pos = int((unpriv_df[self.target_col].astype(str) == str(pos_val)).sum())

        priv_rate = priv_pos / total_priv if total_priv > 0 else 0.0
        unpriv_rate = unpriv_pos / total_unpriv if total_unpriv > 0 else 0.0

        # Disparate Impact Ratio: unprivileged_rate / privileged_rate
        if priv_rate > 0:
            disparate_impact = round(unpriv_rate / priv_rate, 4)
        elif unpriv_rate == 0 and priv_rate == 0:
            disparate_impact = 1.0
        else:
            disparate_impact = 0.0

        # Demographic Parity Difference: |priv_rate - unpriv_rate|
        demographic_parity_diff = round(abs(priv_rate - unpriv_rate), 4)

        # EEOC 80% Rule (0.80 <= DI <= 1.25)
        is_compliant = (0.80 <= disparate_impact <= 1.25)

        # Fairness Score (0 to 100): 100 is perfect parity (DI = 1.0)
        # Scaled penalty for deviation from 1.0
        di_deviation = abs(1.0 - disparate_impact)
        fairness_score = round(max(0.0, min(100.0, 100.0 - (di_deviation * 60.0) - (demographic_parity_diff * 40.0))), 1)

        # Metric explanation
        if is_compliant:
            explanation = (
                f"Passes EEOC 4/5ths (80%) Rule with Disparate Impact Ratio of {disparate_impact:.2f} "
                f"between '{self.unprivileged_group}' ({unpriv_rate*100:.1f}%) and "
                f"'{self.privileged_group}' ({priv_rate*100:.1f}%)."
            )
        else:
            explanation = (
                f"Alert: Potential demographic disparity detected. Disparate Impact Ratio is {disparate_impact:.2f} "
                f"(outside regulatory [0.80, 1.25] threshold) between '{self.unprivileged_group}' "
                f"({unpriv_rate*100:.1f}%) and '{self.privileged_group}' ({priv_rate*100:.1f}%)."
            )

        return {
            "has_protected_attribute": True,
            "fairness_evaluable": True,
            "status": "EVALUATED",
            "protected_attribute": self.protected_col,
            "target_outcome": self.target_col,
            "positive_outcome_label": str(pos_val),
            "privileged_group": str(self.privileged_group),
            "unprivileged_group": str(self.unprivileged_group),
            "privileged_sample_count": total_priv,
            "unprivileged_sample_count": total_unpriv,
            "privileged_positive_rate": round(priv_rate, 4),
            "unprivileged_positive_rate": round(unpriv_rate, 4),
            "disparate_impact_ratio": disparate_impact,
            "demographic_parity_diff": demographic_parity_diff,
            "fairness_score": fairness_score,
            "is_compliant": is_compliant,
            "compliance_status": "COMPLIANT (EEOC 80% Rule Met)" if is_compliant else "DISPARATE IMPACT ALERT (EEOC Violation Risk)",
            "explanation": explanation,
            "candidate_attributes": self.detect_candidate_protected_attributes(self.df)
        }


# Standalone Demonstration Execution
if __name__ == "__main__":
    print("=" * 65)
    print("      AEGIS AI — BIAS & FAIRNESS AUDITOR MODULE DEMO")
    print("=" * 65)

    # 1. Test Banking dataset with Gender
    print("\n[1] Testing Dataset WITH Protected Attribute ('Gender'):")
    bank_df = pd.DataFrame({
        "Gender": ["Male"] * 100 + ["Female"] * 100,
        "Income": [50000] * 200,
        "Loan_Approved": [1] * 70 + [0] * 30 + [1] * 45 + [0] * 55
    })
    det1 = BiasDetector(bank_df, target_col="Loan_Approved")
    res1 = det1.evaluate_bias()
    print(f" -> Protected Attribute: {res1['protected_attribute']}")
    print(f" -> Disparate Impact   : {res1['disparate_impact_ratio']}")
    print(f" -> Demographic Parity : {res1['demographic_parity_diff']}")
    print(f" -> Fairness Score     : {res1['fairness_score']}/100")
    print(f" -> Compliance Status  : {res1['compliance_status']}")
    print(f" -> Explanation        : {res1['explanation']}")

    # 2. Test Manufacturing dataset WITHOUT Protected Attribute
    print("\n[2] Testing Dataset WITHOUT Protected Attribute (Manufacturing Sensors):")
    mfg_df = pd.DataFrame({
        "Sensor_Temp": [65.2, 70.1, 72.4, 68.0],
        "Vibration": [0.42, 0.55, 0.49, 0.40],
        "Machine_Failure": [0, 0, 1, 0]
    })
    det2 = BiasDetector(mfg_df, target_col="Machine_Failure")
    res2 = det2.evaluate_bias()
    print(f" -> Fairness Evaluable : {res2['fairness_evaluable']}")
    print(f" -> Fairness Score     : {res2['fairness_score']}")
    print(f" -> Explanation        : {res2['explanation']}")
    print("=" * 65)
