"""
AI Risk & Reliability Platform - Transparent Risk & Reliability Engine (Stage 4)
File: risk_calculator.py
Description: Evaluates enterprise AI risk scores, Trust Index, and Pre-Deployment Readiness.
             Fully transparent rule-based composite scoring engine with component-level weights,
             mathematical breakdowns, and regulatory risk categorizations.
             NOTE: This is a deterministic rule-based assessment engine, NOT an ML prediction.
"""

from typing import Dict, Any, Optional, List


class RiskCalculator:
    """
    Transparent rule-based AI Governance & Pre-Deployment Risk Evaluation Engine.
    Evaluates empirical evidence across 5 core pillars:
      1. Data Quality & Completeness (30% base weight)
      2. Demographic Fairness & Parity (20% base weight)
      3. Statistical Drift & Stability (20% base weight)
      4. Adversarial Robustness & Noise Resilience (15% base weight)
      5. Model Performance & Generalization (15% base weight)
    """

    def __init__(self, 
                 quality_metrics: Dict[str, Any], 
                 fairness_metrics: Optional[Dict[str, Any]] = None, 
                 drift_metrics: Optional[Dict[str, Any]] = None,
                 robustness_metrics: Optional[Dict[str, Any]] = None,
                 model_metrics: Optional[Dict[str, Any]] = None):
        self.quality = quality_metrics or {}
        self.fairness = fairness_metrics or {}
        self.drift = drift_metrics or {}
        self.robustness = robustness_metrics or {}
        self.model = model_metrics or {}

    def calculate_all(self) -> Dict[str, Any]:
        """
        Calculates composite Trust Score, Component Contributions, Pre-Deployment Readiness Score,
        and Enterprise Risk Level with complete mathematical transparency.
        """
        breakdown = []
        active_weights = {}

        # -------------------------------------------------------------
        # 1. Quality Pillar (Base weight: 30%)
        # -------------------------------------------------------------
        completeness = float(self.quality.get("completeness_score", 100.0))
        missing_pct = float(self.quality.get("missing_pct", 0.0))
        duplicate_pct = float(self.quality.get("duplicate_pct", 0.0))
        outlier_pct = float(self.quality.get("outlier_pct", 0.0))
        constant_cols_count = len(self.quality.get("constant_columns", []))

        # Explicit quality formula
        quality_score = max(0.0, min(100.0, completeness - (missing_pct * 0.5) - (duplicate_pct * 1.0) - (constant_cols_count * 2.0)))
        quality_score = round(quality_score, 1)

        breakdown.append({
            "pillar": "Data Quality & Completeness",
            "base_weight_pct": 30.0,
            "raw_score": quality_score,
            "formula": "Completeness (100) - (Missing% * 0.5) - (Duplicate% * 1.0) - (ConstantCols * 2.0)",
            "key_factors": f"Missing: {missing_pct}%, Duplicates: {duplicate_pct}%, Constant Columns: {constant_cols_count}",
            "status": "PASS" if quality_score >= 80 else ("WARNING" if quality_score >= 60 else "FAIL")
        })

        # -------------------------------------------------------------
        # 2. Fairness Pillar (Base weight: 20%)
        # -------------------------------------------------------------
        fairness_evaluable = bool(self.fairness.get("fairness_evaluable", False))
        if fairness_evaluable:
            fairness_score = float(self.fairness.get("fairness_score", 90.0))
            di_ratio = self.fairness.get("disparate_impact_ratio", 1.0)
            fairness_status = "PASS" if fairness_score >= 80 else ("WARNING" if fairness_score >= 65 else "FAIL")
            fairness_factors = f"Disparate Impact: {di_ratio}, Attribute: '{self.fairness.get('protected_attribute', 'N/A')}'"
            fairness_formula = "EEOC 4/5ths Rule Parity Function (100 - |1 - DI| * 60 - ParityDiff * 40)"
        else:
            # When fairness is not evaluable (no demographic columns), we assign neutral benchmark or redistribute
            fairness_score = 95.0
            fairness_status = "NEUTRAL"
            fairness_factors = "No protected demographic attributes in dataset (Neutral Baseline Applied)"
            fairness_formula = "Standard Baseline (No Demographic Feature Bias Present)"

        breakdown.append({
            "pillar": "Fairness & Demographic Parity",
            "base_weight_pct": 20.0,
            "raw_score": fairness_score,
            "formula": fairness_formula,
            "key_factors": fairness_factors,
            "status": fairness_status,
            "evaluable": fairness_evaluable
        })

        # -------------------------------------------------------------
        # 3. Drift & Population Stability Pillar (Base weight: 20%)
        # -------------------------------------------------------------
        psi_score = float(self.drift.get("overall_psi_score", self.drift.get("psi_score", 0.04)))
        drifted_count = int(self.drift.get("drifted_features_count", self.drift.get("drifted_count", 0)))
        
        # Basel standard: PSI < 0.10 => 100, 0.10-0.25 => 75, >0.25 => 40
        if psi_score < 0.10:
            drift_score = round(max(85.0, 100.0 - (psi_score * 100.0)), 1)
            drift_status = "PASS"
        elif psi_score < 0.25:
            drift_score = round(max(60.0, 85.0 - ((psi_score - 0.10) * 150.0)), 1)
            drift_status = "WARNING"
        else:
            drift_score = round(max(20.0, 60.0 - ((psi_score - 0.25) * 100.0)), 1)
            drift_status = "FAIL"

        breakdown.append({
            "pillar": "Data Drift & Population Stability",
            "base_weight_pct": 20.0,
            "raw_score": drift_score,
            "formula": "Population Stability Index (PSI) Benchmark Scaling",
            "key_factors": f"Overall PSI: {psi_score:.4f}, Drifted Features: {drifted_count}",
            "status": drift_status
        })

        # -------------------------------------------------------------
        # 4. Adversarial Robustness Pillar (Base weight: 15%)
        # -------------------------------------------------------------
        model_available = bool(self.robustness.get("model_available", False)) or bool(self.model.get("model_available", False))
        if model_available and "robustness_score" in self.robustness and self.robustness["robustness_score"] is not None:
            robustness_score = float(self.robustness["robustness_score"])
            rob_status = "PASS" if robustness_score >= 80 else ("WARNING" if robustness_score >= 65 else "FAIL")
            rob_factors = f"Degradation under perturbation: {self.robustness.get('avg_degradation_pct', 0.0)}%"
            rob_formula = "Empirical Noise & Masking Stress Retention Score"
        else:
            robustness_score = 90.0
            rob_status = "NEUTRAL"
            rob_factors = "Pre-training Data Stability Baseline (Model stress tests pending)"
            rob_formula = "Default Pre-Training Robustness Baseline"

        breakdown.append({
            "pillar": "Model Robustness & Resilience",
            "base_weight_pct": 15.0,
            "raw_score": robustness_score,
            "formula": rob_formula,
            "key_factors": rob_factors,
            "status": rob_status,
            "evaluable": model_available
        })

        # -------------------------------------------------------------
        # 5. Model Performance Pillar (Base weight: 15%)
        # -------------------------------------------------------------
        if model_available and self.model and ("accuracy" in self.model or "r2_score" in self.model or "f1_score" in self.model or "metrics" in self.model):
            metrics_dict = self.model.get("metrics", self.model)
            if "accuracy" in metrics_dict:
                perf_score = round(float(metrics_dict["accuracy"]) * 100.0, 1)
                perf_factors = f"Accuracy: {metrics_dict['accuracy']*100:.1f}%, F1: {metrics_dict.get('f1_score', 0)*100:.1f}%"
            elif "r2_score" in metrics_dict:
                perf_score = round(max(0.0, float(metrics_dict["r2_score"])) * 100.0, 1)
                perf_factors = f"R²: {metrics_dict['r2_score']:.3f}, RMSE: {metrics_dict.get('rmse', 0):.2f}"
            else:
                perf_score = 88.0
                perf_factors = "Trained Model Standard Validation Metrics"
            perf_status = "PASS" if perf_score >= 80 else ("WARNING" if perf_score >= 65 else "FAIL")
            perf_formula = "Validation Holdout Metric Score (0-100)"
        else:
            perf_score = 90.0
            perf_status = "NEUTRAL"
            perf_factors = "Data-phase Baseline (Train a model in Model Factory to replace with empirical metrics)"
            perf_formula = "Default Pre-Training Performance Baseline"

        breakdown.append({
            "pillar": "Model Generalization Performance",
            "base_weight_pct": 15.0,
            "raw_score": perf_score,
            "formula": perf_formula,
            "key_factors": perf_factors,
            "status": perf_status,
            "evaluable": model_available
        })

        # -------------------------------------------------------------
        # Transparent Weighted Composite Trust & Risk Calculation
        # Weights: Quality (30%) + Fairness (20%) + Drift (20%) + Robustness (15%) + Performance (15%) = 100%
        # -------------------------------------------------------------
        w_quality = 0.30
        w_fairness = 0.20
        w_drift = 0.20
        w_robustness = 0.15
        w_performance = 0.15

        weighted_trust = (
            (quality_score * w_quality) +
            (fairness_score * w_fairness) +
            (drift_score * w_drift) +
            (robustness_score * w_robustness) +
            (perf_score * w_performance)
        )
        trust_score = round(max(0.0, min(100.0, weighted_trust)), 1)

        # Pre-Deployment Reliability Score
        reliability_score = round(max(0.0, min(100.0, (quality_score * 0.4) + (drift_score * 0.3) + (robustness_score * 0.3))), 1)

        # Pre-Deployment Readiness Score
        readiness_score = round(max(0.0, min(100.0, (trust_score * 0.6) + (reliability_score * 0.4))), 1)

        # Standardized Risk Classification Mapping
        if trust_score >= 85.0:
            risk_level = "LOW"
            deployment_recommendation = "APPROVED FOR DEPLOYMENT"
            deployment_verdict = "READY"
        elif trust_score >= 70.0:
            risk_level = "MEDIUM"
            deployment_recommendation = "CONDITIONAL APPROVAL — Remediation Recommended"
            deployment_verdict = "CONDITIONAL"
        elif trust_score >= 50.0:
            risk_level = "HIGH"
            deployment_recommendation = "DEPLOYMENT BLOCKED — Mandatory Remediation Required"
            deployment_verdict = "BLOCKED"
        else:
            risk_level = "CRITICAL"
            deployment_recommendation = "CRITICAL VIOLATION — Immediate Governance Intervention"
            deployment_verdict = "CRITICAL_BLOCKED"

        # Contributions for UI chart
        contributions = {
            "quality_contribution": round(quality_score * w_quality, 1),
            "fairness_contribution": round(fairness_score * w_fairness, 1),
            "drift_contribution": round(drift_score * w_drift, 1),
            "robustness_contribution": round(robustness_score * w_robustness, 1),
            "performance_contribution": round(perf_score * w_performance, 1)
        }

        # Clear Governance Explanation
        methodology_note = (
            "Rule-Based Governance Methodology: Composite score is calculated via deterministic weighted rule evaluation "
            "(30% Data Quality + 20% Demographic Fairness + 20% Population Drift + 15% Model Robustness + 15% Generalization). "
            "This is a deterministic analytical audit score, NOT a machine learning prediction."
        )

        return {
            "trust_score": trust_score,
            "reliability_score": reliability_score,
            "readiness_score": readiness_score,
            "risk_level": risk_level,
            "deployment_verdict": deployment_verdict,
            "deployment_recommendation": deployment_recommendation,
            "is_rule_based": True,
            "methodology_note": methodology_note,
            "weights": {
                "quality": int(w_quality * 100),
                "fairness": int(w_fairness * 100),
                "drift": int(w_drift * 100),
                "robustness": int(w_robustness * 100),
                "performance": int(w_performance * 100)
            },
            "contributions": contributions,
            "pillar_breakdown": breakdown
        }


# Standalone Demonstration Execution
if __name__ == "__main__":
    print("=" * 65)
    print("      AEGIS AI — TRANSPARENT RISK ENGINE DEMO")
    print("=" * 65)

    calc = RiskCalculator(
        quality_metrics={"completeness_score": 94.5, "missing_pct": 2.1, "duplicate_pct": 0.0, "outlier_pct": 1.2, "constant_columns": []},
        fairness_metrics={"fairness_evaluable": True, "fairness_score": 88.0, "disparate_impact_ratio": 0.85, "protected_attribute": "Gender"},
        drift_metrics={"overall_psi_score": 0.05, "drifted_features_count": 0},
        robustness_metrics={"model_available": True, "robustness_score": 91.5, "avg_degradation_pct": 4.2},
        model_metrics={"model_available": True, "metrics": {"accuracy": 0.935, "f1_score": 0.92}}
    )

    res = calc.calculate_all()

    print(f"Trust Score       : {res['trust_score']} / 100")
    print(f"Reliability Score : {res['reliability_score']} / 100")
    print(f"Readiness Score   : {res['readiness_score']} / 100")
    print(f"Risk Level        : {res['risk_level']} -> [{res['deployment_verdict']}]")
    print(f"Verdict           : {res['deployment_recommendation']}")
    print(f"\nPillar Breakdown:")
    for p in res['pillar_breakdown']:
        print(f" - {p['pillar']:<32} (Weight {p['base_weight_pct']}%): Raw Score = {p['raw_score']}/100 [{p['status']}]")
    print(f"\nMethodology: {res['methodology_note']}")
    print("=" * 65)
