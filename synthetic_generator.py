"""
AI Risk & Reliability Platform - Synthetic Data Lab Generator Engine
File: synthetic_generator.py
Description: Generates controlled tabular synthetic datasets for ML experimentation,
             testing, and demonstration. Supports schema cloning from existing datasets
             and controlled injection of missingness, duplicates, outliers, Gaussian noise,
             class imbalance, and casing inconsistency.
             All generated data is explicitly tagged as 'SYNTHETIC'.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import random
import time


class SyntheticDataGenerator:
    """
    AEGIS Controlled Tabular Data Synthesizer.
    Enables generation from scratch (by industry domain) or from existing dataset schema.
    Applies measurable fault injections with a detailed manifest.
    """

    DOMAINS = {
        "Manufacturing": {
            "features": [
                ("Machine_ID", "id", "MCH-"),
                ("Machine_Type", "categorical", ["Heavy_Industrial", "Precision_CNC", "Rotary_Motor", "Hydraulic_Press"]),
                ("Air_Temperature_K", "numeric", (295.0, 315.0, 300.0, 4.5)),
                ("Process_Temperature_K", "numeric", (305.0, 320.0, 310.0, 3.8)),
                ("Rotational_Speed_RPM", "numeric", (1100.0, 2900.0, 1550.0, 180.0)),
                ("Torque_Nm", "numeric", (10.0, 85.0, 40.0, 12.0)),
                ("Tool_Wear_Min", "numeric", (0.0, 260.0, 105.0, 65.0)),
                ("Vibration_mm_s", "numeric", (0.5, 8.5, 2.8, 1.1)),
                ("Voltage_V", "numeric", (200.0, 245.0, 220.0, 8.5)),
                ("Current_A", "numeric", (10.0, 45.0, 25.0, 6.0)),
                ("Production_Load_Pct", "numeric", (20.0, 100.0, 75.0, 16.0)),
                ("Maintenance_Cycle_Days", "numeric", (5.0, 180.0, 45.0, 28.0))
            ],
            "target_clf": ("Machine_Failure", [0, 1]),
            "target_reg": ("Remaining_Useful_Life_Hours", (10.0, 3000.0, 1200.0, 650.0))
        },
        "E-Commerce": {
            "features": [
                ("Order_ID", "id", "ORD-"),
                ("Product_Category", "categorical", ["Apparel", "Electronics", "Home_Kitchen", "Footwear", "Beauty"]),
                ("Customer_Age", "numeric", (18.0, 75.0, 36.0, 13.0)),
                ("Customer_Tenure_Months", "numeric", (1.0, 84.0, 24.0, 18.0)),
                ("Item_Price_USD", "numeric", (8.0, 950.0, 68.0, 75.0)),
                ("Discount_Pct", "numeric", (0.0, 60.0, 15.0, 12.0)),
                ("Transit_Days", "numeric", (1.0, 14.0, 3.5, 2.1)),
                ("Customer_Rating", "numeric", (1.0, 5.0, 4.2, 0.9)),
                ("Previous_Returns", "numeric", (0.0, 12.0, 1.2, 1.8)),
                ("Fulfillment_Type", "categorical", ["Standard", "Express", "Same_Day", "Economy"]),
                ("Payment_Method", "categorical", ["Credit_Card", "Debit_Card", "Digital_Wallet", "BNPL"])
            ],
            "target_clf": ("Returned", [0, 1]),
            "target_reg": ("Net_Order_Value_USD", (5.0, 1200.0, 85.0, 90.0))
        },
        "Banking & Finance": {
            "features": [
                ("Account_ID", "id", "ACC-"),
                ("Age", "numeric", (21.0, 72.0, 42.0, 12.0)),
                ("Gender", "categorical", ["Male", "Female"]),
                ("Annual_Income", "numeric", (22000.0, 350000.0, 78000.0, 38000.0)),
                ("Credit_Score", "numeric", (480.0, 850.0, 690.0, 65.0)),
                ("Employment_Years", "numeric", (0.5, 35.0, 8.5, 6.2)),
                ("Debt_To_Income_Ratio", "numeric", (0.05, 0.75, 0.32, 0.14)),
                ("Loan_Amount", "numeric", (3000.0, 120000.0, 28000.0, 21000.0)),
                ("Interest_Rate_Pct", "numeric", (3.5, 28.0, 11.5, 4.2)),
                ("Loan_Term_Months", "categorical", ["12", "24", "36", "48", "60", "84"]),
                ("Previous_Defaults_Count", "numeric", (0.0, 5.0, 0.3, 0.8)),
                ("Credit_Utilization_Pct", "numeric", (0.02, 0.98, 0.42, 0.22))
            ],
            "target_clf": ("Default", [0, 1]),
            "target_reg": ("Loss_Given_Default_USD", (0.0, 85000.0, 12000.0, 16000.0))
        },
        "Healthcare": {
            "features": [
                ("Patient_ID", "id", "PAT-"),
                ("Age", "numeric", (18.0, 92.0, 56.0, 17.0)),
                ("Gender", "categorical", ["Male", "Female"]),
                ("Admission_Type", "categorical", ["Emergency", "Elective", "Urgent", "Trauma"]),
                ("Length_Of_Stay_Days", "numeric", (1.0, 35.0, 4.8, 3.9)),
                ("Blood_Pressure_Systolic", "numeric", (90.0, 195.0, 128.0, 18.0)),
                ("Glucose_Level_mg_dL", "numeric", (65.0, 320.0, 115.0, 38.0)),
                ("BMI", "numeric", (16.0, 48.0, 27.5, 5.4)),
                ("Medication_Count", "numeric", (1.0, 24.0, 6.0, 3.8)),
                ("Comorbidity_Score", "numeric", (0.0, 8.0, 1.8, 1.5)),
                ("Previous_Admissions", "numeric", (0.0, 10.0, 0.9, 1.4))
            ],
            "target_clf": ("Readmitted", [0, 1]),
            "target_reg": ("Total_Hospital_Cost_USD", (1200.0, 85000.0, 14500.0, 12000.0))
        },
        "Logistics": {
            "features": [
                ("Shipment_ID", "id", "SHP-"),
                ("Distance_KM", "numeric", (15.0, 3200.0, 620.0, 410.0)),
                ("Package_Weight_KG", "numeric", (0.2, 95.0, 12.5, 14.0)),
                ("Carrier", "categorical", ["SwiftLogistics", "ApexFreight", "GlobalExpress", "MetroCourier", "TransAir"]),
                ("Traffic_Level", "categorical", ["Low", "Medium", "High", "Severe"]),
                ("Weather", "categorical", ["Clear", "Rain", "Fog", "Snow", "Storm"]),
                ("Route_Type", "categorical", ["Highway", "Urban", "Rural", "Mountain"]),
                ("Warehouse_Delay_Hours", "numeric", (0.0, 42.0, 3.8, 4.5)),
                ("Fuel_Cost_USD", "numeric", (15.0, 520.0, 135.0, 68.0)),
                ("Vehicle_Age_Years", "numeric", (1.0, 18.0, 5.5, 3.6))
            ],
            "target_clf": ("Delayed", [0, 1]),
            "target_reg": ("Delivery_Days", (0.5, 14.0, 3.8, 2.1))
        },
        "General": {
            "features": [
                ("Record_ID", "id", "REC-"),
                ("Category_A", "categorical", ["Type_Alpha", "Type_Beta", "Type_Gamma", "Type_Delta"]),
                ("Feature_1", "numeric", (10.0, 100.0, 50.0, 15.0)),
                ("Feature_2", "numeric", (100.0, 1000.0, 500.0, 120.0)),
                ("Feature_3", "numeric", (0.0, 1.0, 0.5, 0.18)),
                ("Feature_4", "numeric", (5.0, 50.0, 22.0, 8.0)),
                ("Segment", "categorical", ["Tier_1", "Tier_2", "Tier_3"]),
                ("Feature_5", "numeric", (500.0, 5000.0, 2200.0, 650.0))
            ],
            "target_clf": ("Target_Class", [0, 1]),
            "target_reg": ("Target_Continuous", (10.0, 500.0, 210.0, 75.0))
        }
    }

    @classmethod
    def generate_synthetic_dataset(
        cls,
        domain: str = "Manufacturing",
        n_rows: int = 1000,
        target_type: str = "classification",
        class_imbalance_ratio: float = 0.20,
        source_df: Optional[pd.DataFrame] = None,
        custom_feature_count: Optional[int] = None,
        missing_pct: float = 0.0,
        duplicate_pct: float = 0.0,
        outlier_pct: float = 0.0,
        noise_pct: float = 0.0,
        casing_inconsistency_pct: float = 0.0,
        feature_perturbation_pct: float = 0.0,
        random_seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Main synthesizer generating either from schema clone or domain specification,
        then applying measurable fault injections.
        """
        t0 = time.time()
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)

        n_rows = max(50, min(int(n_rows), 100000))
        domain_key = domain if domain in cls.DOMAINS else "Manufacturing"
        config = cls.DOMAINS[domain_key]

        manifest = {
            "domain": domain_key,
            "target_type": target_type,
            "generated_rows": n_rows,
            "is_synthetic": True,
            "dataset_label": "SYNTHETIC",
            "injections_applied": {},
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }

        # 1. Base Synthesis
        if source_df is not None and isinstance(source_df, pd.DataFrame) and len(source_df) > 0:
            df, target_col = cls._synthesize_from_existing_schema(source_df, n_rows, target_type)
            manifest["synthesis_source"] = "Schema-Cloned from Active Project Dataset"
        else:
            df, target_col = cls._synthesize_from_domain_spec(config, n_rows, target_type, class_imbalance_ratio)
            manifest["synthesis_source"] = f"Domain Specification ({domain_key})"

        # 2. Apply Injections
        injected_records_count = 0

        # Missing values injection
        if missing_pct > 0.0:
            pct = min(float(missing_pct), 0.50)
            null_cells = 0
            cols_to_null = [c for c in df.columns if c != target_col and not c.endswith("_ID")]
            for col in cols_to_null:
                mask = np.random.rand(len(df)) < pct
                df.loc[mask, col] = np.nan
                null_cells += int(mask.sum())
            manifest["injections_applied"]["missing_values"] = {
                "target_rate_pct": round(pct * 100, 1),
                "total_null_cells": null_cells,
                "affected_columns": cols_to_null
            }

        # Duplicates injection
        if duplicate_pct > 0.0:
            dup_rate = min(float(duplicate_pct), 0.30)
            n_dups = int(len(df) * dup_rate)
            if n_dups > 0 and len(df) > n_dups:
                dup_indices = np.random.choice(len(df) - n_dups, size=n_dups, replace=False)
                dup_rows = df.iloc[dup_indices].copy()
                # Append duplicates
                df = pd.concat([df.iloc[:len(df) - n_dups], dup_rows], ignore_index=True)
                manifest["injections_applied"]["duplicates"] = {
                    "target_rate_pct": round(dup_rate * 100, 1),
                    "duplicate_rows_added": n_dups
                }

        # Outliers injection
        if outlier_pct > 0.0:
            outlier_rate = min(float(outlier_pct), 0.20)
            n_outliers = int(len(df) * outlier_rate)
            numeric_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number) and c != target_col]
            if n_outliers > 0 and numeric_cols:
                out_idx = np.random.choice(len(df), size=n_outliers, replace=False)
                for c in numeric_cols[:3]:
                    col_mean = df[c].dropna().mean()
                    col_std = df[c].dropna().std() or 1.0
                    # Inject 5x - 12x extreme standard deviation outliers
                    spikes = col_mean + (np.random.choice([-1, 1], size=n_outliers) * np.random.uniform(5.0, 12.0, size=n_outliers) * col_std)
                    df.loc[out_idx, c] = np.round(spikes, 2)
                manifest["injections_applied"]["outliers"] = {
                    "target_rate_pct": round(outlier_rate * 100, 1),
                    "outlier_rows_modified": n_outliers,
                    "columns_spiked": numeric_cols[:3]
                }

        # Gaussian Noise injection
        if noise_pct > 0.0:
            noise_intensity = min(float(noise_pct), 0.60)
            numeric_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number) and c != target_col]
            for c in numeric_cols:
                std = df[c].dropna().std() or 1.0
                noise = np.random.normal(0, std * noise_intensity, size=len(df))
                df[c] = (df[c] + noise).round(2)
            manifest["injections_applied"]["gaussian_noise"] = {
                "noise_intensity_pct": round(noise_intensity * 100, 1),
                "columns_perturbed": numeric_cols
            }

        # Category Casing Inconsistencies
        if casing_inconsistency_pct > 0.0:
            casing_rate = min(float(casing_inconsistency_pct), 0.40)
            cat_cols = [c for c in df.columns if df[c].dtype == "object" and not c.endswith("_ID") and c != target_col]
            modified_cells = 0
            for c in cat_cols:
                mask = np.random.rand(len(df)) < casing_rate
                df.loc[mask, c] = df.loc[mask, c].apply(lambda x: str(x).lower() if random.random() > 0.5 else str(x).upper())
                modified_cells += int(mask.sum())
            manifest["injections_applied"]["casing_inconsistency"] = {
                "target_rate_pct": round(casing_rate * 100, 1),
                "modified_categorical_cells": modified_cells
            }

        # Feature Perturbation (Random cross-feature swap/shuffle)
        if feature_perturbation_pct > 0.0:
            pert_rate = min(float(feature_perturbation_pct), 0.40)
            n_pert = int(len(df) * pert_rate)
            numeric_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number) and c != target_col]
            if n_pert > 0 and len(numeric_cols) >= 2:
                col_a, col_b = numeric_cols[0], numeric_cols[1]
                pert_idx = np.random.choice(len(df), size=n_pert, replace=False)
                # Swap values between column A and B for these rows
                temp_vals = df.loc[pert_idx, col_a].copy()
                df.loc[pert_idx, col_a] = df.loc[pert_idx, col_b]
                df.loc[pert_idx, col_b] = temp_vals
                manifest["injections_applied"]["feature_perturbation"] = {
                    "perturbation_rate_pct": round(pert_rate * 100, 1),
                    "swapped_pairs": f"{col_a} <-> {col_b}",
                    "affected_rows": n_pert
                }

        # Class Imbalance Skew for classification if specified
        if target_type == "classification" and class_imbalance_ratio is not None and target_col in df.columns:
            target_series = df[target_col].dropna()
            classes = list(target_series.unique())
            if len(classes) == 2:
                maj_class, min_class = classes[0], classes[1]
                ratio = max(0.01, min(float(class_imbalance_ratio), 0.50))
                # Adjust target column to match exact ratio
                n_min = int(len(df) * ratio)
                n_maj = len(df) - n_min
                new_targets = [maj_class] * n_maj + [min_class] * n_min
                np.random.shuffle(new_targets)
                df[target_col] = new_targets
                manifest["injections_applied"]["class_imbalance"] = {
                    "minority_class": str(min_class),
                    "minority_ratio_pct": round(ratio * 100, 1),
                    "majority_class": str(maj_class),
                    "majority_ratio_pct": round((1.0 - ratio) * 100, 1)
                }

        latency_ms = round((time.time() - t0) * 1000, 2)
        manifest["generation_time_ms"] = latency_ms

        # Summary telemetry
        manifest["summary"] = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "target_column": target_col,
            "target_type": target_type,
            "synthetic_disclaimer": "This dataset was synthetically synthesized in the AEGIS Synthetic Data Lab for experimentation, stress testing, and demonstration. It is NOT real-world ground truth evidence."
        }

        # Return serializable structure + preview records
        sample_records = df.head(100).replace({np.nan: None}).to_dict(orient="records")

        return {
            "status": "success",
            "manifest": manifest,
            "columns": list(df.columns),
            "total_rows": len(df),
            "target_column": target_col,
            "target_type": target_type,
            "preview_sample": sample_records,
            "dataframe": df
        }

    @classmethod
    def _synthesize_from_domain_spec(
        cls,
        config: dict,
        n: int,
        target_type: str,
        class_imbalance_ratio: float
    ) -> Tuple[pd.DataFrame, str]:
        """Generates synthetic dataset from predefined industry domain specifications."""
        data = {}
        for feat_name, feat_type, feat_spec in config["features"]:
            if feat_type == "id":
                prefix = feat_spec
                data[feat_name] = [f"{prefix}{100000 + i}" for i in range(n)]
            elif feat_type == "categorical":
                categories = feat_spec
                data[feat_name] = np.random.choice(categories, size=n)
            elif feat_type == "numeric":
                min_val, max_val, mean_val, std_val = feat_spec
                vals = np.random.normal(mean_val, std_val, size=n)
                vals = np.clip(vals, min_val, max_val).round(2)
                data[feat_name] = vals

        df = pd.DataFrame(data)

        # Generate realistic target correlated with features
        if target_type == "classification":
            target_col, classes = config["target_clf"]
            # Multi-feature logit score
            numeric_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
            z = np.zeros(n)
            for i, col in enumerate(numeric_cols[:4]):
                mean = df[col].mean()
                std = df[col].std() or 1.0
                weight = 0.6 if i % 2 == 0 else -0.5
                z += weight * ((df[col] - mean) / std)

            prob = 1.0 / (1.0 + np.exp(-z))
            ratio = max(0.05, min(class_imbalance_ratio or 0.20, 0.50))
            threshold = np.quantile(prob, 1.0 - ratio)
            df[target_col] = (prob >= threshold).astype(int)
        else:
            target_col, spec = config["target_reg"]
            min_val, max_val, mean_val, std_val = spec
            numeric_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
            target_vals = np.full(n, mean_val)
            for i, col in enumerate(numeric_cols[:4]):
                mean = df[col].mean()
                std = df[col].std() or 1.0
                target_vals += (1.2 / (i + 1)) * (df[col] - mean)

            target_vals += np.random.normal(0, std_val * 0.4, size=n)
            df[target_col] = np.clip(target_vals, min_val, max_val).round(2)

        return df, target_col

    @classmethod
    def _synthesize_from_existing_schema(
        cls,
        source_df: pd.DataFrame,
        n: int,
        target_type: str
    ) -> Tuple[pd.DataFrame, str]:
        """Clones schema and distributional properties from an existing dataset."""
        data = {}
        target_col = source_df.columns[-1]

        for col in source_df.columns:
            s = source_df[col].dropna()
            is_id = col.lower().endswith("_id") or col.lower().startswith("id_") or "id" in col.lower()
            
            if is_id:
                prefix = col.split("_")[0][:3].upper() + "-"
                data[col] = [f"{prefix}{200000 + i}" for i in range(n)]
            elif np.issubdtype(source_df[col].dtype, np.number):
                if s.nunique() <= 5:
                    # Discrete / binary
                    cats = list(s.unique())
                    probs = s.value_counts(normalize=True).values
                    data[col] = np.random.choice(cats, size=n, p=probs)
                else:
                    mean_val = float(s.mean()) if len(s) > 0 else 50.0
                    std_val = float(s.std()) if len(s) > 1 else 10.0
                    min_val = float(s.min()) if len(s) > 0 else 0.0
                    max_val = float(s.max()) if len(s) > 0 else 100.0
                    vals = np.random.normal(mean_val, std_val or 1.0, size=n)
                    data[col] = np.clip(vals, min_val, max_val).round(2)
            else:
                cats = list(s.unique()) if len(s) > 0 else ["Category_A", "Category_B"]
                counts = s.value_counts(normalize=True)
                probs = counts.values if len(counts) > 0 else None
                data[col] = np.random.choice(cats, size=n, p=probs)

        df = pd.DataFrame(data)
        return df, target_col
