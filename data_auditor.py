"""
AEGIS AI — Master Data Integrity Auditor Engine
File: data_auditor.py
Description: Performs comprehensive structural data validation, missing value profiling,
             IQR outlier detection, duplicate row analysis, constant column flagging,
             cardinality checking, memory profiling, and data completeness scoring.
             All calculations are strictly empirical based on actual input datasets.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Union, Optional


class DataAuditor:
    """
    Automated dataset integrity auditor evaluating structural completeness,
    data type distributions, outlier profiles, duplicate patterns, and anomaly flags.
    """

    def __init__(self, data: Union[pd.DataFrame, str]):
        if isinstance(data, str):
            self.df = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            self.df = data.copy()
        else:
            raise ValueError("Input data must be a pandas DataFrame or valid CSV file path.")
        
    def audit(self) -> Dict[str, Any]:
        """
        Runs a comprehensive empirical structural audit on the loaded dataset.
        Returns a rich telemetry dictionary with complete column-level profiling.
        """
        total_rows, total_cols = self.df.shape
        total_cells = total_rows * total_cols
        
        # 1. Missing values analysis
        missing_matrix = self.df.isnull()
        missing_count = int(missing_matrix.sum().sum())
        missing_pct = round((missing_count / total_cells) * 100, 2) if total_cells > 0 else 0.0

        # 2. Duplicate rows analysis
        duplicate_mask = self.df.duplicated()
        duplicate_count = int(duplicate_mask.sum())
        duplicate_pct = round((duplicate_count / total_rows) * 100, 2) if total_rows > 0 else 0.0

        # 3. Column Profiling & Anomaly Detection
        column_stats = {}
        constant_columns = []
        id_columns = []
        high_cardinality_cols = []
        total_outliers_count = 0
        null_counts = {}

        for col in self.df.columns:
            col_series = self.df[col]
            col_missing = int(col_series.isnull().sum())
            null_counts[col] = col_missing
            col_missing_pct = round((col_missing / total_rows) * 100, 2) if total_rows > 0 else 0.0
            unique_vals = int(col_series.nunique(dropna=True))
            dtype_str = str(col_series.dtype)

            # Check for constant / single-value columns
            if unique_vals <= 1 and total_rows > 1:
                constant_columns.append(col)

            # Check for ID columns
            col_lower = str(col).lower()
            if col_lower.endswith("_id") or col_lower.startswith("id_") or col_lower in ["id", "uuid", "guid", "account_id", "order_id", "machine_id", "patient_id", "shipment_id"]:
                id_columns.append(col)
            elif dtype_str == "object" and unique_vals > (total_rows * 0.95) and total_rows > 50:
                id_columns.append(col)

            # Check for high-cardinality categorical columns
            if dtype_str == "object" and unique_vals > 50 and unique_vals < (total_rows * 0.95):
                high_cardinality_cols.append(col)

            stats = {
                "dtype": dtype_str,
                "missing_count": col_missing,
                "missing_pct": col_missing_pct,
                "unique_values": unique_vals,
                "is_constant": unique_vals <= 1,
                "is_id_column": col in id_columns,
                "outlier_count": 0,
                "outlier_pct": 0.0
            }

            # Numerical Statistical Profiling & 1.5x IQR Outlier Detection
            if pd.api.types.is_numeric_dtype(col_series):
                valid_num = col_series.dropna()
                if len(valid_num) > 0:
                    q25 = float(valid_num.quantile(0.25))
                    q75 = float(valid_num.quantile(0.75))
                    iqr = q75 - q25
                    lower_bound = q25 - (1.5 * iqr)
                    upper_bound = q75 + (1.5 * iqr)

                    outliers_mask = (valid_num < lower_bound) | (valid_num > upper_bound)
                    col_outliers = int(outliers_mask.sum())
                    total_outliers_count += col_outliers

                    stats.update({
                        "mean": round(float(valid_num.mean()), 3),
                        "std": round(float(valid_num.std()), 3) if len(valid_num) > 1 else 0.0,
                        "min": round(float(valid_num.min()), 3),
                        "max": round(float(valid_num.max()), 3),
                        "median": round(float(valid_num.median()), 3),
                        "q25": round(q25, 3),
                        "q75": round(q75, 3),
                        "iqr": round(iqr, 3),
                        "lower_bound": round(lower_bound, 3),
                        "upper_bound": round(upper_bound, 3),
                        "outlier_count": col_outliers,
                        "outlier_pct": round((col_outliers / len(valid_num)) * 100, 2)
                    })
            else:
                # Categorical top frequencies
                top_counts = col_series.value_counts(dropna=True).head(5).to_dict()
                stats["top_categories"] = {str(k): int(v) for k, v in top_counts.items()}

            column_stats[col] = stats

        # 4. Completeness & Integrity Score Calculation
        completeness_penalty = (missing_pct * 1.5) + (duplicate_pct * 1.2) + (min(20.0, (total_outliers_count / max(1, total_rows)) * 20.0))
        if constant_columns:
            completeness_penalty += (len(constant_columns) * 3.0)

        completeness_score = max(0.0, min(100.0, round(100.0 - completeness_penalty, 2)))

        # 5. Memory Usage
        memory_bytes = int(self.df.memory_usage(deep=True).sum())
        memory_mb = round(memory_bytes / (1024 * 1024), 2)

        return {
            "total_rows": total_rows,
            "total_cols": total_cols,
            "total_cells": total_cells,
            "missing_count": missing_count,
            "missing_pct": missing_pct,
            "null_counts": null_counts,
            "duplicate_count": duplicate_count,
            "duplicate_pct": duplicate_pct,
            "outlier_count": total_outliers_count,
            "outlier_pct": round((total_outliers_count / max(1, total_cells)) * 100, 2),
            "constant_columns": constant_columns,
            "id_columns": id_columns,
            "high_cardinality_cols": high_cardinality_cols,
            "completeness_score": completeness_score,
            "memory_usage_bytes": memory_bytes,
            "memory_mb": memory_mb,
            "columns": list(self.df.columns),
            "column_stats": column_stats
        }


# Standalone Demonstration Execution
if __name__ == "__main__":
    print("=" * 70)
    print("      AEGIS AI — DATA INTEGRITY AUDITOR MODULE")
    print("=" * 70)

    demo_df = pd.DataFrame({
        "Order_ID": [f"ORD-{i}" for i in range(100)],
        "Age": [None if i % 10 == 0 else 20 + (i * 2) % 50 for i in range(100)],
        "Income": [None if i % 15 == 0 else 30000 + (i * 1500) for i in range(100)],
        "Constant_Flag": [1] * 100,
        "Status": ["Approved" if i % 2 == 0 else "Rejected" for i in range(100)]
    })

    auditor = DataAuditor(demo_df)
    res = auditor.audit()
    print(f"Total Rows: {res['total_rows']}, Cols: {res['total_cols']}")
    print(f"Completeness Score: {res['completeness_score']}/100")
    print(f"Missing Values: {res['missing_count']} ({res['missing_pct']}%)")
    print(f"Constant Columns: {res['constant_columns']}")
    print(f"ID Columns: {res['id_columns']}")
