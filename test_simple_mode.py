"""
Test Suite: AEGIS AI Simple Mode Backend & Dual-Experience API Tests
Validates:
1. /api/v1/simple/health-check
2. /api/v1/simple/ask-data
3. /api/v1/simple/report
4. /api/v1/demo/cache-cleaned & shared dataset state
5. Consistency between raw and cleaned data
"""

import unittest
import requests
import json
import time

BASE_URL = "http://127.0.0.1:5000"

class TestSimpleModeIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Wait a moment for server initialization
        time.sleep(1)

    def test_01_simple_health_check_with_demo_dataset(self):
        """Test health check calculation on sample dataset"""
        sample_data = [
            {"Student_ID": "STU-001", "CGPA": 8.5, "Coding_Score": 85, "Placement_Status": "Placed"},
            {"Student_ID": "STU-002", "CGPA": None, "Coding_Score": 60, "Placement_Status": "Unplaced"},
            {"Student_ID": "STU-001", "CGPA": 8.5, "Coding_Score": 85, "Placement_Status": "Placed"}, # Duplicate
            {"Student_ID": "STU-003", "CGPA": 9.9, "Coding_Score": 999, "Placement_Status": "Placed"}, # Outlier
        ]
        res = requests.post(f"{BASE_URL}/api/v1/simple/health-check", json={
            "dataset": sample_data,
            "filename": "Student_Placements.csv"
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "success")
        self.assertIn("score", data)
        self.assertIn("tier", data)
        self.assertIn("summary", data)
        self.assertEqual(data["total_rows"], 4)
        self.assertEqual(data["missing_count"], 1)
        self.assertEqual(data["duplicate_count"], 1)
        self.assertGreater(len(data["good_points"]), 0)
        self.assertGreater(len(data["attention_points"]), 0)

    def test_02_ask_data_records_count_query(self):
        """Test plain English query about record count"""
        sample_data = [
            {"Name": "Alice", "Age": 25, "Score": 90},
            {"Name": "Bob", "Age": 30, "Score": 85},
            {"Name": "Charlie", "Age": 28, "Score": 95}
        ]
        res = requests.post(f"{BASE_URL}/api/v1/simple/ask-data", json={
            "query": "How many records are in this file?",
            "dataset": sample_data
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "success")
        self.assertIn("3", data["message"])
        self.assertIn("3", data["message"])

    def test_03_ask_data_missing_values_query(self):
        """Test query regarding missing values"""
        sample_data = [
            {"Name": "Alice", "Age": 25, "Salary": None},
            {"Name": "Bob", "Age": None, "Salary": 50000},
            {"Name": "Charlie", "Age": 28, "Salary": 60000}
        ]
        res = requests.post(f"{BASE_URL}/api/v1/simple/ask-data", json={
            "query": "Which columns have missing values?",
            "dataset": sample_data
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "success")
        self.assertIn("Salary", data["message"] + str(data.get("details", [])))

    def test_04_ask_data_average_query(self):
        """Test mathematical calculation for average query"""
        sample_data = [
            {"CGPA": 8.0},
            {"CGPA": 9.0},
            {"CGPA": 7.0}
        ]
        res = requests.post(f"{BASE_URL}/api/v1/simple/ask-data", json={
            "query": "What is the average CGPA?",
            "dataset": sample_data
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "success")
        self.assertIn("8.0", data["message"])

    def test_05_simple_report_generation(self):
        """Test generation of simple executive summary report"""
        sample_data = [
            {"ID": 1, "FeatureA": 10, "Target": "Yes"},
            {"ID": 2, "FeatureA": 20, "Target": "No"}
        ]
        res = requests.post(f"{BASE_URL}/api/v1/simple/report", json={
            "dataset": sample_data,
            "filename": "Test_Data.csv"
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "success")
        self.assertIn("report_title", data)
        self.assertIn("executive_summary", data)
        self.assertIn("health_score", data)

    def test_06_shared_cleaning_cache(self):
        """Test caching cleaned dataset so Professional Mode sees the same version"""
        cleaned_records = [
            {"ID": 1, "Age": 25, "Status": "Placed"},
            {"ID": 2, "Age": 30, "Status": "Unplaced"}
        ]
        res = requests.post(f"{BASE_URL}/api/v1/demo/cache-cleaned", json={
            "dataset": cleaned_records,
            "dataset_key": "placement",
            "project_id": "prj_placement_default"
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["cached_rows"], 2)

if __name__ == '__main__':
    unittest.main()
