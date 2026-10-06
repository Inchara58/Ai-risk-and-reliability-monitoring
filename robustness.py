"""
AI Risk & Reliability Platform - Robustness Alias Module
File: robustness.py
Description: Convenience wrapper pointing directly to robustness_tester.py.
"""

from robustness_tester import RobustnessTester

if __name__ == "__main__":
    import subprocess
    import sys
    print("Executing Robustness Tester Module...")
    subprocess.run([sys.executable, "robustness_tester.py"])
