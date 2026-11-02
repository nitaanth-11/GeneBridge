"""
GeneBridge MLOps Traffic Simulator
Sends simulated clinical prediction queries to populate Prometheus and Grafana metrics.
Run: python scripts/simulate_traffic.py
"""

import time
import random
import requests

BASE_URL = "http://localhost:5000"

CONDITIONS = [
    "spinal_cord_injury",
    "muscular_dystrophy",
    "retinal_degeneration",
    "heart_disease",
    "parkinsons",
    "als"
]

SEVERITIES = ["mild", "moderate", "severe"]
GENDERS = ["male", "female", "other"]

def main():
    print(f"[*] Starting GeneBridge traffic simulator against {BASE_URL}...")
    print("[*] Press Ctrl+C to stop.")
    success_count = 0

    while True:
        condition = random.choice(CONDITIONS)
        payload = {
            "condition": condition,
            "age": random.randint(20, 75),
            "severity": random.choice(SEVERITIES),
            "gender": random.choice(GENDERS)
        }

        try:
            res = requests.post(f"{BASE_URL}/api/predict", json=payload, timeout=5)
            if res.status_code == 200:
                success_count += 1
                data = res.json()
                best = data["summary"]["best_species"]
                viability = data["summary"]["best_viability"]
                print(f"[{success_count}] Condition: {condition:22} -> Best match: {best:15} (Score: {viability:.2f})")
            else:
                print(f"[!] Request returned {res.status_code}: {res.text}")
        except Exception as e:
            print(f"[!] Connection error: {e}")

        # Sleep briefly between queries to create continuous monitoring traffic
        time.sleep(random.uniform(0.3, 1.2))

if __name__ == "__main__":
    main()

