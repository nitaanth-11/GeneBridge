"""
GeneBridge MLOps Traffic Simulator
Sends simulated clinical prediction queries to populate Prometheus and Grafana metrics.
Run: python scripts/simulate_traffic.py [BASE_URL]
"""

import sys
import time
import random
import requests

# Default to localhost:5000 or allow passing custom URL (e.g. http://54.226.145.100:5000)
BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5000"
BASE_URL = BASE_URL.rstrip("/")

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
            print(f"[!] Connection error to {BASE_URL}: {e}")

        # Sleep briefly between queries to create continuous monitoring traffic
        time.sleep(random.uniform(0.3, 1.0))

if __name__ == "__main__":
    main()
