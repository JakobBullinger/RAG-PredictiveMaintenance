# eval/latency_eval.py

import random
import statistics
import csv
import time
from tqdm import tqdm

# relative import from the same package
from .utils import rag_answer  

# ── 1. Define your question set ───────────────────────────────────────────────
QUESTIONS = [
    "How do I fix a heat dissipation failure?",
    "What tool‑wear limit triggers service?",
    "Torque threshold for over‑strain?",
    "Cooling system inspection steps?",
    "Recommended spindle rpm for aluminium?",
    "Lubrication schedule for axis‑X?",
    "Reset after vibration alarm?",
    "Which sensor checks tool imbalance?",
    "Process‑temperature tolerance?",
    "How to recalibrate the encoder?"
]

# ── 2. Configuration ─────────────────────────────────────────────────────────
N_RUNS     = 40      # number of timed queries
THROTTLE_S = 3       # seconds to wait between calls

# ── 3. Warm‑up call (not measured) ───────────────────────────────────────────
_ = rag_answer(random.choice(QUESTIONS))

# ── 4. Timed loop ────────────────────────────────────────────────────────────
latencies = []
for _ in tqdm(range(N_RUNS), desc="Timing requests"):
    q = random.choice(QUESTIONS)
    try:
        _, _, t = rag_answer(q)    # returns (answer, sources, elapsed_sec)
        latencies.append(t)
    except Exception as e:
        print(f"\nError during RAG call: {e}")
        break
    time.sleep(THROTTLE_S)

# ── 5. Compute stats ─────────────────────────────────────────────────────────
mean_latency = statistics.mean(latencies)
p95_latency  = statistics.quantiles(latencies, n=20)[18]  # 95th percentile

print(f"\nLatency  mean={mean_latency:.2f}s   p95={p95_latency:.2f}s   (n={len(latencies)})")

# ── 6. Save raw data for appendix ────────────────────────────────────────────
with open("eval/latency_raw.csv", "w", newline="") as fh:
    writer = csv.writer(fh)
    for x in latencies:
        writer.writerow([x])



# import pandas as pd
# import matplotlib.pyplot as plt

# # 1. Load the raw timings saved by latency_eval.py
# lat = pd.read_csv("eval/latency_raw.csv", header=None)[0]

# # 2. Calculate key stats
# mean_latency = lat.mean()
# p95_latency  = lat.quantile(0.95)

# # 3. Plot
# plt.figure(figsize=(6, 4))
# plt.hist(lat, bins=20, edgecolor="black")        # 20 bins ≈ 0.1 s each
# plt.axvline(mean_latency,  linestyle="--", linewidth=1.5, label=f"mean = {mean_latency:.2f}s")
# plt.axvline(p95_latency,   linestyle="--", linewidth=1.5, label=f"p95  = {p95_latency:.2f}s")
# plt.axvline(2.0,           linestyle=":",  linewidth=1.2, label="2 s usability target")

# plt.xlabel("End‑to‑end latency (seconds)")
# plt.ylabel("Number of queries (n = 100)")
# plt.title("Latency distribution of chatbot responses")
# plt.legend()
# plt.tight_layout()
# plt.savefig("fig_latency_hist.png", dpi=300)
# plt.show()
