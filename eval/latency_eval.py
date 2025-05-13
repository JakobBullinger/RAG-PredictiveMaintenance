import random
import statistics
import csv
import time
from tqdm import tqdm

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
        _, _, t = rag_answer(q)    
        latencies.append(t)
    except Exception as e:
        print(f"\nError during RAG call: {e}")
        break
    time.sleep(THROTTLE_S)

# ── 5. Compute stats ─────────────────────────────────────────────────────────
mean_latency = statistics.mean(latencies)
p95_latency  = statistics.quantiles(latencies, n=20)[18]  

print(f"\nLatency  mean={mean_latency:.2f}s   p95={p95_latency:.2f}s   (n={len(latencies)})")

# ── 6. Save raw data for appendix ────────────────────────────────────────────
with open("eval/latency_raw.csv", "w", newline="") as fh:
    writer = csv.writer(fh)
    for x in latencies:
        writer.writerow([x])
