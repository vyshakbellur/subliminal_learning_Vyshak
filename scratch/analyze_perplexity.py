import csv
from collections import defaultdict

path = "/Users/vyshakathreya/Documents/subliminal_learning_Vyshak/outputs/real_mgnify_hierarchical/samples_summary.csv"

# Time stats
stats = defaultdict(lambda: {"base": [], "adapt": [], "times": []})
total_time = 0.0

with open(path, "r") as f:
    reader = csv.DictReader(f)
    for r in reader:
        cat = r["sample_id"].split("__")[0]
        stats[cat]["base"].append(float(r["ppl_base"]))
        stats[cat]["adapt"].append(float(r["ppl_adapt"]))
        stats[cat]["times"].append(float(r["time_seconds"]))
        total_time += float(r["time_seconds"])

print("--- 10-Epoch Perplexity Analysis ---")
for cat, s in sorted(stats.items()):
    m_base = sum(s["base"]) / len(s["base"])
    m_adapt = sum(s["adapt"]) / len(s["adapt"])
    m_delta = m_base - m_adapt
    m_time = sum(s["times"]) / len(s["times"])
    print(f"{cat:12s}: Base PPL={m_base:8.2f} | Adapt PPL={m_adapt:8.2f} | Gain={m_delta:7.2f} | Time/sample={m_time:6.2f}s")

avg_time = total_time / sum(len(s["times"]) for s in stats.values())
print(f"\nOverall testing efficiency: ~{avg_time:.2f} seconds per sample (10 steps of adaptation).")
