import csv
from collections import defaultdict

path = "/Users/vyshakathreya/Documents/subliminal_learning_Vyshak/outputs/real_mgnify_hierarchical/samples_summary.csv"

def get_cat(sid):
    return sid.split("__")[0]

rows = []
with open(path, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

total = len(rows)
hits = 0
cat_stats = defaultdict(lambda: {"hits": 0, "total": 0, "ppl": 0.0, "gain": 0.0})

for r in rows:
    cat = get_cat(r["sample_id"])
    nn_cat = get_cat(r["nn1_latent"])
    ppl = float(r["ppl_adapt"])
    gain = float(r["info_gain"])
    
    cat_stats[cat]["total"] += 1
    cat_stats[cat]["ppl"] += ppl
    cat_stats[cat]["gain"] += gain
    
    if cat == nn_cat:
        hits += 1
        cat_stats[cat]["hits"] += 1

print(f"Overall Category Clustering Accuracy: {hits/total:.2%}")
print("-" * 40)
for cat, s in sorted(cat_stats.items()):
    acc = s["hits"] / s["total"]
    m_ppl = s["ppl"] / s["total"]
    m_gain = s["gain"] / s["total"]
    print(f"{cat:12s}: Accuracy={acc:6.2%} | Avg PPL={m_ppl:10.2f} | Avg Gain={m_gain:8.4f}")
