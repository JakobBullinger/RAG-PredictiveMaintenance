import json, csv
from utils import rag_answer

with open("data/gold_set.json") as fh:
    GOLD = json.load(fh)

hits, log = 0, []
for item in GOLD:
    _, sources, _ = rag_answer(item["question"])
    hit = any(item["must_have"].split("#")[0] in s for s in sources[:3])
    hits += hit
    log.append({"q": item["question"], "hit": hit, "sources": sources})

hit_rate = hits / len(GOLD)
print(f"Top‑3 Hit‑Rate  {hits}/{len(GOLD)}  =  {hit_rate:.0%}")

with open("eval/retrieval_log.csv", "w", newline="") as fh:
    wr = csv.writer(fh); wr.writerow(["question", "hit", "sources"])
    for row in log:
        wr.writerow([row["q"], row["hit"], "; ".join(row["sources"])])
