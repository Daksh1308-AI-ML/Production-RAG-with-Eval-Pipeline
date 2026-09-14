"""Build a stratified A/B subset from eval_dataset.json: 8 pairs, mixed
company / difficulty / type, so the free-tier judge quota gives a defensible
cross-company signal instead of the first-N slice (which is all AAPL)."""

import json
from pathlib import Path

SRC = Path("data/evaluation/eval_dataset.json")
OUT = Path("data/evaluation/ab_stratified.json")
ASK = ["AAPL--easy", "AAPL--hard", "MSFT--medium", "GOOGL--hard",
       "AMZN--medium", "NVDA--hard", "NVDA--medium", "GOOGL--medium"]

data = json.loads(SRC.read_text(encoding="utf-8"))
tag = {}
for i, item in enumerate(data):
    item["_idx"] = i
    m = item["metadata"]
    key = f"{m['company']}--{m['difficulty']}"
    tag.setdefault(key, []).append(item)

subset, used = [], set()
for a in ASK:
    for item in tag[a]:
        if item["_idx"] not in used:
            subset.append(item)
            used.add(item["_idx"])
            break

assert len(subset) == len(ASK), f"wanted {len(ASK)} pairs, got {len(subset)}"
for item in subset:
    item.pop("_idx")
OUT.write_text(json.dumps(subset, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"wrote {len(subset)} pairs -> {OUT}")
for item in subset:
    m = item["metadata"]
    print(f"  {m['company']:5} {m['difficulty']:7} {m['question_type']:11} {item['question'][:60]}")