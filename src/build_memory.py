"""
Build resolved memory from askplaystation_threads.jsonl
- Uses classify_resolution heuristic + brand fix signals
- Saves data/resolved_memory.jsonl and data/subsample_2k.jsonl
Run: python src/build_memory.py
"""
import json
import re
import random
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
IN_PATH = ROOT / "askplaystation_threads.jsonl"
OUT_RESOLVED = ROOT / "data" / "resolved_memory.jsonl"
OUT_SUBSAMPLE = ROOT / "data" / "subsample_2k.jsonl"

RESOLVED_WORDS = ["thank","thanks","thankyou","worked","works now","fixed","solved","appreciate","perfect","awesome","great, thanks","all good","sorted","resolved","got it working","it worked"]
UNRESOLVED_WORDS = ["still not","still doesn't","still isn't","not working","doesn't work","didn't work","not fixed","useless","worse","waste of time","ridiculous","still broken","no help","not helpful","still having","still get"]

BRAND_FIX_SIGNALS = ["try","please try","power cycle","restore licenses","rebuild database","reset","check","follow these steps","https://","dm","private message","download queue"]

def classify_ending(thread):
    turns = thread["turns"]
    last = turns[-1]
    text = last["text"].lower()
    if last["speaker"] == "customer":
        if any(w in text for w in RESOLVED_WORDS):
            return "resolved_confirmed"
        if any(w in text for w in UNRESOLVED_WORDS):
            return "unresolved_signal"
        return "ends_on_customer_unclear"
    else:
        # brand last - check if brand gave actionable fix
        if any(s in text for s in BRAND_FIX_SIGNALS):
            return "ends_on_brand_with_fix"
        return "ends_on_brand_generic"

def load_threads():
    threads = []
    with open(IN_PATH, "r", encoding="utf-8") as f:
        for line in f:
            threads.append(json.loads(line))
    return threads

def main():
    print(f"Loading {IN_PATH}...")
    threads = load_threads()
    print(f"Total: {len(threads)}")
    cats = Counter()
    for t in threads:
        cats[classify_ending(t)] += 1
    print("Categories:")
    for k,v in cats.most_common():
        print(f"  {k:30s} {v}")

    # Build resolved memory: resolved_confirmed + ends_on_brand_with_fix (actionable)
    # We include ends_on_brand_with_fix because 81.8% ends_on_brand - many are deflections but with fix links are useful grounding
    resolved = []
    for t in threads:
        cat = classify_ending(t)
        if cat in ("resolved_confirmed", "ends_on_brand_with_fix"):
            # create memory entry: query = first customer message, resolution = brand messages concatenated
            customer_msgs = [turn["text"] for turn in t["turns"] if turn["speaker"]=="customer"]
            brand_msgs = [turn["text"] for turn in t["turns"] if turn["speaker"]=="brand"]
            if not customer_msgs or not brand_msgs:
                continue
            entry = {
                "thread_id": t["thread_id"],
                "num_turns": t["num_turns"],
                "category": cat,
                "query": customer_msgs[0],
                "customer_all": " | ".join(customer_msgs),
                "brand_resolution": " | ".join(brand_msgs),
                "full_thread": t["turns"]
            }
            resolved.append(entry)

    print(f"\nResolved memory candidates: {len(resolved)} ({len(resolved)/len(threads)*100:.1f}%)")
    # dedup by query
    seen = set()
    deduped = []
    for e in resolved:
        q = e["query"].strip().lower()[:120]
        if q not in seen:
            seen.add(q)
            deduped.append(e)
    print(f"After dedup: {len(deduped)}")

    OUT_RESOLVED.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_RESOLVED, "w", encoding="utf-8") as f:
        for e in deduped:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"Saved -> {OUT_RESOLVED}")

    # Subsample 2k for fast eval / 15-min reproduce (stratified by category)
    random.seed(42)
    # also create a 2k random subsample of ALL threads for eval sampling pool
    subsample = random.sample(threads, min(2000, len(threads)))
    with open(OUT_SUBSAMPLE, "w", encoding="utf-8") as f:
        for t in subsample:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"Saved subsample {len(subsample)} -> {OUT_SUBSAMPLE}")

    # quick stats
    lens = [e["num_turns"] for e in deduped]
    if lens:
        print(f"Memory thread lengths: min {min(lens)} max {max(lens)} avg {sum(lens)/len(lens):.1f}")

if __name__ == "__main__":
    main()
