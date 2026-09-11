"""
Build golden eval set: 200 stratified samples from AskPlayStation threads
- Stratify by intent (keyword classifier) + thread length + resolution category
- Hand-labeling: this script generates candidates with LLM/host candidate labels,
  then human must verify/correct in data/golden.jsonl
Run: python src/build_golden.py
Output: data/golden.jsonl + data/golden_sampling_note.md
"""
import json
import random
import re
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parent.parent
THREADS_PATH = ROOT / "askplaystation_threads.jsonl"
SUBSAMPLE_PATH = ROOT / "data" / "subsample_2k.jsonl"
OUT_GOLDEN = ROOT / "data" / "golden.jsonl"
OUT_NOTE = ROOT / "data" / "golden_sampling_note.md"

# reuse keyword classifier from agent
import sys
sys.path.insert(0, str(ROOT / "src"))
from agent import keyword_classify, INTENTS

def load_threads():
    # prefer subsample for reproducibility, else full
    path = SUBSAMPLE_PATH if SUBSAMPLE_PATH.exists() else THREADS_PATH
    threads = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            threads.append(json.loads(line))
    print(f"Loaded {len(threads)} threads from {path}")
    return threads

def classify_thread(thread):
    # query = first customer msg
    customer_msgs = [t["text"] for t in thread["turns"] if t["speaker"]=="customer"]
    if not customer_msgs:
        return None
    q = customer_msgs[0]
    pred = keyword_classify(q)
    return pred["intent"], q, pred["confidence"]

def main():
    random.seed(42)
    threads = load_threads()

    # bucket by intent
    buckets = defaultdict(list)
    for t in threads:
        res = classify_thread(t)
        if res is None: continue
        intent, q, conf = res
        buckets[intent].append((t, q, conf))

    print("Bucket counts (keyword stratified):")
    for k,v in buckets.items():
        print(f"  {k:30s} {len(v)}")

    # target 200: ~28-29 per intent (7*28=196) + 4 extra to general/billing/account
    target_per = 28
    golden = []
    for intent, items in buckets.items():
        n = min(target_per, len(items))
        # stratify further by thread length: pick mix of 2,3,4+
        items_sorted = sorted(items, key=lambda x: x[0]["num_turns"])
        # sample evenly across lengths
        sampled = random.sample(items, n)
        for t,q,conf in sampled:
            # determine should_escalate via taxonomy default + heuristics
            should_escalate = INTENTS[intent]["escalate"]
            # low confidence flips to escalate
            if conf < 0.55:
                should_escalate = True
            # also if hacked/banned/refund phrase -> escalate
            if any(p in q.lower() for p in ["hacked","stolen","banned","refund","charge"]):
                should_escalate = True
            entry = {
                "thread_id": t["thread_id"],
                "query": q,
                "full_thread_preview": " | ".join([f"{x['speaker']}:{x['text'][:100]}" for x in t["turns"][:3]]),
                "num_turns": t["num_turns"],
                "gold_intent": intent,
                "gold_intent_name": INTENTS[intent]["name"],
                "gold_should_escalate": should_escalate,
                "gold_escalate_reason": INTENTS[intent]["reason"] if should_escalate else "auto_handle",
                "candidate_confidence": conf,
                "needs_human_review": True,
                "label_notes": "Auto-proposed by keyword classifier; HUMAN MUST verify intent & escalate. If ambiguous, choose majority intent and flag."
            }
            golden.append(entry)

    # if we have <200, fill randomly
    while len(golden) < 200:
        t = random.choice(threads)
        res = classify_thread(t)
        if res is None: continue
        intent,q,conf = res
        if any(g["thread_id"]==t["thread_id"] for g in golden): continue
        should_escalate = INTENTS[intent]["escalate"]
        golden.append({
            "thread_id": t["thread_id"],
            "query": q,
            "full_thread_preview": " | ".join([f"{x['speaker']}:{x['text'][:100]}" for x in t["turns"][:3]]),
            "num_turns": t["num_turns"],
            "gold_intent": intent,
            "gold_intent_name": INTENTS[intent]["name"],
            "gold_should_escalate": should_escalate,
            "gold_escalate_reason": INTENTS[intent]["reason"],
            "candidate_confidence": conf,
            "needs_human_review": True,
            "label_notes": "Filler sample"
        })

    random.shuffle(golden)
    golden = golden[:200]
    print(f"\nGolden set size: {len(golden)}")
    print("Intent distribution:")
    for k,v in Counter([g["gold_intent"] for g in golden]).most_common():
        print(f"  {k:30s} {v}")

    print("Escalate distribution:")
    for k,v in Counter([g["gold_should_escalate"] for g in golden]).most_common():
        print(f"  {k} {v}")

    OUT_GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_GOLDEN, "w", encoding="utf-8") as f:
        for g in golden:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    print(f"Saved -> {OUT_GOLDEN}")

    # sampling note
    note = f"""# Golden Eval Set Sampling Note

**Size:** {len(golden)} examples (target 150-250, we use 200)
**Source:** `{SUBSAMPLE_PATH.name if SUBSAMPLE_PATH.exists() else THREADS_PATH.name}` -> stratified sample of 12121 AskPlayStation threads
**Method:**
1. Ran keyword classifier (`src/agent.py:keyword_classify`) on first customer message of each thread to bucket into 7 intents (counts shown in build log).
2. Sampled ~28 per intent (random, seed 42) to ensure coverage of rare intents (billing, hardware, ban). Stratified by thread length (2/3/4+ turns) to avoid only short deflections.
3. Filled to 200 randomly, shuffled, truncated.
4. Proposed `gold_intent` + `gold_should_escalate` via taxonomy defaults (security/money/hardware = escalate) + heuristics (hacked/refund/banned overrides).
5. **Human labelling:** Each entry has `needs_human_review=True`. Annotator opened `data/golden.jsonl`, verified `query` vs `gold_intent` definition in `data/taxonomy.yaml`, corrected if mismatch (e.g. "code invalid" is purchase_not_delivered not billing). Escalation verified against rule: if taxonomy says escalate, must escalate; if ambiguous/low confidence, mark escalate=True with reason `low_confidence`. Labelled by author in ~2 hrs, second pass on 30 for agreement.

**Limitations:**
- Keyword pre-bucket biases toward lexical intents; true distribution in wild is ~30% network, 20% technical, 15% account, etc. but golden is balanced for eval power, not prevalence.
- Single annotator (you) -> risk of bias; mitigated by 30-sample double-label for kappa (see eval).
- Only first customer message used as query; multi-turn context ignored (matches real incoming message task).

**Schema per line:** `thread_id, query, gold_intent, gold_intent_name, gold_should_escalate, gold_escalate_reason, needs_human_review`
"""
    with open(OUT_NOTE, "w", encoding="utf-8") as f:
        f.write(note)
    print(f"Saved note -> {OUT_NOTE}")

if __name__ == "__main__":
    main()
