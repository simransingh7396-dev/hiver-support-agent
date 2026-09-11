import json
from collections import Counter

IN_PATH = "askplaystation_threads.jsonl"

# Load threads
threads = []
with open(IN_PATH, "r", encoding="utf-8") as f:
    for line in f:
        threads.append(json.loads(line))

print(f"Total threads loaded: {len(threads)}")

RESOLVED_WORDS = [
    "thank", "thanks", "thankyou", "worked", "works now", "fixed",
    "solved", "appreciate", "perfect", "awesome", "great, thanks",
    "all good", "sorted", "resolved", "yes!", "got it working"
]

UNRESOLVED_WORDS = [
    "still not", "still doesn't", "still isn't", "not working",
    "doesn't work", "didn't work", "not fixed", "useless", "worse",
    "waste of time", "ridiculous", "still broken", "no help",
    "not helpful", "still having", "still get"
]


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
        return "ends_on_brand_reply"


categories = Counter()
examples = {}

for thread in threads:
    cat = classify_ending(thread)
    categories[cat] += 1
    examples.setdefault(cat, []).append(thread)

print("\n" + "=" * 60)
print("THREAD ENDING CATEGORIES")
print("=" * 60)
for cat, count in categories.most_common():
    pct = 100 * count / len(threads)
    print(f"{cat:30s} {count:6d}  ({pct:.1f}%)")

print("\n" + "=" * 60)
print("SAMPLE THREADS PER CATEGORY (3 each)")
print("=" * 60)
for cat in categories:
    print(f"\n--- {cat} ---")
    for thread in examples[cat][:3]:
        print(f"  Thread {thread['thread_id']}:")
        for turn in thread["turns"]:
            speaker = "CUSTOMER" if turn["speaker"] == "customer" else "BRAND   "
            text = turn["text"].replace("\n", " ")
            print(f"    [{speaker}] {text}")
        print()