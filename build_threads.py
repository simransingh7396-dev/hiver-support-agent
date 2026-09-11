import pandas as pd
import json

DATA_PATH = r"C:\Users\simra\OneDrive\Desktop\archive\twcs\twcs.csv"
BRAND = "AskPlayStation"
OUT_PATH = "askplaystation_threads.jsonl"

print("Loading data...")
df = pd.read_csv(DATA_PATH)
df = df.set_index("tweet_id", drop=False)


def get_first_response_id(response_tweet_id):
    """response_tweet_id can be a comma-separated list; take the first."""
    if pd.isna(response_tweet_id):
        return None
    first = str(response_tweet_id).split(",")[0].strip()
    try:
        return int(first)
    except ValueError:
        return None


def walk_backward_to_root(tweet_id, df, max_steps=15):
    """Walk up in_response_to_tweet_id until we hit the start of the conversation."""
    current = tweet_id
    steps = 0
    while steps < max_steps:
        row = df.loc[current]
        parent = row["in_response_to_tweet_id"]
        if pd.isna(parent) or parent not in df.index:
            return current
        current = parent
        steps += 1
    return current


def walk_forward_from_root(root_id, df, max_steps=15):
    """Walk down response_tweet_id from the root to reconstruct the full thread."""
    chain = []
    current = root_id
    steps = 0
    seen = set()
    while current is not None and current in df.index and current not in seen and steps < max_steps:
        seen.add(current)
        row = df.loc[current]
        chain.append(row)
        current = get_first_response_id(row["response_tweet_id"])
        steps += 1
    return chain


# Find every reply this brand ever sent
brand_replies = df[(df["author_id"] == BRAND) & (df["inbound"] == False)]
print(f"Brand replies found: {len(brand_replies)}")

roots_seen = set()
threads = []

for i, tweet_id in enumerate(brand_replies["tweet_id"]):
    if i % 2000 == 0:
        print(f"  processed {i}/{len(brand_replies)}...")

    root = walk_backward_to_root(tweet_id, df)
    if root in roots_seen:
        continue  # already reconstructed this conversation
    roots_seen.add(root)

    chain = walk_forward_from_root(root, df)
    if len(chain) < 2:
        continue  # not a real back-and-forth, skip

    thread = {
        "thread_id": str(root),
        "num_turns": len(chain),
        "turns": [
            {
                "tweet_id": int(row["tweet_id"]),
                "speaker": "customer" if row["inbound"] else "brand",
                "text": str(row["text"]),
                "created_at": str(row["created_at"]),
            }
            for row in chain
        ],
    }
    threads.append(thread)

print(f"\nTotal unique reconstructed threads: {len(threads)}")

with open(OUT_PATH, "w", encoding="utf-8") as f:
    for t in threads:
        f.write(json.dumps(t) + "\n")

print(f"Saved to {OUT_PATH}")

# Quick sanity stats
lengths = [t["num_turns"] for t in threads]
print(f"\nThread length stats:")
print(f"  min: {min(lengths)}, max: {max(lengths)}, avg: {sum(lengths)/len(lengths):.1f}")