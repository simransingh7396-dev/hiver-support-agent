import pandas as pd

DATA_PATH = r"C:\Users\simra\OneDrive\Desktop\archive\twcs\twcs.csv"

print("Loading data...")
df = pd.read_csv(DATA_PATH)

# Index tweets by tweet_id for fast lookup
df = df.set_index("tweet_id", drop=False)

CANDIDATE_BRANDS = ["SpotifyCares", "Delta", "AskPlayStation", "ChipotleTweets"]


def build_thread(start_tweet_id, df, max_len=6):
    """
    Walk backwards from a tweet to the start of its conversation,
    then return the thread in chronological order.
    """
    chain = []
    current_id = start_tweet_id
    seen = set()

    while pd.notna(current_id) and current_id in df.index and current_id not in seen:
        seen.add(current_id)
        row = df.loc[current_id]
        chain.append(row)
        current_id = row["in_response_to_tweet_id"]
        if len(chain) >= max_len:
            break

    chain.reverse()  # oldest first
    return chain


def print_thread(chain):
    for row in chain:
        speaker = "CUSTOMER" if row["inbound"] else "BRAND   "
        text = str(row["text"]).replace("\n", " ")
        print(f"  [{speaker}] {text}")
    print()


for brand in CANDIDATE_BRANDS:
    print("=" * 70)
    print(f"BRAND: {brand}")
    print("=" * 70)

    # Find brand replies (outbound messages from this brand)
    brand_replies = df[(df["author_id"] == brand) & (df["inbound"] == False)]

    # Sample 3 random brand replies, and reconstruct the thread each belongs to
    sample = brand_replies.sample(min(3, len(brand_replies)), random_state=42)

    for _, reply_row in sample.iterrows():
        thread = build_thread(reply_row["tweet_id"], df)
        print(f"Thread ending at tweet_id={reply_row['tweet_id']}:")
        print_thread(thread)