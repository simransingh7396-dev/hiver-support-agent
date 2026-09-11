import pandas as pd
import re
from collections import Counter

DATA_PATH = r"C:\Users\simra\OneDrive\Desktop\archive\twcs\twcs.csv"

print("Loading data...")
df = pd.read_csv(DATA_PATH)

BRAND = "AskPlayStation"

# --- Step 1: find all inbound customer messages directed at this brand ---
# A customer message "directed at" the brand is one whose response_tweet_id
# eventually leads to a brand reply, OR simply mentions @AskPlayStation.
# Simplest reliable approach: customer messages that are in_response_to a
# brand tweet, OR that @mention the brand and are inbound.

brand_tweet_ids = set(df[df["author_id"] == BRAND]["tweet_id"])

customer_msgs = df[
    (df["inbound"] == True) &
    (df["text"].str.contains(f"@{BRAND}", case=False, na=False))
].copy()

print(f"Total customer messages mentioning @{BRAND}: {len(customer_msgs)}")

# --- Step 2: crude English filter ---
# Keep messages that are mostly ASCII (drops most Spanish/French/etc. accented text)
def is_mostly_ascii(text):
    if not isinstance(text, str):
        return False
    non_ascii = sum(1 for c in text if ord(c) > 127)
    return non_ascii / max(len(text), 1) < 0.05

customer_msgs = customer_msgs[customer_msgs["text"].apply(is_mostly_ascii)]
print(f"After English-ish filter: {len(customer_msgs)}")

# --- Step 3: sample ---
SAMPLE_SIZE = 300
sample = customer_msgs.sample(min(SAMPLE_SIZE, len(customer_msgs)), random_state=42)

# --- Step 4: clean text and count common words ---
STOPWORDS = set("""
the a an is are was were be been being to of and or but if in on at for with
i you he she it we they my your his her its our their this that these those
have has had do does did not no so just get got can cant can't dont don't
im i'm please help me my hi hey thanks thank you please pls u ur w
""".split())

# manually add brand mention / boilerplate noise
STOPWORDS.update({"askplaystation", "ps4", "ps5"})  # we'll re-surface these separately as needed

word_counter = Counter()
for text in sample["text"]:
    text = re.sub(r"http\S+", "", str(text))  # strip links
    text = re.sub(r"@\w+", "", text)  # strip @mentions
    words = re.findall(r"[a-zA-Z']+", text.lower())
    words = [w for w in words if w not in STOPWORDS and len(w) > 2]
    word_counter.update(words)

print("\n" + "=" * 60)
print("TOP 40 MOST COMMON WORDS IN CUSTOMER MESSAGES")
print("=" * 60)
for word, count in word_counter.most_common(40):
    print(f"{word:20s} {count}")

# --- Step 5: for a handful of interesting keywords, show 3 real example messages ---
KEYWORDS_TO_INSPECT = [
    "error", "code", "login", "password", "account", "network",
    "download", "update", "refund", "charge", "money", "bought",
    "purchase", "controller", "disc", "psn", "store", "game",
    "crash", "freeze", "reset", "connect", "wifi", "online",
    "ban", "banned", "hacked", "stolen", "subscription", "plus"
]

print("\n" + "=" * 60)
print("SAMPLE MESSAGES PER KEYWORD")
print("=" * 60)
for kw in KEYWORDS_TO_INSPECT:
    matches = sample[sample["text"].str.contains(kw, case=False, na=False)]
    if len(matches) == 0:
        continue
    print(f"\n--- '{kw}' ({len(matches)} matches) ---")
    for text in matches["text"].head(3):
        clean = str(text).replace("\n", " ")
        print(f"  - {clean}")

# Save the full sample to a CSV too, in case you want to skim later
OUT_PATH = "askplaystation_customer_sample.csv"
sample[["tweet_id", "text"]].to_csv(OUT_PATH, index=False)
print(f"\nFull sample saved to {OUT_PATH}")