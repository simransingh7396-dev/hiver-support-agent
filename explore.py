import pandas as pd

# Path to the dataset
DATA_PATH = r"C:\Users\simra\OneDrive\Desktop\archive\twcs\twcs.csv"

# Load the data
print("Loading data...")
df = pd.read_csv(DATA_PATH)

print(f"\nTotal rows: {len(df):,}")
print(f"\nColumns: {list(df.columns)}")

# Count messages per brand (author_id that looks like a brand name, not a number)
print("\nTop 20 brands by message volume:")
brand_counts = df[df['inbound'] == False]['author_id'].value_counts().head(20)
print(brand_counts)