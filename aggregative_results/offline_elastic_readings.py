# TODO: REMOVE THIS FILE AND ADD THE CAPABILITY OF CALCULATING AGGREGATIONS OVER A PREDEFINED INTERVAL IN THE SECOND FILE


# elastic_to_dataframes.py

from elasticsearch import Elasticsearch
import pandas as pd
from datetime import datetime, timedelta
import os

# ------------------------------------------------------------
# Step 1: Connect to Elasticsearch
# ------------------------------------------------------------
es = Elasticsearch("http://127.0.0.1:9200", basic_auth=("elastic", "SVR4mUZl"), verify_certs=False)

if not es.ping():
    raise ValueError("Elasticsearch connection failed")
print("Connected to Elasticsearch")

# ------------------------------------------------------------
# Step 2: Ask how many days back to load
# ------------------------------------------------------------
days_back_input = input("Enter number of days back to load (e.g., 2): ").strip()
try:
    days_back = int(days_back_input)
except ValueError:
    raise ValueError("Invalid input. Please enter an integer number of days.")

end_time = datetime.utcnow()
start_time = end_time - timedelta(days=days_back)

query_body = {
    "size": 10000,
    "query": {
        "range": {
            "timestamp": {
                "gte": start_time.isoformat(),
                "lte": end_time.isoformat()
            }
        }
    }
}

print(f"Fetching data from {start_time.isoformat()} to {end_time.isoformat()}")


# ------------------------------------------------------------
# Step 3: Scroll and collect all matching documents
# ------------------------------------------------------------
scroll_time = '2m'
response = es.search(index="system_metrics", body=query_body, scroll=scroll_time)
scroll_id = response['_scroll_id']
all_docs = [hit['_source'] for hit in response['hits']['hits']]

while True:
    response = es.scroll(scroll_id=scroll_id, scroll=scroll_time)
    hits = response['hits']['hits']
    if not hits:
        break
    all_docs.extend(hit['_source'] for hit in hits)

es.clear_scroll(scroll_id=scroll_id)

print(f"Loaded {len(all_docs)} total documents")

# ------------------------------------------------------------
# Step 4: Create DataFrame
# ------------------------------------------------------------
df = pd.DataFrame(all_docs)
print("DataFrame shape:", df.shape)
print("Available columns:", df.columns.tolist())
print(df.head())

# ------------------------------------------------------------
# Step 5: Group by session_id
# ------------------------------------------------------------
if "session_id" not in df.columns:
    raise KeyError("'session_id' column not found in the data")

grouped_sessions = df.groupby("session_id")

print(f"\nNumber of unique sessions: {len(grouped_sessions)}")
for session_id, group_df in grouped_sessions:
    print(f"Session: {session_id} | Rows: {len(group_df)}")


# ------------------------------------------------------------
# Step 6: Extract to csv by session_id
# ------------------------------------------------------------

# Create output folder if it doesn't exist
output_dir = "dataframes"
os.makedirs(output_dir, exist_ok=True)

# Export each session to a CSV file (skip if it already exists)
for session_id, group_df in df.groupby("session_id"):
    safe_id = session_id.replace(" ", "_").replace("|", "_")
    filename = f"{safe_id}.csv"
    filepath = os.path.join(output_dir, filename)

    if os.path.exists(filepath):
        print(f"Skipping: {filepath} (already exists)")
        continue

    group_df.to_csv(filepath, index=False)
    print(f"Saved: {filepath} ({len(group_df)} rows)")
