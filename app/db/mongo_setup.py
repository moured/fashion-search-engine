from pymongo import MongoClient
import pandas as pd

# Mongo connection
client = MongoClient("mongodb://localhost:27017")
db = client["product_matching"]
collection = db["products"]
collection.delete_many({})  # Clear if needed

# Load the CSV with image paths
csv_path = "/home/omoured/Desktop/product-match/data/fashion_with_paths.csv"
df = pd.read_csv(csv_path)

# Prepare records
records = []
for _, row in df.iterrows():
    records.append({
        "product_id": row["ProductId"],
        "gender": row["Gender"],
        "category": row["Category"],
        "sub_category": row["SubCategory"],
        "product_type": row["ProductType"],
        "color": row["Colour"],
        "usage": row["Usage"],
        "title": row["ProductTitle"],
        "image_path": row["image_path"]
    })

# Insert
collection.insert_many(records)
print(f"✅ Inserted {len(records)} records into MongoDB")
