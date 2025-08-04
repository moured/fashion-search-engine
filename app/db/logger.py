from pymongo import MongoClient
from datetime import datetime

import numpy as np

client = MongoClient("mongodb://localhost:27017")
db = client["product_matching"]
logs_collection = db["logs"]

def log_inference(input_image_path, top_results, error=None):
    # Convert numpy types to native Python
    for result in top_results:
        result["match_id"] = int(result.get("match_id", -1))  # convert match_id to int
        result["score"] = float(result.get("score", 0.0))     # convert score to float

    doc = {
        "timestamp": datetime.utcnow(),
        "input_image": input_image_path,
        "top_matches": top_results,
        "error": error,
    }

    try:
        logs_collection.insert_one(doc)
    except Exception as e:
        print(f"❌ Failed to log inference: {e}")
