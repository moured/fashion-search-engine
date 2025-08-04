import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import onnxruntime as ort
from transformers import CLIPTokenizer

# --- Config ---
CSV_PATH = "/home/omoured/Desktop/product-match/data/fashion_with_paths.csv"
OUTPUT_CSV = "/home/omoured/Desktop/product-match/data/fashion_clip_text_embeddings.csv"
ONNX_MODEL_PATH = "/home/omoured/Desktop/product-match/models/clip_text_encoder.onnx"

# --- Load tokenizer ---
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

# --- Load ONNX model ---
session = ort.InferenceSession(ONNX_MODEL_PATH)
input_name = session.get_inputs()[0].name

# --- Load data ---
df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=["ProductTitle"])

records = []

# --- Encode each product to text embedding ---
for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating text embeddings"):
    try:
        image_path = row["image_path"]
        # Construct sentence from metadata
        sentence = f"{row['ProductTitle']}, a {row['Colour']} {row['ProductType'].lower()} for {row['Gender'].lower()}s, used for {row['Usage'].lower()} in the {row['Category']} category."

        # Tokenize: output should be int64 IDs of shape [1, 77]
        tokens = tokenizer(sentence, padding="max_length", truncation=True, max_length=77, return_tensors="np")
        input_ids = tokens["input_ids"].astype(np.int64)

        # Inference
        embedding = session.run(None, {input_name: input_ids})[0]
        embedding_flat = embedding.flatten()

        # Store result
        records.append({
            "product_id": row["ProductId"],
            "text": row.get("ProductTitle", ""),
            "Category": row.get("Category", "N/A"),
            "Colour": row.get("Colour", "N/A"),
            "ProductType": row.get("ProductType", "N/A"),
            "Gender": row.get("Gender", "N/A"),
            "Usage": row.get("Usage", "N/A"),
            "image_path": image_path,
            "embedding": ",".join(map(str, embedding_flat))
        })
        
    except Exception as e:
        print(f"⚠️ Error embedding text for product {row.get('ProductId', 'unknown')}: {e}")

# --- Save output CSV ---
output_df = pd.DataFrame(records)
output_df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Text embeddings saved to: {OUTPUT_CSV}")
