import os
import uuid
import onnxruntime as ort
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd
from torchvision import transforms

# --- Config ---
MODEL_PATH = "/home/omoured/Desktop/product-match/models/clip_vitb32.onnx"
CSV_PATH = "/home/omoured/Desktop/product-match/data/fashion_with_paths.csv"
OUTPUT_CSV = "/home/omoured/Desktop/product-match/data/fashion_clip_embeddings.csv"

# --- Preprocessing: adjust to CLIP input format ---
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
])

# --- Load model ---
session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name

# --- Load metadata CSV with image paths ---
df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=["image_path"])  # Safety check

records = []

# --- Embed each image ---
for _, row in tqdm(df.iterrows(), total=len(df), desc="Embedding images"):
    image_path = row["image_path"]

    try:
        # Load and preprocess image
        img = Image.open(image_path).convert("RGB")
        tensor = preprocess(img).unsqueeze(0).numpy()

        # Run ONNX model
        embedding = session.run(None, {input_name: tensor})[0]
        embedding_flat = embedding.flatten()

        # Append metadata + embedding
        records.append({
            "product_id": row["ProductId"],
            "title": row.get("ProductTitle", ""),         # ✅ Used by FAISS build script
            "Category": row.get("Category", "N/A"),        # ✅ Match case exactly
            "Colour": row.get("Colour", "N/A"),
            "ProductType": row.get("ProductType", "N/A"),
            "Gender": row.get("Gender", "N/A"),
            "Usage": row.get("Usage", "N/A"),
            "image_path": image_path,
            "embedding": ",".join(map(str, embedding_flat))
        })

    except Exception as e:
        print(f"⚠️ Error processing {image_path}: {e}")

# --- Save output CSV ---
output_df = pd.DataFrame(records)
output_df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Saved embeddings to {OUTPUT_CSV}")
