import os
import pandas as pd
import numpy as np
import faiss
import pickle

# --- Config ---
IMG_CSV = "/home/omoured/Desktop/product-match/data/fashion_clip_embeddings.csv"
TXT_CSV = "/home/omoured/Desktop/product-match/data/fashion_clip_text_embeddings.csv"

OUT_DIR = "/home/omoured/Desktop/product-match/app/faiss"
os.makedirs(OUT_DIR, exist_ok=True)

IMG_INDEX_PATH = os.path.join(OUT_DIR, "faiss_image.index")
TXT_INDEX_PATH = os.path.join(OUT_DIR, "faiss_text.index")
IMG_META_PATH = os.path.join(OUT_DIR, "image_id_to_meta.pkl")
TXT_META_PATH = os.path.join(OUT_DIR, "text_id_to_meta.pkl")

# --- Helpers ---
def parse_embedding(s):
    return np.array([float(x) for x in s.split(",")], dtype=np.float32)


def build_index(df, index_path, meta_path, mode="text"):
    embeddings = np.stack(df["embedding"].apply(parse_embedding).tolist())
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, index_path)

    # ✅ FIX: Use a dictionary instead of a list
    id_to_meta = {}
    for _, row in df.iterrows():
        product_id = int(row["product_id"])
        if mode == "text":
            id_to_meta[product_id] = {
                "title": row.get("text", ""),
                "color": row.get("Colour", "N/A"),
                "product_type": row.get("ProductType", "N/A"),
                "gender": row.get("Gender", "N/A"),
                "usage": row.get("Usage", "N/A"),
                "category": row.get("Category", "N/A"),
                "image_path": row.get("image_path", None)
            }
        elif mode == "image":
            id_to_meta[product_id] = {
                "title": row.get("ProductTitle", ""),       # ✅ FIXED
                "color": row.get("Colour", "N/A"),
                "product_type": row.get("ProductType", "N/A"),
                "gender": row.get("Gender", "N/A"),
                "usage": row.get("Usage", "N/A"),
                "category": row.get("Category", "N/A"),
                "image_path": row.get("image_path", None)
            }
            
    with open(meta_path, "wb") as f:
        pickle.dump(id_to_meta, f)  # ✅ Store as dict

    print(f"✅ Built index: {index_path} with {index.ntotal} vectors")
    print(f"📦 Saved metadata to: {meta_path}")



# --- Load CSVs ---
img_df = pd.read_csv(IMG_CSV)
txt_df = pd.read_csv(TXT_CSV)

print(f"📷 Loaded {len(img_df)} image embeddings")
print(f"📝 Loaded {len(txt_df)} text embeddings")

# --- Build ---
build_index(img_df, index_path=IMG_INDEX_PATH, meta_path=IMG_META_PATH, mode="image")
build_index(txt_df, index_path=TXT_INDEX_PATH, meta_path=TXT_META_PATH, mode="text")
