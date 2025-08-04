import pickle
import os

# --- Paths ---
IMG_META_PATH = "/home/omoured/Desktop/product-match/app/faiss/image_id_to_meta.pkl"
TXT_META_PATH = "/home/omoured/Desktop/product-match/app/faiss/text_id_to_meta.pkl"

# === Helper Function ===
def test_metadata(meta_path, label=""):
    print(f"\n🔍 Testing {label} metadata from: {meta_path}")

    if not os.path.isfile(meta_path):
        print(f"❌ File not found: {meta_path}")
        return

    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    # Should be a dictionary
    if not isinstance(meta, dict):
        print(f"❌ ERROR: Expected dict, got {type(meta)}")
        return

    print(f"✅ Loaded {len(meta)} entries")
    
    sample_keys = list(meta.keys())[:5]
    print("🗝️ Sample keys:", sample_keys)

    first_key = sample_keys[0]
    first_meta = meta[first_key]
    print("📦 Sample metadata:")

    for k, v in first_meta.items():
        print(f"  {k}: {v}")

    # Extra checks
    required_keys = ["image_path", "title", "category"]
    for rk in required_keys:
        if rk not in first_meta:
            print(f"⚠️ Missing key in metadata: {rk}")
        elif not first_meta[rk]:
            print(f"⚠️ Empty value for key: {rk}")

    if "image_path" in first_meta and first_meta["image_path"]:
        exists = os.path.isfile(first_meta["image_path"])
        print(f"🖼️ Image path exists: {exists} -> {first_meta['image_path']}")
    print("✅ Done.\n")


# === Run Tests ===
test_metadata(IMG_META_PATH, label="Vision")
test_metadata(TXT_META_PATH, label="Text")

# import pandas as pd

# df = pd.read_csv("/home/omoured/Desktop/product-match/data/fashion_clip_text_embeddings.csv")
# print(df.columns.tolist())

