import onnxruntime as ort
import numpy as np
import faiss
import pickle
import argparse
from PIL import Image
from torchvision import transforms

# --- Config ---
ONNX_MODEL_PATH = "/home/omoured/Desktop/product-match/models/clip_vitb32.onnx"
INDEX_PATH = "/home/omoured/Desktop/product-match/app/faiss/faiss_index.index"
META_PATH = "/home/omoured/Desktop/product-match/app/faiss/id_to_meta.pkl"

# --- Preprocessing (same as training) ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

# --- Load Model + Index + Metadata ---
session = ort.InferenceSession(ONNX_MODEL_PATH)
input_name = session.get_inputs()[0].name

index = faiss.read_index(INDEX_PATH)

with open(META_PATH, "rb") as f:
    id_to_meta = pickle.load(f)

# --- Search Function ---
def search(query_image_path, top_k=5):
    image = Image.open(query_image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).numpy()

    embedding = session.run(None, {input_name: tensor})[0]
    embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)  # normalize

    scores, indices = index.search(embedding.astype(np.float32), top_k)

    print(f"\n🔍 Top {top_k} Matches for: {query_image_path}")
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        match_id = list(id_to_meta.keys())[idx]
        meta = id_to_meta[match_id]
        print(f"  - {score:.4f} | {meta['category']} | {meta['image_path']}")

# --- CLI ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to query image")
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    search(args.image, args.top_k)
