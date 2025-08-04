import os
import sys
import gradio as gr
import onnxruntime as ort
import numpy as np
import faiss
import pickle
import tempfile
from PIL import Image
from torchvision import transforms
from transformers import CLIPTokenizer

# Fix sys.path for db.logger import
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)
from db.logger import log_inference


# === Config ===
TOP_K = 3
IMG_ONNX_PATH = "/home/omoured/Desktop/product-match/models/clip_vitb32.onnx"
TXT_ONNX_PATH = "/home/omoured/Desktop/product-match/models/clip_text_encoder.onnx"
IMG_INDEX_PATH = "/home/omoured/Desktop/product-match/app/faiss/faiss_image.index"
TXT_INDEX_PATH = "/home/omoured/Desktop/product-match/app/faiss/faiss_text.index"
IMG_META_PATH = "/home/omoured/Desktop/product-match/app/faiss/image_id_to_meta.pkl"
TXT_META_PATH = "/home/omoured/Desktop/product-match/app/faiss/text_id_to_meta.pkl"

# === Load models and index ===
img_session = ort.InferenceSession(IMG_ONNX_PATH)
txt_session = ort.InferenceSession(TXT_ONNX_PATH)
img_input_name = img_session.get_inputs()[0].name
txt_input_name = txt_session.get_inputs()[0].name

img_index = faiss.read_index(IMG_INDEX_PATH)
txt_index = faiss.read_index(TXT_INDEX_PATH)

with open(IMG_META_PATH, "rb") as f:
    img_meta = pickle.load(f)  # it’s already a dict
img_meta = list(img_meta.items())  # convert to list of (id, metadata)


with open(TXT_META_PATH, "rb") as f:
    txt_meta = pickle.load(f)
txt_meta = list(txt_meta.items())


tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

# === Preprocessing ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def search(input_img, input_text):
    top_results = []
    input_text_clean = input_text.strip() if isinstance(input_text, str) else ""
    tmp_path = None

    try:
        real_img = isinstance(input_img, Image.Image)
        has_text = input_text_clean != ""

        if not real_img and not has_text:
            log_inference(
                input_image_path="none",
                top_results=[],
                error="❌ No image or text provided."
            )
            return [], "❌ Please upload an image or type a query."

        output_images = []
        captions = []

        if real_img:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                input_img.save(tmp.name)
                tmp_path = tmp.name

            image = input_img.convert("RGB")
            tensor = transform(image).unsqueeze(0).numpy()
            embedding = img_session.run(None, {img_input_name: tensor})[0]
            embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)

            scores, indices = img_index.search(embedding.astype(np.float32), TOP_K)
            meta_list = img_meta
        else:
            query = f"Looking for a {input_text_clean}"
            inputs = tokenizer(query, padding="max_length", max_length=77, return_tensors="np")
            token_ids = inputs["input_ids"].astype(np.int64)
            embedding = txt_session.run(None, {txt_input_name: token_ids})[0]
            embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
            scores, indices = txt_index.search(embedding.astype(np.float32), TOP_K)
            meta_list = txt_meta
            tmp_path = "text-query"

        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1:
                continue

            try:
                match_id, meta = meta_list[idx]
            except Exception:
                continue

            img_path = meta.get("image_path")
            if not img_path or not os.path.isfile(img_path):
                continue

            image = Image.open(img_path).convert("RGB")

            caption = "\n".join([
                f"🆔 ID: {match_id}",
                f"🎨 Color: {meta.get('color', 'N/A')}",
                f"👗 Product Type: {meta.get('product_type', 'N/A')}",
                f"🚻 Gender: {meta.get('gender', 'N/A')}",
                f"🛍️ Usage: {meta.get('usage', 'N/A')}",
                f"📦 Category: {meta.get('category', 'N/A')}",
                f"📈 Score: {score:.3f}"
            ])

            output_images.append(image)
            captions.append(caption)
            top_results.append({
                "match_id": match_id,
                "score": float(score),
                "metadata": meta,
                "image_path": img_path
            })

        log_inference(
            input_image_path=tmp_path or "no-image",
            top_results=top_results,
            error=None if output_images else "No matches found",
        )

        if not output_images:
            return [], "⚠️ No matching results found."

        return output_images, "\n\n".join(captions)

    except Exception as e:
        log_inference(
            input_image_path=tmp_path or "unknown",
            top_results=[],
            error=f"❌ Exception: {str(e)} | Text input: '{input_text_clean}'"
        )
        return [], f"❌ Error: {str(e)}"

example_queries = [
    ["/home/omoured/Desktop/product-match/data/Apparel/Girls/Images/images_with_product_ids/2697.jpg", ""],
    ["/home/omoured/Desktop/product-match/data/Footwear/Men/Images/images_with_product_ids/3150.jpg", ""],
    [None, "blue denim jeans"],
    [None, "white floral dress for summer"]  # Text-only example
]

# === Gradio UI ===
iface = gr.Interface(
    fn=search,
    inputs=[
        gr.Image(type="pil", label="Upload Image (optional)", height=224),
        gr.Textbox(label="Text Query (optional)", placeholder="e.g., red cotton top for girls")
    ],
    outputs=[
        gr.Gallery(label="Top 3 Matches", columns=3, height=300),
        gr.Textbox(label="Result Details")
    ],
    title="🛍️ Find your Fashion with Text or Image",
    description="Upload a product image or enter a description to find similar fashion items.",
    examples=example_queries,
    examples_per_page=4  # optional
)


if __name__ == "__main__":
    iface.launch(share=True, server_port=7860, server_name="0.0.0.0")
