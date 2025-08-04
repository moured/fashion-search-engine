import torch
import clip
import numpy as np
import onnxruntime as ort
import time
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode

# ---------------------------
# Load and preprocess a sample image
# ---------------------------
image_path = "/home/omoured/Desktop/product-match/data/images/BABY_PRODUCTS/13993_BABY_P_val.jpeg"  # adjust as needed

transform = Compose([
    Resize(224, interpolation=InterpolationMode.BICUBIC),
    CenterCrop(224),
    ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073),
              (0.26862954, 0.26130258, 0.27577711))
])

image = transform(Image.open(image_path).convert("RGB"))
image = image.unsqueeze(0).numpy()  # for ONNX

# ---------------------------
# Original CLIP model embedding
# ---------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

# Time PyTorch inference
start_time = time.time()
with torch.no_grad():
    clip_embedding = model.encode_image(torch.tensor(image).to(device)).cpu().numpy()
torch_time = time.time() - start_time

print(f"📦 Original CLIP embedding shape: {clip_embedding.shape}")
print(f"🕒 PyTorch inference time: {torch_time:.4f} seconds")

# ---------------------------
# ONNX model embedding
# ---------------------------
onnx_path = "/home/omoured/Desktop/product-match/models/clip_vitb32.onnx"
session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

start_time = time.time()
onnx_embedding = session.run(
    output_names=["output"],
    input_feed={"input": image}
)[0]
onnx_time = time.time() - start_time

print(f"📦 ONNX embedding shape: {onnx_embedding.shape}")
print(f"🕒 ONNX inference time: {onnx_time:.4f} seconds")

# ---------------------------
# Cosine similarity between embeddings
# ---------------------------
from numpy.linalg import norm

def cosine_sim(a, b):
    return np.dot(a, b.T) / (norm(a) * norm(b))

similarity = cosine_sim(clip_embedding[0], onnx_embedding[0])
print(f"✅ Cosine similarity between CLIP and ONNX: {similarity:.4f}")

# ---------------------------
# Comparison Summary
# ---------------------------
speedup = torch_time / onnx_time if onnx_time > 0 else float("inf")
print(f"🚀 Inference Speedup (ONNX vs PyTorch): {speedup:.2f}x faster")
