import torch
import clip

# Output ONNX path
onnx_output_path = "./clip_vitb32.onnx"

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

model = model.float()
model.eval()

# Dummy input in FP32 (not half)
dummy_input = torch.randn(1, 3, 224, 224).to(device)

# Export ONNX
print("Exporting CLIP vision encoder to ONNX...")
torch.onnx.export(
    model.visual,
    dummy_input,
    onnx_output_path,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}},
    opset_version=14
)

print(f"✅ ONNX model saved to: {onnx_output_path}")
