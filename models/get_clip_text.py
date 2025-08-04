import torch
import clip
import torch.nn as nn

class CLIPTextEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.token_embedding = model.token_embedding
        self.transformer = model.transformer
        self.ln_final = model.ln_final
        self.text_projection = model.text_projection
        self.positional_embedding = model.positional_embedding

    def forward(self, token_ids):
        x = self.token_embedding(token_ids).float()  # [B, 77, D]
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # [77, B, D]
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # [B, 77, D]

        # Get <EOT> token embedding (49407)
        eot_mask = (token_ids == 49407)
        eot_positions = eot_mask.float().argmax(dim=1)

        batch_size = token_ids.size(0)
        x = x[torch.arange(batch_size), eot_positions]  # [B, D]
        x = self.ln_final(x)
        x = x @ self.text_projection
        return x

# === Load CLIP ===
device = "cpu"  # Or "cuda" if you want
model, _ = clip.load("ViT-B/32", device=device)
model = model.float().eval()
text_encoder = CLIPTextEncoder(model).float().eval()

# === Dummy input (1 sentence tokenized into 77 tokens) ===
dummy_input = torch.randint(0, 49408, (1, 77)).long()

# === Export to ONNX ===
onnx_output_path = "/home/omoured/Desktop/product-match/models/clip_text_encoder.onnx"
torch.onnx.export(
    text_encoder,
    dummy_input,
    onnx_output_path,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}},
    opset_version=14
)

print(f"✅ Exported text encoder to: {onnx_output_path}")
