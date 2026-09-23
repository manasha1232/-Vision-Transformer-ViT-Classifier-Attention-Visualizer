"""
Vision Transformer (ViT) Image Classifier & Attention Rollout Heatmap Visualizer
Day 27 - 30-Day Computer Vision Challenge

Features:
- PyTorch Vision Transformer (ViT-16x16 Patch Embedding) Network
- Multi-Head Self-Attention (MHSA) Weight Matrix Extraction
- Attention Rollout Map Generator (Projecting [CLS] Token Attention back to 14x14 Spatial Patch Grid)
- 4-Panel Visualizer Grid (Input, 16x16 Patch Token Grid, Attention Rollout Heatmap, Class Probabilities)
- Structured Telemetry JSON Audit Exporter
"""

import os
import sys
import time
import json
import math
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------------------------------------------
# PyTorch Vision Transformer Architecture & Attention Rollout Module
# -------------------------------------------------------------------
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=192):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W) -> (B, EmbedDim, H_patch, W_patch) -> (B, NumPatches, EmbedDim)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class SelfAttentionWithWeights(nn.Module):
    def __init__(self, embed_dim=192, num_heads=3):
        super(SelfAttentionWithWeights, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1) # (B, Heads, N, N)

        x = (attn_weights @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x, attn_weights

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim=192, num_heads=3, mlp_ratio=4.0):
        super(TransformerEncoderBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = SelfAttentionWithWeights(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, embed_dim)
        )

    def forward(self, x):
        norm_x = self.norm1(x)
        attn_out, attn_weights = self.attn(norm_x)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, attn_weights

class VisionTransformerClassifier(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=5, embed_dim=192, depth=4, num_heads=3):
        super(VisionTransformerClassifier, self).__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim) * 0.02)
        
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Classes
        self.classes = ["Target Bullseye", "Geometric Ring", "Sports Ball", "Abstract Art", "Background Distractor"]

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        attn_weights_list = []
        for block in self.blocks:
            x, weights = block(x)
            attn_weights_list.append(weights)

        x = self.norm(x)
        cls_out = x[:, 0]
        logits = self.head(cls_out)
        return logits, attn_weights_list

# -------------------------------------------------------------------
# Attention Rollout Computation
# -------------------------------------------------------------------
def compute_attention_rollout(attn_weights_list):
    # attn_weights_list: List of (B, Heads, N, N) tensors
    rollout = torch.eye(attn_weights_list[0].shape[-1]).unsqueeze(0) # (1, N, N)

    for weights in attn_weights_list:
        # Average over attention heads
        mean_weights = weights.mean(dim=1).detach().cpu() # (B, N, N)
        # Add identity matrix for residual connection & normalize
        eye = torch.eye(mean_weights.shape[-1]).unsqueeze(0)
        attn_fused = 0.5 * mean_weights + 0.5 * eye
        attn_fused = attn_fused / attn_fused.sum(dim=-1, keepdim=True)

        rollout = torch.bmm(attn_fused, rollout)

    # Extract attention from [CLS] token (index 0) to all patch tokens (indices 1..196)
    cls_attention = rollout[0, 0, 1:].numpy()
    grid_size = int(math.sqrt(len(cls_attention)))
    attention_map = cls_attention.reshape(grid_size, grid_size)
    
    # Normalize map to [0, 1]
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
    return attention_map

# -------------------------------------------------------------------
# Pipeline Execution
# -------------------------------------------------------------------
def run_vit_classification(image_path, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)
    report_json_path = os.path.join(output_dir, "sample_vit_report.json")
    output_img_path = os.path.join(output_dir, "sample_vit_attention_output.jpg")

    start_time = time.time()

    if not os.path.exists(image_path):
        from generate_demo_vit_input import create_synthetic_vit_demo
        image_path = create_synthetic_vit_demo(output_dir)

    orig_img = cv2.imread(image_path)
    if orig_img is None:
        raise ValueError(f"Failed to load test image at {image_path}")

    # Resize to standard ViT resolution (224x224)
    img_224 = cv2.resize(orig_img, (224, 224))
    rgb_img = cv2.cvtColor(img_224, cv2.COLOR_BGR2RGB)

    # Prepare PyTorch Tensor
    img_tensor = torch.from_numpy(rgb_img).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    # Model Forward Pass
    device = torch.device("cpu")
    model = VisionTransformerClassifier(img_size=224, patch_size=16, num_classes=5).to(device)
    model.eval()

    with torch.no_grad():
        logits, attn_weights_list = model(img_tensor)
        probs = F.softmax(logits, dim=-1).squeeze(0).numpy()

    top_class_idx = int(np.argmax(probs))
    top_class_name = model.classes[top_class_idx]
    top_class_prob = float(probs[top_class_idx])

    # Compute Attention Rollout Map
    attn_map_14x14 = compute_attention_rollout(attn_weights_list)

    # Upsample 14x14 attention grid to 224x224
    attn_map_224 = cv2.resize(attn_map_14x14, (224, 224), interpolation=cv2.INTER_CUBIC)
    heatmap_colored = cv2.applyColorMap((attn_map_224 * 255).astype(np.uint8), cv2.COLORMAP_JET)

    # Blend Heatmap with original image
    heatmap_overlay = cv2.addWeighted(img_224, 0.5, heatmap_colored, 0.5, 0)

    # Generate 4-Panel Visualization Grid
    # Panel 1: Original Input Image
    p1 = img_224.copy()
    cv2.putText(p1, "1. INPUT (224x224)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    # Panel 2: 16x16 Patch Token Grid Overlay (14x14 patches)
    p2 = img_224.copy()
    for i in range(0, 224, 16):
        cv2.line(p2, (i, 0), (i, 224), (0, 255, 255), 1)
        cv2.line(p2, (0, i), (224, i), (0, 255, 255), 1)
    cv2.putText(p2, "2. 16x16 PATCH GRID", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    # Panel 3: Self-Attention Rollout Heatmap Overlay
    p3 = heatmap_overlay.copy()
    cv2.putText(p3, "3. ATTENTION ROLLOUT", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    # Panel 4: Top Class Probabilities Bar Chart
    p4 = np.ones((224, 224, 3), dtype=np.uint8) * 245
    cv2.putText(p4, "4. PREDICTION PROBS", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
    for i, (cls_n, pr) in enumerate(zip(model.classes, probs)):
        bar_w = int(pr * 140)
        y_pos = 50 + i * 35
        cv2.rectangle(p4, (10, y_pos), (10 + bar_w, y_pos + 18), (220, 100, 30), -1)
        cv2.putText(p4, f"{cls_n[:12]}: {pr*100:.1f}%", (15 + bar_w, y_pos + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    top_row = np.hstack([p1, p2])
    bottom_row = np.hstack([p3, p4])
    grid_montage = np.vstack([top_row, bottom_row])

    execution_duration = time.time() - start_time

    cv2.imwrite(output_img_path, grid_montage)
    print(f"[SUCCESS] Saved 4-Panel ViT Attention Grid to: {output_img_path}")

    # Build Telemetry JSON
    report_data = {
        "project": "Vision Transformer (ViT) Classifier & Attention Visualizer",
        "day": 27,
        "status": "SUCCESS",
        "architecture": {
            "patch_size": "16x16",
            "num_patch_tokens": 196,
            "embedding_dimension": 192,
            "num_heads": 3,
            "num_layers": 4
        },
        "classification_results": {
            "top_predicted_class": top_class_name,
            "confidence_score": float(round(top_class_prob, 4)),
            "all_class_probabilities": {cls_n: float(round(p, 4)) for cls_n, p in zip(model.classes, probs)}
        },
        "attention_metrics": {
            "mean_attention_weight": float(round(float(np.mean(attn_map_14x14)), 4)),
            "max_attention_weight": float(round(float(np.max(attn_map_14x14)), 4)),
            "attention_entropy": float(round(float(-np.sum(attn_map_14x14 * np.log(attn_map_14x14 + 1e-8))), 4))
        },
        "performance": {
            "execution_duration_sec": float(round(execution_duration, 3))
        },
        "output_files": {
            "attention_grid_image": output_img_path,
            "telemetry_report": report_json_path
        }
    }

    with open(report_json_path, "w") as f:
        json.dump(report_data, f, indent=4)

    print(f"[SUCCESS] Telemetry JSON report exported to: {report_json_path}")
    print("\n--- Vision Transformer Summary ---")
    print(f"Predicted Class: {top_class_name} ({top_class_prob * 100:.1f}%)")
    print(f"Max Attention Peak: {report_data['attention_metrics']['max_attention_weight']}")
    print(f"Execution Time: {report_data['performance']['execution_duration_sec']} sec")

    return report_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vision Transformer Attention Visualizer")
    parser.add_argument("--image", type=str, default="output/demo_vit_test_image.jpg", help="Path to input test image")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    args = parser.parse_args()

    run_vit_classification(image_path=args.image, output_dir=args.output)
