# 🧠 Vision Transformer (ViT) Classifier & Attention Rollout Heatmap Visualizer

[![Day](https://img.shields.io/badge/Day-27--30-blue?style=for-the-badge&logo=python)](https://github.com/manasha1232/30-Day-Computer-Vision-Challenge)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.0.0-green?style=for-the-badge&logo=opencv)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-red?style=for-the-badge)](LICENSE)

A Deep Learning & Explainable AI (XAI) engine implementing a **PyTorch Vision Transformer (ViT) Image Classifier** with **Multi-Head Self-Attention (MHSA) Rollout Heatmap Visualization**. Projects $16 \times 16$ spatial patch token embeddings and visualizes neural self-attention rollout maps back onto the input image space.

---

## 🌟 Key Features

- 🧩 **$16 \times 16$ Patch Embedding Tokenization**: Tokenizes $224 \times 224$ images into $196$ spatial visual tokens + $1$ `[CLS]` token.
- ⚡ **Multi-Head Self-Attention (MHSA)**: Extracts attention weight matrices $\mathbf{A}^{(l)} \in \mathbb{R}^{197 \times 197}$ across transformer encoder layers.
- 🗺️ **Attention Rollout Heatmaps**: Projects self-attention rollout ($\mathbf{A}_{\text{rollout}} = \prod_{l=1}^L (0.5 \mathbf{A}^{(l)} + 0.5 \mathbf{I})$) back to pixel space.
- 🖼️ **4-Panel Visualizer Grid**:
  1. Original Input Image ($224 \times 224$)
  2. $16 \times 16$ Patch Token Grid Overlay ($14 \times 14$ patches)
  3. Self-Attention Rollout Heatmap Overlay
  4. Class Probabilities Bar Chart
- 📊 **Telemetry Audit Exporter**: Exports JSON telemetry detailing class probabilities, attention entropy, and layer execution metrics.

---

## 🛠️ Installation & Setup

```bash
# Clone the repository
git clone https://github.com/manasha1232/vision_transformer_classifier.git
cd vision_transformer_classifier

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Execution Guide

### 1️⃣ Generate Synthetic Test Image & Run Pipeline
```bash
python generate_demo_vit_input.py
python vit_attention_classifier.py
```

### 2️⃣ Run with Custom Test Image
```bash
python vit_attention_classifier.py --image path/to/image.jpg
```

---

## 📊 Sample Output Telemetry JSON

```json
{
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
        "top_predicted_class": "Target Bullseye",
        "confidence_score": 0.4125,
        "all_class_probabilities": {
            "Target Bullseye": 0.4125,
            "Geometric Ring": 0.2104,
            "Sports Ball": 0.1852,
            "Abstract Art": 0.1118,
            "Background Distractor": 0.0801
        }
    },
    "attention_metrics": {
        "mean_attention_weight": 0.3842,
        "max_attention_weight": 1.0,
        "attention_entropy": 24.18
    },
    "performance": {
        "execution_duration_sec": 0.428
    }
}
```

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
