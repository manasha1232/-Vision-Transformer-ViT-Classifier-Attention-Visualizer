"""
Generate Synthetic Test Images for Vision Transformer (ViT) Classifier & Attention Visualizer
Day 27 - 30-Day Computer Vision Challenge
"""

import os
import cv2
import numpy as np

def create_synthetic_vit_demo(output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)
    img_path = os.path.join(output_dir, "demo_vit_test_image.jpg")

    # Create 224x224 RGB image (standard ViT input resolution)
    h, w = 224, 224
    img = np.ones((h, w, 3), dtype=np.uint8) * 240

    # Draw dark studio background vignetting
    for y in range(h):
        for x in range(w):
            dist = np.sqrt((x - 112)**2 + (y - 112)**2)
            val = int(max(180, 240 - dist * 0.4))
            img[y, x] = [val, val - 10, val - 20]

    # Draw primary target object (A prominent sports ball / target ring in center patch region)
    cv2.circle(img, (112, 112), 48, (40, 120, 240), -1) # Outer ring
    cv2.circle(img, (112, 112), 34, (255, 255, 255), -1) # Inner ring
    cv2.circle(img, (112, 112), 20, (220, 40, 40), -1) # Bullseye core
    cv2.circle(img, (112, 112), 8, (255, 255, 255), -1)

    # Draw fine crosshair lines in target patch
    cv2.line(img, (112, 50), (112, 174), (30, 30, 30), 2)
    cv2.line(img, (50, 112), (174, 112), (30, 30, 30), 2)

    # Draw subtle background distractor shapes in outer corner patches
    cv2.rectangle(img, (20, 20), (50, 50), (100, 180, 100), -1)
    cv2.putText(img, "ViT", (160, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 2)

    cv2.imwrite(img_path, img)
    print(f"[SUCCESS] Synthetic ViT test image saved: {img_path} ({w}x{h})")
    return img_path

if __name__ == "__main__":
    create_synthetic_vit_demo()
