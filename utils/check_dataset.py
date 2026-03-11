# check_dataset.py — Verify images + labels are correctly paired
import os
import cv2
import random
import matplotlib.pyplot as plt

TRAIN_DIR = "data/train"
ACTIONS   = ["Go Straight", "Brake", "Accelerate", "Turn Left", "Turn Right"]

def check_dataset():
    print("=== Dataset Label Verification ===\n")

    total = 0
    class_data = {}

    # 1. Count per class
    for action in ACTIONS:
        path  = os.path.join(TRAIN_DIR, action)
        files = [f for f in os.listdir(path)
                 if f.endswith(('.jpg', '.jpeg', '.png'))] \
                if os.path.exists(path) else []
        class_data[action] = files
        total += len(files)
        print(f"  {action:15s}: {len(files):5d} images")

    print(f"\n  {'TOTAL':15s}: {total:5d} images")

    # 2. Check random samples — show image + label
    print("\n=== Random Sample Check (5 images) ===")
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    for i, ax in enumerate(axes):
        # Pick random class and random image
        action = random.choice(ACTIONS)
        files  = class_data[action]
        if not files:
            continue

        fname    = random.choice(files)
        img_path = os.path.join(TRAIN_DIR, action, fname)
        img      = cv2.imread(img_path)
        img      = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ax.imshow(img)
        ax.set_title(f"Label: {action}\n{fname[:30]}...",
                     fontsize=8, color='green')
        ax.axis('off')

        print(f"  Sample {i+1}: [{action}] → {fname}")

    plt.tight_layout()
    plt.savefig("dataset_check.png", dpi=100, bbox_inches='tight')
    print("\nSaved → dataset_check.png")
    print("Open it to visually verify images match their labels!")

    # 3. Check label index matches ACTIONS list
    print("\n=== Label Index Mapping ===")
    for idx, action in enumerate(ACTIONS):
        print(f"  Index {idx} → '{action}'")

    print("\n=== This is what model sees during training ===")
    print("  (image_tensor, label_index)")
    for action in ACTIONS:
        files = class_data[action]
        if files:
            print(f"  ({files[0][:40]}..., {ACTIONS.index(action)})")

if __name__ == "__main__":
    check_dataset()
