# src/feedback.py
import os
import shutil
import torch
from src.decision import ACTIONS

APPROVED_DIR   = "data/flagged/approved"
TRAIN_DIR      = "data/train"
MODEL_PATH     = "models/driving_cnn.pth"
CHECKPOINT_DIR = "models/checkpoints"

def merge_approved_into_train():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    total = 0

    for action in ACTIONS:
        src_dir = os.path.join(APPROVED_DIR, action)
        dst_dir = os.path.join(TRAIN_DIR, action)
        os.makedirs(dst_dir, exist_ok=True)

        if not os.path.exists(src_dir):
            continue

        files = [f for f in os.listdir(src_dir)
                 if f.endswith(('.jpg', '.jpeg', '.png'))]

        for f in files:
            dst = os.path.join(dst_dir, f"feedback_{f}")
            if not os.path.exists(dst):
                shutil.copy(os.path.join(src_dir, f), dst)
                total += 1

    print(f"Merged {total} approved frames into data/train/")
    return total

def save_checkpoint(version):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    dst = os.path.join(CHECKPOINT_DIR, f"v{version}_driving_cnn.pth")
    shutil.copy(MODEL_PATH, dst)
    print(f"Checkpoint saved -> {dst}")

def run_feedback_cycle():
    print("\nStarting Feedback Retrain Cycle...")

    existing = len(os.listdir(CHECKPOINT_DIR)) \
               if os.path.exists(CHECKPOINT_DIR) else 0
    version = existing + 1

    save_checkpoint(version)

    added = merge_approved_into_train()
    if added == 0:
        print("No new approved frames found.")
        print("Go to Review Flags tab first to approve flagged frames.")
        return

    print(f"\nRetraining with {added} new frames...")
    os.system("python train.py")

    print(f"\nFeedback Cycle {version} complete!")
    print(f"New frames added  : {added}")
    print(f"Checkpoint saved  : models/checkpoints/v{version}_driving_cnn.pth")

if __name__ == "__main__":
    run_feedback_cycle()
