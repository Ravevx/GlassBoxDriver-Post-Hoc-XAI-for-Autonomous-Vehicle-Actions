# balance_dataset.py — Undersample all classes to equal size
import os
import random

TRAIN_DIR = "data/train"
ACTIONS   = ["Go Straight", "Brake", "Accelerate", "Turn Left", "Turn Right"]

def balance():
    # Count current images per class
    class_files = {}
    for action in ACTIONS:
        path  = os.path.join(TRAIN_DIR, action)
        files = [f for f in os.listdir(path)
                 if f.endswith(('.jpg', '.jpeg', '.png'))
                 and not f.startswith('feedback_')
                 and not f.startswith('aug_')]
        class_files[action] = files

    counts = {a: len(f) for a, f in class_files.items()}
    print("Before balancing:")
    for action, count in counts.items():
        print(f"  {action:15s}: {count}")

    # Target = smallest class
    target = min(counts.values())
    print(f"\nTarget count per class: {target}")

    # Remove excess randomly
    print("\nBalancing...")
    for action, files in class_files.items():
        if len(files) > target:
            to_remove = random.sample(files, len(files) - target)
            for f in to_remove:
                os.remove(os.path.join(TRAIN_DIR, action, f))
            print(f"  {action:15s}: {len(files)} -> {target} "
                  f"(removed {len(to_remove)})")
        else:
            print(f"  {action:15s}: {len(files)} (unchanged)")

    print(f"\nDone! All classes now have {target} images.")
    print(f"Total dataset size: {target * len(ACTIONS)}")

if __name__ == "__main__":
    balance()
