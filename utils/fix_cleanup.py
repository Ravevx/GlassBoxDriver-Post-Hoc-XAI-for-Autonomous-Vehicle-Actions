# fix_cleanup.py
import os

left_dir  = "data/train/Turn Left"
right_dir = "data/train/Turn Right"

# Delete all augmented files (keep only originals)
for folder in [left_dir, right_dir]:
    deleted = 0
    for fname in os.listdir(folder):
        if fname.startswith("aug_"):  # only delete augmented ones
            os.remove(os.path.join(folder, fname))
            deleted += 1
    print(f"🗑️  Deleted {deleted} augmented files from {folder}")

# Show clean counts
print(f"\nClean Dataset Count:")
print(f"{'─'*30}")
train_dir = "data/train"
total = 0
for action in sorted(os.listdir(train_dir)):
    action_path = os.path.join(train_dir, action)
    if os.path.isdir(action_path):
        count = len([f for f in os.listdir(action_path)
                     if f.endswith(('.jpg','.jpeg','.png'))])
        print(f"  {action:20s}: {count} images")
        total += count
print(f"{'─'*30}")
print(f"  {'TOTAL':20s}: {total} images")
