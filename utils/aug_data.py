# fix_data.py
import cv2, os

left_dir  = "data/train/Turn Left"
right_dir = "data/train/Turn Right"

os.makedirs(left_dir,  exist_ok=True)
os.makedirs(right_dir, exist_ok=True)

# Only flip ORIGINAL files (not aug_ files)
left_originals  = [f for f in os.listdir(left_dir)
                   if f.endswith(('.jpg','.jpeg','.png')) and not f.startswith('aug_')]
right_originals = [f for f in os.listdir(right_dir)
                   if f.endswith(('.jpg','.jpeg','.png')) and not f.startswith('aug_')]

# 1. Flip Left originals → Right
count_r = 0
for i, fname in enumerate(left_originals):
    out = os.path.join(right_dir, f"aug_from_left_{i:05d}.jpg")
    if not os.path.exists(out):
        img = cv2.imread(os.path.join(left_dir, fname))
        cv2.imwrite(out, cv2.flip(img, 1))
        count_r += 1

# 2. Flip Right originals → Left
count_l = 0
for i, fname in enumerate(right_originals):
    out = os.path.join(left_dir, f"aug_from_right_{i:05d}.jpg")
    if not os.path.exists(out):
        img = cv2.imread(os.path.join(right_dir, fname))
        cv2.imwrite(out, cv2.flip(img, 1))
        count_l += 1

print(f"✅ Added {count_r} images → Turn Right")
print(f"✅ Added {count_l} images → Turn Left")

# Show all class counts
print(f"\n📊 Final Dataset Count:")
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
