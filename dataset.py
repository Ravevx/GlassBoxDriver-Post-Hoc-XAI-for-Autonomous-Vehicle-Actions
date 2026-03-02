import os
import json
import cv2
import shutil
import numpy as np
from tqdm import tqdm
import random
# ─── CONFIG ───────────────────────────────────────────────
NUSCENES_ROOT = os.path.join("data", "nuscenes")
OUTPUT_ROOT   = os.path.join("data", "train")
VERSION       = "v1.0-mini"
# ──────────────────────────────────────────────────────────

ACTIONS = ["Go Straight", "Brake", "Accelerate", "Turn Left", "Turn Right"]

def load_table(name):
    path = os.path.join(NUSCENES_ROOT, VERSION, f"{name}.json")
    print(f"  Loading: {path}")
    with open(path, "r") as f:
        records = json.load(f)
    return {r["token"]: r for r in records}

def get_action_label(ego_pose_curr, ego_pose_next):
    curr  = np.array(ego_pose_curr["translation"])
    nxt   = np.array(ego_pose_next["translation"])
    delta = nxt - curr
    speed = np.linalg.norm(delta[:2])

    # Get yaw from both poses
    def get_yaw(pose):
        w, x, y, z = pose["rotation"]
        return np.degrees(np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z)))

    yaw_curr = get_yaw(ego_pose_curr)
    yaw_next = get_yaw(ego_pose_next)

    # Yaw change between frames
    yaw_diff = yaw_next - yaw_curr
    # Normalize to [-180, 180]
    yaw_diff = (yaw_diff + 180) % 360 - 180

    if speed < 0.2:
        return "Brake"
    elif yaw_diff > 1.5:
        return "Turn Left"
    elif yaw_diff < -1.5:
        return "Turn Right"
    elif speed > 5.0:
        return "Accelerate"
    else:
        return "Go Straight"



def extract_dataset():
    print("📂 Loading nuScenes tables...")

    samples     = load_table("sample")
    sample_data = load_table("sample_data")
    ego_poses   = load_table("ego_pose")

    # Pre-filter: only CAM_FRONT keyframes
    # We identify them by "CAM_FRONT" in filename AND is_key_frame = True
    cam_front_by_sample = {}  # sample_token → sample_data record
    for sd in sample_data.values():
        if "CAM_FRONT" in sd["filename"] and sd["is_key_frame"]:
            cam_front_by_sample[sd["sample_token"]] = sd

    print(f"  Found {len(cam_front_by_sample)} CAM_FRONT keyframes")

    # Create output folders
    for action in ACTIONS:
        os.makedirs(os.path.join(OUTPUT_ROOT, action), exist_ok=True)

    print("🔄 Extracting frames and assigning action labels...")

    counts      = {a: 0 for a in ACTIONS}
    sample_list = list(samples.values())

    for i in tqdm(range(len(sample_list) - 1)):
        sample_curr = sample_list[i]
        sample_next = sample_list[i + 1]

        # Only consecutive samples from same scene
        if sample_curr["scene_token"] != sample_next["scene_token"]:
            continue

        # Get CAM_FRONT for both
        sd_curr = cam_front_by_sample.get(sample_curr["token"])
        sd_next = cam_front_by_sample.get(sample_next["token"])

        if sd_curr is None or sd_next is None:
            continue

        # Get ego poses
        ego_curr = ego_poses.get(sd_curr["ego_pose_token"])
        ego_next = ego_poses.get(sd_next["ego_pose_token"])

        if ego_curr is None or ego_next is None:
            continue

        # Derive action label
        action = get_action_label(ego_curr, ego_next)

        # Copy image
        img_src = os.path.join(NUSCENES_ROOT, sd_curr["filename"])
        if not os.path.exists(img_src):
            continue

        img_dst = os.path.join(
            OUTPUT_ROOT, action,
            f"{action.replace(' ', '_')}_{counts[action]:05d}.jpg"
        )
        shutil.copy(img_src, img_dst)
        counts[action] += 1

    print("\n✅ Dataset extraction complete!")
    print("Images per class:")
    total = 0
    for action, count in counts.items():
        print(f"  {action:15s}: {count} images")
        total += count
    print(f"\n  Total: {total} images")
    print(f"  Saved to: {OUTPUT_ROOT}")
    left_dir  = os.path.join(OUTPUT_ROOT, "Turn Left")
    right_dir = os.path.join(OUTPUT_ROOT, "Turn Right")

    left_images = os.listdir(left_dir)
    existing_right = counts["Turn Right"]

    print("🔄 Augmenting Turn Right by flipping Turn Left images...")
    for i, fname in enumerate(left_images):
        src = os.path.join(left_dir, fname)
        img = cv2.imread(src)
        if img is None:
            continue
        flipped = cv2.flip(img, 1)  # Horizontal flip = mirror
        dst = os.path.join(right_dir,
            f"Turn_Right_{existing_right + i:05d}_aug.jpg")
        cv2.imwrite(dst, flipped)

    print(f"  Added {len(left_images)} augmented Turn Right images")
    print(f"  Turn Right now: {existing_right + len(left_images)} images")
if __name__ == "__main__":
    extract_dataset()
