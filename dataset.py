# dataset.py — nuScenes Mini sweeps extractor with direct timestamp parsing
import os, json, shutil, numpy as np
from tqdm import tqdm
from bisect import bisect_left

NUSCENES_ROOT = r"J:\Agent My Learning\Other\XAI Driving\data\nuscenes"
OUTPUT_DIR    = "data/train"
ACTIONS       = ["Go Straight", "Brake", "Accelerate", "Turn Left", "Turn Right"]
CANBUS_DIR    = os.path.join(NUSCENES_ROOT, "can_bus")

# 1. Changed from 'samples/' to 'sweeps/'
CAM_FOLDERS   = [
    "sweeps/CAM_FRONT",
    "sweeps/CAM_FRONT_LEFT",
    "sweeps/CAM_FRONT_RIGHT",
    "samples/CAM_FRONT",
    "samples/CAM_FRONT_LEFT",
    "samples/CAM_FRONT_RIGHT",
]

MINI_SCENES = [
    "scene-0061", "scene-0553", "scene-0655", "scene-0757",
    "scene-0796", "scene-0916", "scene-1077", "scene-1094",
    "scene-1100", "scene-1106"
]

def get_label(steering_rad, brake, brake_switch, throttle, speed):
    if brake_switch in (2, 3) or brake > 5:
        return "Brake"
    if steering_rad > 0.3:
        return "Turn Left"
    if steering_rad < -0.3:
        return "Turn Right"
    if throttle > 200 and speed > 5:
        return "Accelerate"
    return "Go Straight"

def load_scene_canbus(scene_name):
    steer_records = {}
    vm_records    = {}

    steer_path = os.path.join(CANBUS_DIR, f"{scene_name}_steeranglefeedback.json")
    if os.path.exists(steer_path):
        with open(steer_path) as f:
            data = json.load(f)
        for rec in data:
            ut  = rec.get('utime', 0)
            val = rec.get('value', 0)
            steer_records[ut] = val[0] if isinstance(val, list) else val

    vm_path = os.path.join(CANBUS_DIR, f"{scene_name}_vehicle_monitor.json")
    if os.path.exists(vm_path):
        with open(vm_path) as f:
            data = json.load(f)
        for rec in data:
            ut = rec.get('utime', 0)
            vm_records[ut] = {
                'brake'        : rec.get('brake', 0),
                'brake_switch' : rec.get('brake_switch', 1),
                'throttle'     : rec.get('throttle', 0),
                'speed'        : rec.get('vehicle_speed', 0),
                'steering_deg' : rec.get('steering', 0),
            }

    all_utimes = sorted(set(list(steer_records.keys()) + list(vm_records.keys())))
    last_vm = {'brake': 0, 'brake_switch': 1, 'throttle': 0, 'speed': 5, 'steering_deg': 0}

    records = []
    for ut in all_utimes:
        if ut in vm_records:
            last_vm = vm_records[ut]
        steering_rad = steer_records.get(ut, last_vm['steering_deg'] * 0.0175)
        records.append((
            ut, steering_rad, last_vm['brake'],
            last_vm['brake_switch'], last_vm['throttle'], last_vm['speed']
        ))
    return records

def find_nearest_fast(utime, times_list, records, max_diff=2000000):
    idx  = bisect_left(times_list, utime)
    best = None
    best_diff = max_diff

    for i in [idx - 1, idx]:
        if 0 <= i < len(records):
            diff = abs(records[i][0] - utime)
            if diff < best_diff:
                best_diff = diff
                best = records[i]
    return best

def extract_dataset():
    for action in ACTIONS:
        dst = os.path.join(OUTPUT_DIR, action)
        os.makedirs(dst, exist_ok=True)
        for f in os.listdir(dst):
            if not f.startswith('feedback_') and not f.startswith('aug_'):
                os.remove(os.path.join(dst, f))

    flat_records = []
    for scene in MINI_SCENES:
        recs = load_scene_canbus(scene)
        flat_records.extend(recs)

    flat_records.sort(key=lambda x: x[0])
    times_list = [r[0] for r in flat_records]
    print(f"\nTotal can_bus records: {len(flat_records)}")

    counts   = {a: 0 for a in ACTIONS}
    no_match = 0
    total    = 0

    for cam_folder in CAM_FOLDERS:
        cam_path = os.path.join(NUSCENES_ROOT, cam_folder)
        if not os.path.exists(cam_path):
            print(f"Skipping {cam_folder} — not found")
            continue

        images = [f for f in os.listdir(cam_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        print(f"\nProcessing {cam_folder}: {len(images)} images")

        for img_file in tqdm(images, desc=cam_folder.split('/')[-1]):
            # 2. Extract timestamp directly from filename
            # Example format: n015-2018-07-24-11-22-45+0800__CAM_FRONT__1532402927612460.jpg
            try:
                ts_str = img_file.split('__')[-1].split('.')[0]
                ts = int(ts_str)
            except (IndexError, ValueError):
                print(f"Skipping {img_file} — could not parse timestamp")
                continue

            label = "Go Straight"
            cb = find_nearest_fast(ts, times_list, flat_records)
            
            if cb:
                _, steer, brake, brake_sw, throttle, speed = cb
                label = get_label(steer, brake, brake_sw, throttle, speed)
            else:
                no_match += 1
                continue # Skip copying if we have no CAN bus data at all

            src = os.path.join(cam_path, img_file)
            dst = os.path.join(OUTPUT_DIR, label, f"{cam_folder.replace('/', '_')}_{img_file}")
            
            if not os.path.exists(dst):
                shutil.copy(src, dst)
                counts[label] += 1
                total += 1

    print(f"\n=== Extraction Complete ===")
    print(f"Total images extracted : {total}")
    print(f"Images with no CAN bus : {no_match}")
    print(f"\nClass distribution:")
    for action, count in counts.items():
        bar = '#' * (count // 50) # Scaled down bar for larger image volume
        print(f"  {action:15s}: {count:5d}  {bar}")

if __name__ == "__main__":
    extract_dataset()