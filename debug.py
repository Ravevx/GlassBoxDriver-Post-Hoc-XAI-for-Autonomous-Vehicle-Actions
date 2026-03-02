import os, json, numpy as np

root = os.path.join("data", "nuscenes", "v1.0-mini")

with open(os.path.join(root, "ego_pose.json")) as f:
    ego_poses = {r["token"]: r for r in json.load(f)}

with open(os.path.join(root, "sample_data.json")) as f:
    sample_data = [r for r in json.load(f) if "CAM_FRONT" in r["filename"] and r["is_key_frame"]]

# Print first 10 consecutive deltas
print("dx     | dy     | speed  | yaw    | move_angle | diff")
print("-" * 65)
for i in range(min(10, len(sample_data)-1)):
    curr = np.array(ego_poses[sample_data[i]["ego_pose_token"]]["translation"])
    nxt  = np.array(ego_poses[sample_data[i+1]["ego_pose_token"]]["translation"])
    delta = nxt - curr
    dx, dy = delta[0], delta[1]
    speed = np.linalg.norm(delta[:2])

    rot = ego_poses[sample_data[i]["ego_pose_token"]]["rotation"]
    w, x, y, z = rot
    yaw = np.degrees(np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z)))
    move_angle = np.degrees(np.arctan2(dy, dx))
    diff = (move_angle - yaw + 180) % 360 - 180

    print(f"{dx:6.3f} | {dy:6.3f} | {speed:.4f} | {yaw:6.1f} | {move_angle:10.1f} | {diff:6.1f}")
