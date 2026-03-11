# check_canbus.py — run this first to see structure
import os, json

CANBUS_DIR = r"J:\Agent My Learning\Other\XAI Driving\data\nuscenes\can_bus"

files = [f for f in os.listdir(CANBUS_DIR) if f.endswith('.json')]
print(f"Found {len(files)} JSON files:")
for f in files:
    print(f"  {f}")

# Show structure of first file
first = os.path.join(CANBUS_DIR, files[0])
with open(first, 'r') as f:
    data = json.load(f)

print(f"\nFile: {files[0]}")
print(f"Type: {type(data)}")

if isinstance(data, list):
    print(f"Length: {len(data)}")
    print(f"First item type: {type(data[0])}")
    print(f"First item: {json.dumps(data[0], indent=2)[:500]}")
elif isinstance(data, dict):
    print(f"Keys: {list(data.keys())}")
    # Show first key's value
    first_key = list(data.keys())[0]
    print(f"First key '{first_key}' value: {json.dumps(data[first_key], indent=2)[:500]}")
