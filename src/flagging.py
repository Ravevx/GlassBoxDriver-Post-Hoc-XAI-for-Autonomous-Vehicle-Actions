# src/flagging.py
import os
import cv2
import csv
import numpy as np
from datetime import datetime

LOW_TRUST_THRESHOLD = 0.5

def flag_uncertain_frames(results, session_id, frames_dir="data/flagged/review"):
    """
    results: list of dicts from analyse.py
    Each dict: {frame_id, image, action, confidence, trust_score, heatmap}
    """
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    flagged = []

    for r in results:
        reasons = []

        if r['confidence'] < 0.6:
            reasons.append("low_confidence")
        if r['trust_score'] < LOW_TRUST_THRESHOLD:
            reasons.append("low_trust")

        if reasons:
            # Save original frame
            fname = f"session{session_id}_frame{r['frame_id']:05d}.jpg"
            fpath = os.path.join(frames_dir, fname)
            cv2.imwrite(fpath, r['image'])

            # Save heatmap alongside
            hname = f"session{session_id}_frame{r['frame_id']:05d}_heatmap.jpg"
            hpath = os.path.join(frames_dir, hname)
            if r.get('heatmap') is not None:
                cv2.imwrite(hpath, r['heatmap'])

            flagged.append({
                'session_id'  : session_id,
                'frame_id'    : r['frame_id'],
                'filename'    : fname,
                'action'      : r['action'],
                'confidence'  : round(r['confidence'], 4),
                'trust_score' : round(r['trust_score'], 4),
                'flag_reason' : '+'.join(reasons)
            })

    # Save flagged log CSV
    log_path = f"logs/flagged_{session_id}.csv"
    if flagged:
        with open(log_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=flagged[0].keys())
            writer.writeheader()
            writer.writerows(flagged)

    print(f"Flagged {len(flagged)} uncertain frames - {frames_dir}")
    print(f"Log saved - {log_path}")
    return flagged
