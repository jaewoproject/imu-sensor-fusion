"""Save meta.pkl to match the trained model"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pickle, json, os, glob
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path

DATASET_DIR = "dataset"
files = sorted(glob.glob(os.path.join(DATASET_DIR, "*.json")))
labels_set = set()
all_raw = []

for f in files:
    try:
        with open(f, 'r', encoding='utf-8') as fp:
            data = json.load(fp)
        if isinstance(data, dict) and "strokes" in data:
            label = data.get("label", "?").upper()
            strokes = data["strokes"]
        else:
            label = os.path.basename(f).split('_')[0].upper()
            strokes = [data]
        labels_set.add(label)
        
        # Quick feature extraction for scaler
        pts = []
        lx, ly = None, None
        for st in strokes:
            for pi, pt in enumerate(st):
                cx, cy = pt.get('x',0), pt.get('y',0)
                dx = (cx-lx) if lx is not None else 0
                dy = (cy-ly) if ly is not None else 0
                ns = 1.0 if pi==0 else 0.0
                pts.append([cx,cy,dx,dy,ns,pt.get('ax',0),pt.get('ay',0),pt.get('az',0),pt.get('gx',0),pt.get('gy',0),pt.get('gz',0)])
                lx, ly = cx, cy
        if len(pts) > 5:
            arr = np.array(pts, dtype=np.float32)
            arr[:,0] -= arr[0,0]
            arr[:,1] -= arr[0,1]
            all_raw.append(arr)
    except:
        pass

label_map = {l: i for i, l in enumerate(sorted(labels_set))}
scaler = StandardScaler()
scaler.fit(np.vstack(all_raw))

with open("weights/meta.pkl", "wb") as f:
    pickle.dump({"label_map": label_map, "scaler": scaler, "model_type": "pure_bilstm"}, f)

print(f"Saved meta.pkl: {label_map}")
