# -*- coding: utf-8 -*-
import json, glob, sys, io
try: sys.stdout.reconfigure(encoding='utf-8')
except: pass

files = sorted(glob.glob("dataset/A_*.json"))
print(f"총 {len(files)}개 A 파일\n")

for f in files[:3]:  # 처음 3개만 상세
    with open(f, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    label = data.get("label", "?")
    strokes = data.get("strokes", [])
    total_pts = sum(len(s) for s in strokes)
    print(f"[{f}]")
    print(f"  label: {label}")
    print(f"  strokes: {len(strokes)}개, 총 {total_pts}포인트")
    for i, s in enumerate(strokes):
        if s:
            keys = sorted(s[0].keys())
            print(f"    획 {i}: {len(s)}pt | 키: {keys}")
    # extract_features 호환성 테스트
    from airwriting_imu.core.ai_model import extract_features
    feat = extract_features(strokes)
    print(f"  extract_features: shape={feat.shape}, dtype={feat.dtype}")
    print(f"  is_new_stroke 위치: {list(feat[:, 4].nonzero()[0])}")
    print()

# 전체 요약
print("=== 전체 요약 ===")
for f in files:
    with open(f, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    strokes = data.get("strokes", [])
    pts = sum(len(s) for s in strokes)
    print(f"  {data.get('label','?')} | {len(strokes)}획 | {pts}pt | {f.split('/')[-1]}")
