"""
AirWriting Training v3 - Quick fix: train + save model AND meta.pkl together
py -u train_now.py
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from airwriting_imu.core.ai_model import PureBiLSTMAttention, extract_features as _extract_features

DATASET_DIR = Path(__file__).parent / "dataset"
WEIGHTS_DIR = Path(__file__).parent / "weights"


def extract_features(strokes_list):
    """Thin wrapper around ai_model.extract_features: float32 + min-length guard."""
    seq = _extract_features(strokes_list)
    if len(seq) <= 5:
        return None
    return seq.astype(np.float32, copy=False)


def resample(seq, target_len=200):
    curr_len = len(seq)
    if curr_len == target_len:
        return seq
    x_old = np.linspace(0, 1, curr_len)
    x_new = np.linspace(0, 1, target_len)
    resampled = np.zeros((target_len, seq.shape[1]), dtype=np.float32)
    for i in range(seq.shape[1]):
        if i == 4:
            # is_new_stroke은 이진 플래그. 선형보간하면 획경계가 0.3/0.7 같은
            # 중간값으로 뭉개진다. nearest-neighbor로 0/1을 보존해 추론 경로
            # (_resample_imu / GestureDataset._resample)와 정합성을 맞춘다.
            nn_idx = np.round(x_new * (curr_len - 1)).astype(int)
            resampled[:, i] = seq[nn_idx, i]
        else:
            resampled[:, i] = np.interp(x_new, x_old, seq[:, i])
    return resampled


def augment_time_warp(seq, sigma=0.2):
    T = len(seq)
    warp = np.cumsum(np.abs(np.random.randn(T) * sigma) + 1.0)
    warp = warp / warp[-1] * (T - 1)
    new_seq = np.zeros_like(seq)
    for i in range(seq.shape[1]):
        if i == 4:
            # is_new_stroke 이진 플래그 보존. 각 출력 위치에 대응하는 원본
            # 인덱스를 nearest-neighbor로 골라 0/1을 유지(선형보간 시 획경계 손상).
            src = np.clip(np.round(np.interp(np.arange(T), warp, np.arange(T))).astype(int), 0, T - 1)
            new_seq[:, i] = seq[src, i]
        else:
            new_seq[:, i] = np.interp(np.arange(T), warp, seq[:, i])
    return new_seq


def augment_jitter(seq, sigma=0.03):
    noise = np.random.randn(*seq.shape).astype(np.float32) * sigma
    noise[:, 4] = 0.0
    return seq + noise


def augment_scale(seq, low=0.8, high=1.2):
    sx = np.random.uniform(low, high)
    sy = np.random.uniform(low, high)
    out = seq.copy()
    out[:, 0] *= sx; out[:, 1] *= sy; out[:, 2] *= sx; out[:, 3] *= sy
    return out


def augment_rotation(seq, max_deg=15):
    angle = np.radians(np.random.uniform(-max_deg, max_deg))
    c, s = np.cos(angle), np.sin(angle)
    out = seq.copy()
    x, y = out[:, 0].copy(), out[:, 1].copy()
    out[:, 0] = x * c - y * s; out[:, 1] = x * s + y * c
    dx, dy = out[:, 2].copy(), out[:, 3].copy()
    out[:, 2] = dx * c - dy * s; out[:, 3] = dx * s + dy * c
    return out


def augment_sample(seq, n=5):
    augmented = []
    for _ in range(n):
        aug = seq.copy()
        if np.random.rand() > 0.3: aug = augment_time_warp(aug, sigma=0.2)
        if np.random.rand() > 0.3: aug = augment_jitter(aug, sigma=0.03)
        if np.random.rand() > 0.3: aug = augment_scale(aug, 0.85, 1.15)
        if np.random.rand() > 0.5: aug = augment_rotation(aug, max_deg=12)
        augmented.append(aug)
    return augmented


def main():
    print("=" * 60)
    print("  AirWriting Training v3 (Correct Scaler)")
    print("=" * 60)

    files = sorted(DATASET_DIR.glob("*.json"))
    raw_by_label = {}
    for f in files:
        try:
            with open(f, 'r', encoding='utf-8') as fp:
                data = json.load(fp)
            if isinstance(data, dict) and "strokes" in data:
                strokes = data["strokes"]
                label = data.get("label", "?").upper()
            else:
                strokes = [data]
                label = f.stem.split('_')[0].upper()
            feat = extract_features(strokes)
            if feat is not None:
                if label not in raw_by_label:
                    raw_by_label[label] = []
                raw_by_label[label].append(feat)
        except Exception:
            pass

    labels_sorted = sorted(raw_by_label.keys())
    label_map = {l: i for i, l in enumerate(labels_sorted)}
    num_classes = len(label_map)

    print(f"\nRaw: {sum(len(v) for v in raw_by_label.values())} samples, {num_classes} classes")

    # Augment
    max_count = max(len(v) for v in raw_by_label.values())
    target_per_class = max(max_count, 120)  # Reduced to 120 for speed

    augmented_data = []
    for label in labels_sorted:
        originals = raw_by_label[label]
        idx = label_map[label]
        for seq in originals:
            augmented_data.append((seq, idx))
        deficit = target_per_class - len(originals)
        if deficit > 0:
            aug_per = max(1, deficit // len(originals) + 1)
            gen = 0
            for seq in originals:
                if gen >= deficit: break
                for a in augment_sample(seq, n=aug_per):
                    if gen >= deficit: break
                    augmented_data.append((a, idx))
                    gen += 1
            print(f"   {label}: {len(originals)} -> {len(originals)+gen}")
        else:
            print(f"   {label}: {len(originals)}")

    print(f"\nTotal: {len(augmented_data)}")

    # CRITICAL: scaler fit on RAW data only (not augmented)
    # 증강 데이터는 jitter/noise를 포함하므로 variance가 부풀려짐.
    # 추론 시 입력은 항상 raw 데이터이므로 scaler는 raw 분포에 맞춰야 함.
    scaler = StandardScaler()
    all_raw_pts = np.vstack([seq for seqs in raw_by_label.values() for seq in seqs])
    scaler.fit(all_raw_pts)

    samples, labels = [], []
    for seq, lbl in augmented_data:
        seq_r = resample(seq, 200)
        seq_n = scaler.transform(seq_r)
        samples.append(seq_n)
        labels.append(lbl)

    X = torch.tensor(np.array(samples), dtype=torch.float32)
    Y = torch.tensor(labels, dtype=torch.long)

    # Split
    n = len(X)
    perm = torch.randperm(n)
    split = int(n * 0.85)
    train_X, train_Y = X[perm[:split]], Y[perm[:split]]
    val_X, val_Y = X[perm[split:]], Y[perm[split:]]
    print(f"   Train: {len(train_X)} | Val: {len(val_X)}")

    tc = torch.bincount(train_Y, minlength=num_classes).float().clamp(min=1)
    sw = 1.0 / tc[train_Y]
    sampler = WeightedRandomSampler(sw, num_samples=len(train_X), replacement=True)
    train_loader = DataLoader(TensorDataset(train_X, train_Y), batch_size=64, sampler=sampler) # Increased batch size to 64
    val_loader = DataLoader(TensorDataset(val_X, val_Y), batch_size=128, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PureBiLSTMAttention(num_classes=num_classes).to(device)

    counts = torch.bincount(Y, minlength=num_classes).float().clamp(min=1)
    cw = (1.0 / counts); cw = cw / cw.sum() * num_classes
    criterion = nn.CrossEntropyLoss(weight=cw.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.01)
    epochs = 8  # Quick: 8 epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"\n   Device: {device} | Epochs: {epochs}")
    print("-" * 60)

    inv_map = {v: k for k, v in label_map.items()}
    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        tloss, tc_, tt_ = 0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tloss += loss.item()
            tc_ += (out.argmax(1) == y).sum().item()
            tt_ += y.size(0)
        scheduler.step()

        model.eval()
        vc, vt = 0, 0
        pc = torch.zeros(num_classes); pt = torch.zeros(num_classes)
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                vc += (pred == y).sum().item(); vt += y.size(0)
                for c in range(num_classes):
                    m = (y == c); pt[c] += m.sum().item(); pc[c] += ((pred == y) & m).sum().item()
        val_acc = vc / max(vt, 1) * 100

        per_cls = " ".join(f"{inv_map[c]}:{pc[c]/max(pt[c],1)*100:3.0f}%" for c in range(num_classes))
        print(f"  E{epoch+1:3d}/{epochs} | Loss:{tloss/len(train_loader):.3f} | Train:{tc_/tt_*100:5.1f}% | Val:{val_acc:5.1f}% | [{per_cls}]")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            torch.save(best_state, str(WEIGHTS_DIR / "pure_bilstm.pt"))
            import pickle
            with open(WEIGHTS_DIR / "meta.pkl", "wb") as f:
                pickle.dump({
                    "label_map": label_map,
                    "scaler": scaler,
                    "model_type": "pure_bilstm"
                }, f)
            print("   -> Model and Meta saved!")

    print("-" * 60)
    print(f"\n   DONE! Best Val: {best_val_acc:.1f}%")
    print("   SAVED: pure_bilstm.pt + meta.pkl (SAME scaler)")
    print("   Restart server now.")


if __name__ == "__main__":
    main()
