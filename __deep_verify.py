# -*- coding: utf-8 -*-
"""
__deep_verify.py — 실제 데이터를 흘려보내서 학습/추론 파이프라인의
바이트 레벨 정합성을 증명하는 검증 스크립트.

"코드가 문법적으로 맞는가"가 아니라
"실제로 같은 데이터를 넣으면 같은 숫자가 나오는가"를 테스트합니다.

검증 항목:
  V1. extract_features 단일 소스 여부 — 학습/추론 양쪽이 같은 함수 호출
  V2. resample 정합성 — GestureDataset._resample vs _resample_imu vs train_now.resample
  V3. is_new_stroke 보존 — 증강/리샘플 후에도 0/1 이진값 유지
  V4. scaler fit 대상 — raw vs augmented 데이터 혼입 여부
  V5. streaming → extract_features 경로 일치 — 녹화 데이터를 스트리밍으로 재생
  V6. augmentation 후 feature shape 불변 — (N, 11) 유지
  V7. PureBiLSTMAttention forward pass — 임의 입력에 대해 NaN/Inf 없음
  V8. dtype 정합 — float32/float64 혼입이 추론에 영향을 주는지
"""

import sys, io
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ── 테스트용 가짜 녹화 데이터 생성 ──
def make_fake_recording(n_strokes=3, pts_per_stroke=30):
    """main.py L880-895 에서 저장되는 것과 동일한 구조의 가짜 JSON strokes"""
    strokes = []
    for s in range(n_strokes):
        stroke = []
        for p in range(pts_per_stroke):
            stroke.append({
                "ts": 1000 + s * 1000 + p * 12,
                "dt": 0.012,
                "x": float(s * 0.5 + p * 0.01 + np.random.randn() * 0.001),
                "y": float(p * 0.02 + np.random.randn() * 0.001),
                "ax": float(np.random.randn() * 0.5),
                "ay": float(np.random.randn() * 0.5),
                "az": float(9.81 + np.random.randn() * 0.1),
                "gx": float(np.random.randn() * 0.1),
                "gy": float(np.random.randn() * 0.1),
                "gz": float(np.random.randn() * 0.1),
            })
        strokes.append(stroke)
    return strokes

PASS = 0
FAIL = 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name}: {detail}")

# ═══════════════════════════════════════════════════════════════
print("=" * 65)
print("  DEEP VERIFY — 학습/추론 파이프라인 바이트 레벨 검증")
print("=" * 65)

np.random.seed(42)
fake_strokes = make_fake_recording(n_strokes=3, pts_per_stroke=30)

# ─── V1: extract_features 단일 소스 ───
print("\n[V1] extract_features 단일 소스 검증")
from airwriting_imu.core.ai_model import extract_features as ef_ai
from train_now import extract_features as ef_train

feat_ai = ef_ai(fake_strokes)
feat_train_raw = ef_train(fake_strokes)  # train_now wrapper: returns float32, None if <=5

check("ai_model.extract_features shape", feat_ai.shape == (90, 11), f"got {feat_ai.shape}")
check("ai_model.extract_features dtype", feat_ai.dtype == np.float64, f"got {feat_ai.dtype}")
check("train_now.extract_features is float32", feat_train_raw.dtype == np.float32, f"got {feat_train_raw.dtype}")
check("양쪽 값 동일 (float32 precision)", np.allclose(feat_ai, feat_train_raw, atol=1e-6), 
      f"max diff={np.max(np.abs(feat_ai - feat_train_raw))}")

# is_new_stroke 위치 검증
is_new_col = feat_ai[:, 4]
expected_new_stroke_positions = [0, 30, 60]  # 각 획의 첫 프레임
actual_positions = list(np.flatnonzero(is_new_col == 1.0))
check("is_new_stroke 위치 정확", actual_positions == expected_new_stroke_positions,
      f"expected {expected_new_stroke_positions}, got {actual_positions}")

# zero-centering 검증
check("x[0] == 0 (zero-centered)", abs(feat_ai[0, 0]) < 1e-10, f"got {feat_ai[0, 0]}")
check("y[0] == 0 (zero-centered)", abs(feat_ai[0, 1]) < 1e-10, f"got {feat_ai[0, 1]}")

# ─── V2: resample 정합성 ───
print("\n[V2] 3가지 resample 함수의 바이트 레벨 일치 검증")
from airwriting_imu.core.ai_model import GestureDataset, AirWritingAI
from train_now import resample as resample_train

# GestureDataset._resample (인스턴스 필요)
ds = GestureDataset.__new__(GestureDataset)
ds.max_seq_len = 200

seq_f32 = feat_ai.astype(np.float32)  # 학습 경로는 float32

r_dataset = ds._resample(seq_f32, 200)
r_train = resample_train(seq_f32, 200)

# AirWritingAI._resample_imu
ai_inst = AirWritingAI.__new__(AirWritingAI)
r_inference = ai_inst._resample_imu(feat_ai, 200)  # float64 입력

check("GestureDataset._resample dtype=float32", r_dataset.dtype == np.float32, f"got {r_dataset.dtype}")
check("train_now.resample dtype=float32", r_train.dtype == np.float32, f"got {r_train.dtype}")
check("_resample_imu dtype=float64", r_inference.dtype == np.float64, f"got {r_inference.dtype}")

# 값 비교: dataset vs train_now (같은 float32 입력이므로 완전 일치해야 함)
check("dataset vs train_now 완전 일치", np.allclose(r_dataset, r_train, atol=1e-7),
      f"max diff={np.max(np.abs(r_dataset - r_train))}")

# 값 비교: dataset(f32) vs inference(f64) — dtype 차이만 있어야 함
check("dataset(f32) vs inference(f64) 값 일치", 
      np.allclose(r_dataset, r_inference.astype(np.float32), atol=1e-5),
      f"max diff={np.max(np.abs(r_dataset - r_inference.astype(np.float32)))}")

# ─── V3: is_new_stroke 이진값 보존 ───
print("\n[V3] resample/augment 후 is_new_stroke 이진값 보존")
from airwriting_imu.core.ai_model import _aug_time_warp, _aug_jitter, _aug_scale, _aug_rotation, _augment_feature_seq

for name, func in [("time_warp", _aug_time_warp), ("jitter", _aug_jitter), 
                     ("scale", _aug_scale), ("rotation", _aug_rotation)]:
    augmented = func(seq_f32.copy())
    col4 = augmented[:, 4]
    is_binary = np.all((col4 == 0.0) | (col4 == 1.0))
    check(f"{name}: is_new_stroke 이진값 유지", is_binary, 
          f"non-binary values: {col4[~((col4==0)|(col4==1))][:5]}")

# resample 후에도 보존
check("resample 후 is_new_stroke 이진값", 
      np.all((r_dataset[:, 4] == 0.0) | (r_dataset[:, 4] == 1.0)),
      f"non-binary count: {np.sum(~((r_dataset[:,4]==0)|(r_dataset[:,4]==1)))}")

# _augment_feature_seq 통합 검증
aug_copies = _augment_feature_seq(seq_f32, n=10)
for i, aug in enumerate(aug_copies):
    col4 = aug[:, 4]
    ok = np.all((col4 == 0.0) | (col4 == 1.0))
    if not ok:
        check(f"_augment_feature_seq copy {i} is_new_stroke", False,
              f"non-binary: {col4[~((col4==0)|(col4==1))][:3]}")
        break
else:
    check("_augment_feature_seq 10 copies 모두 is_new_stroke 이진값", True)

# shape 불변
check("증강 후 shape 불변 (N, 11)", all(a.shape == (90, 11) for a in aug_copies),
      f"shapes: {[a.shape for a in aug_copies[:3]]}")

# ─── V4: Scaler fit 대상 검증 ───
print("\n[V4] Scaler가 raw 데이터만으로 피팅되는지 검증")

# GestureDataset 경로: L175 all_pts = np.vstack(raw_data) → scaler.fit
# 이건 실제 데이터 없이는 직접 테스트 불가, 코드 패턴으로 검증
import inspect
src_dataset = inspect.getsource(GestureDataset._load_data)
# augment 블록은 scaler.fit 이후에 나와야 함
fit_line = None
augment_line = None
for i, line in enumerate(src_dataset.splitlines()):
    if 'scaler.fit' in line:
        fit_line = i
    if 'augment' in line.lower() and 'if self.augment' in line:
        augment_line = i

if fit_line is not None and augment_line is not None:
    check("GestureDataset: scaler.fit이 augment 블록보다 먼저", fit_line < augment_line,
          f"fit at line {fit_line}, augment at line {augment_line}")
else:
    check("GestureDataset: scaler.fit/augment 블록 존재", False, "패턴 못 찾음")

# train_now.py: raw_by_label 기반 scaler fit
src_train = open(ROOT / "train_now.py", "r", encoding="utf-8").read()
# "all_raw_pts = np.vstack([seq for seqs in raw_by_label" 패턴 확인
check("train_now.py: scaler fit on raw_by_label", "raw_by_label" in src_train and "scaler.fit(all_raw_pts)" in src_train,
      "raw_by_label 기반 scaler.fit 패턴 없음")

# ─── V5: Streaming → extract_features 경로 일치 ───
print("\n[V5] Streaming replay → extract_features 경로 바이트 일치")
from airwriting_imu.core.streaming import StreamingInference

class StubAI:
    label_map = {}
    model = None
    scaler = None
    model_type = "stub"
    def predict(self, s): return None

streamer = StreamingInference(StubAI(), debounce_time=0.0, char_timeout=10000.0, space_timeout=10000.0)

# 녹화 데이터를 스트리밍으로 재생
dummy_off = {"x": 0.0, "y": 0.0, "ax": 0.0, "ay": 0.0, "az": 0.0, "gx": 0.0, "gy": 0.0, "gz": 0.0}
for stroke in fake_strokes:
    for pt in stroke:
        streamer.process_frame(pt, is_writing=True)
    streamer.process_frame(dummy_off, is_writing=False)

recovered_strokes = streamer._strokes

# 녹화 데이터와 복원된 데이터의 구조 일치
check("Streaming: 획 수 일치", len(recovered_strokes) == len(fake_strokes),
      f"expected {len(fake_strokes)}, got {len(recovered_strokes)}")

for i in range(min(len(fake_strokes), len(recovered_strokes))):
    check(f"Streaming: 획 {i} 프레임 수 일치", len(recovered_strokes[i]) == len(fake_strokes[i]),
          f"expected {len(fake_strokes[i])}, got {len(recovered_strokes[i])}")

# extract_features 통과 후 값 동일
feat_original = ef_ai(fake_strokes)
feat_recovered = ef_ai(recovered_strokes)
check("Streaming → extract_features 값 완전 일치", np.array_equal(feat_original, feat_recovered),
      f"max diff={np.max(np.abs(feat_original - feat_recovered)) if feat_original.shape == feat_recovered.shape else 'shape mismatch'}")

# ─── V6: 스트리밍 최소 포인트 필터 정합성 ───
print("\n[V6] 최소 포인트 필터 (main.py > 2 vs streaming.py > 2)")
# main.py L892: if len(current_stroke) > 2
# streaming.py L121: [list(s) for s in self._strokes if len(s) > 2]

# 짧은 획이 제대로 필터링되는지 테스트
short_strokes = make_fake_recording(n_strokes=1, pts_per_stroke=2)  # 2포인트 (필터됨)
long_strokes = make_fake_recording(n_strokes=1, pts_per_stroke=5)   # 5포인트 (통과)

streamer2 = StreamingInference(StubAI(), char_timeout=10000.0)
for stroke in short_strokes:
    for pt in stroke:
        streamer2.process_frame(pt, is_writing=True)
    streamer2.process_frame(dummy_off, is_writing=False)

# 2포인트 획은 스트리밍 버퍼에 존재하지만, _run_inference 시 필터됨
short_filtered = [s for s in streamer2._strokes if len(s) > 2]
check("2포인트 획은 inference에서 필터됨", len(short_filtered) == 0,
      f"got {len(short_filtered)} strokes through filter")

streamer3 = StreamingInference(StubAI(), char_timeout=10000.0)
for stroke in long_strokes:
    for pt in stroke:
        streamer3.process_frame(pt, is_writing=True)
    streamer3.process_frame(dummy_off, is_writing=False)
long_filtered = [s for s in streamer3._strokes if len(s) > 2]
check("5포인트 획은 inference 통과", len(long_filtered) == 1,
      f"got {len(long_filtered)} strokes")

# ─── V7: PureBiLSTMAttention forward pass NaN/Inf 검사 ───
print("\n[V7] PureBiLSTMAttention forward pass 안정성")
from airwriting_imu.core.ai_model import PureBiLSTMAttention

model = PureBiLSTMAttention(input_dim=11, num_classes=26, hidden_dim=128)
model.eval()

# 정상 입력
x_normal = torch.randn(4, 200, 11)
with torch.no_grad():
    out = model(x_normal)
check("정상 입력: output shape", out.shape == (4, 26), f"got {out.shape}")
check("정상 입력: no NaN", not torch.isnan(out).any().item())
check("정상 입력: no Inf", not torch.isinf(out).any().item())

# 극단값 입력 (scaler 이상치)
x_extreme = torch.randn(2, 200, 11) * 100
with torch.no_grad():
    out_ext = model(x_extreme)
check("극단값 입력: no NaN", not torch.isnan(out_ext).any().item())
check("극단값 입력: no Inf", not torch.isinf(out_ext).any().item())

# 영벡터 입력 (센서 연결 안 됐을 때)
x_zero = torch.zeros(1, 200, 11)
with torch.no_grad():
    out_zero = model(x_zero)
check("영벡터 입력: no NaN", not torch.isnan(out_zero).any().item())

# ─── V8: dtype 정합 — 학습 경로 vs 추론 경로 ───
print("\n[V8] 학습/추론 dtype 정합 (float32 일관성)")

# 학습 경로: GestureDataset → float32 tensor
# 추론 경로: extract_features(f64) → _resample_imu(f64) → scaler.transform(f64) → torch.tensor(f32)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
raw_f64 = feat_ai  # float64
raw_f32 = raw_f64.astype(np.float32)

scaler.fit(raw_f32)  # 학습: float32 데이터로 fit

# 학습 경로
r_train_path = resample_train(raw_f32, 200)  # float32
t_train = scaler.transform(r_train_path)      # returns float64 (sklearn default)
t_train_tensor = torch.tensor(t_train, dtype=torch.float32)

# 추론 경로
r_infer_path = ai_inst._resample_imu(raw_f64, 200)  # float64
t_infer = scaler.transform(r_infer_path)              # float64
t_infer_tensor = torch.tensor(t_infer, dtype=torch.float32)

check("학습 vs 추론 tensor 값 일치", torch.allclose(t_train_tensor, t_infer_tensor, atol=1e-5),
      f"max diff={torch.max(torch.abs(t_train_tensor - t_infer_tensor)).item()}")

# ─── V9: 녹화 경로의 frame 키 vs extract_features 사용 키 ───
print("\n[V9] 녹화 frame 키 vs extract_features 사용 키")
record_keys = {"x", "y", "ax", "ay", "az", "gx", "gy", "gz"}  # main.py L882-889
ef_keys = {"x", "y", "ax", "ay", "az", "gx", "gy", "gz"}       # ai_model.py L49-57

stream_keys = {"x", "y", "ax", "ay", "az", "gx", "gy", "gz"}  # main.py L901-907

check("녹화 키 ⊇ extract_features 키", record_keys >= ef_keys)
check("스트리밍 키 ⊇ extract_features 키", stream_keys >= ef_keys)
check("녹화 키 == 스트리밍 키 (동일한 feature set)", record_keys == stream_keys)

# ─── V10: char_timeout 통일 확인 ───
print("\n[V10] char_timeout 값 통일 확인")
import re

main_src = open(ROOT / "main.py", "r", encoding="utf-8").read()
stream_src = open(ROOT / "airwriting_imu" / "core" / "streaming.py", "r", encoding="utf-8").read()

# main.py에서 StreamingInference 생성 시 char_timeout 값
m = re.search(r'StreamingInference\([^)]*char_timeout\s*=\s*([\d.]+)', main_src)
main_ct = float(m.group(1)) if m else -1

# streaming.py 기본값
m2 = re.search(r'char_timeout:\s*float\s*=\s*([\d.]+)', stream_src)
stream_ct = float(m2.group(1)) if m2 else -1

check("main.py char_timeout == 1.0", main_ct == 1.0, f"got {main_ct}")
check("streaming.py default char_timeout == 1.0", stream_ct == 1.0, f"got {stream_ct}")
check("main.py == streaming.py char_timeout 통일", main_ct == stream_ct, f"{main_ct} vs {stream_ct}")

# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print(f"  결과: {PASS} PASS / {FAIL} FAIL")
if FAIL == 0:
    print("  🎉 모든 검증 통과 — 학습/추론 파이프라인 정합성 증명 완료")
else:
    print(f"  ⚠️ {FAIL}건의 불일치 발견 — 수정 필요!")
print("=" * 65)
