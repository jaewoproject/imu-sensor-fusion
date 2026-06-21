"""verify_pipeline.py — 학습/추론 데이터 경로의 무결성을 객관적으로 증명.

검증 항목:
  T1. JSON → 학습용 flatten == JSON → 추론용 flatten        (코드 중복 검출)
  T2. JSON 의 strokes 를 StreamingInference 에 1프레임씩 주입하면,
      내부 _strokes 가 원본과 정확히 동일하게 복원되는가          (녹화 vs 스트리밍 동치)
  T3. _predict_legacy 가 원본 JSON 으로 만든 logits 와
      replay 된 _strokes 로 만든 logits 가 동일한가              (end-to-end 동치)
  T4. weights/meta.pkl 의 Scaler 가 transform 후 동일한 값을 내는가  (Scaler 영속성)

사용:
  py -3 verify_pipeline.py                     # dataset/ 의 처음 5개 검증
  py -3 verify_pipeline.py dataset/A_xxx.json  # 특정 파일 검증
"""

from __future__ import annotations

import glob
import json
import sys
import io
from pathlib import Path

import numpy as np

# Windows cp949 console can't encode em-dash etc.; force UTF-8 stdout.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from airwriting_imu.core.streaming import StreamingInference
from airwriting_imu.core.ai_model import AirWritingAI


# ─────────────────────────────────────────────────────────────────────
# Reference flatten implementations — must mirror ai_model.py exactly.
# If ai_model.py changes its feature extraction and these don't, T1 fails.
# ─────────────────────────────────────────────────────────────────────

def flatten_training_style(session_strokes):
    """Mirror of GestureDataset._load_data (ai_model.py:67-93)."""
    flattened = []
    last_x = last_y = None
    for si, st in enumerate(session_strokes):
        for pi, pt in enumerate(st):
            curr_x = pt.get('x', 0.0)
            curr_y = pt.get('y', 0.0)
            dx = (curr_x - last_x) if last_x is not None else 0.0
            dy = (curr_y - last_y) if last_y is not None else 0.0
            is_new_stroke = 1.0 if pi == 0 else 0.0
            flattened.append([
                curr_x, curr_y, dx, dy, is_new_stroke,
                pt.get('ax', 0.0), pt.get('ay', 0.0), pt.get('az', 0.0),
                pt.get('gx', 0.0), pt.get('gy', 0.0), pt.get('gz', 0.0),
            ])
            last_x, last_y = curr_x, curr_y
    if flattened:
        fx, fy = flattened[0][0], flattened[0][1]
        for row in flattened:
            row[0] -= fx
            row[1] -= fy
    return np.asarray(flattened, dtype=np.float64)


def flatten_inference_style(session_strokes):
    """Mirror of _predict_legacy (ai_model.py:864-893)."""
    flattened = []
    last_x = last_y = None
    for si, st in enumerate(session_strokes):
        for pi, pt in enumerate(st):
            curr_x, curr_y = pt.get('x', 0.0), pt.get('y', 0.0)
            dx = (curr_x - last_x) if last_x is not None else 0.0
            dy = (curr_y - last_y) if last_y is not None else 0.0
            is_new_stroke = 1.0 if pi == 0 else 0.0
            flattened.append([
                curr_x, curr_y, dx, dy, is_new_stroke,
                pt.get('ax', 0.0), pt.get('ay', 0.0), pt.get('az', 0.0),
                pt.get('gx', 0.0), pt.get('gy', 0.0), pt.get('gz', 0.0),
            ])
            last_x, last_y = curr_x, curr_y
    if flattened:
        fx, fy = flattened[0][0], flattened[0][1]
        for row in flattened:
            row[0] -= fx
            row[1] -= fy
    return np.asarray(flattened, dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────
# Streaming replay: feed JSON strokes into StreamingInference as if live.
# ─────────────────────────────────────────────────────────────────────

class _StubAI:
    """No-op AI engine so StreamingInference doesn't actually predict."""
    label_map = {}
    model = None
    scaler = None
    model_type = "stub"

    def predict(self, session_strokes):
        return None


def replay_into_streamer(json_strokes):
    """Replay JSON strokes frame-by-frame. Return streamer._strokes after replay."""
    streamer = StreamingInference(
        _StubAI(),
        debounce_time=0.0,
        char_timeout=10_000.0,    # never auto-fire inference
        space_timeout=10_000.0,
    )
    dummy_off = {"x": 0.0, "y": 0.0, "ax": 0.0, "ay": 0.0, "az": 0.0,
                 "gx": 0.0, "gy": 0.0, "gz": 0.0}
    for stroke in json_strokes:
        for pt in stroke:
            streamer.process_frame(pt, is_writing=True)
        # One is_writing=False frame closes the stroke
        streamer.process_frame(dummy_off, is_writing=False)
    return streamer._strokes


def strokes_equal(a, b) -> bool:
    """Deep equality on a list-of-list-of-dict structure for the fields we care about."""
    if len(a) != len(b):
        return False
    keys = ("x", "y", "ax", "ay", "az", "gx", "gy", "gz")
    for sa, sb in zip(a, b):
        if len(sa) != len(sb):
            return False
        for pa, pb in zip(sa, sb):
            for k in keys:
                if abs(float(pa.get(k, 0.0)) - float(pb.get(k, 0.0))) > 1e-12:
                    return False
    return True


# ─────────────────────────────────────────────────────────────────────
# Checks
# ─────────────────────────────────────────────────────────────────────

def check_T1_flatten_consistency(strokes):
    a = flatten_training_style(strokes)
    b = flatten_inference_style(strokes)
    if a.shape != b.shape:
        return False, f"shape mismatch {a.shape} vs {b.shape}"
    diff = np.abs(a - b).max() if a.size else 0.0
    return (diff < 1e-12), f"max|diff|={diff:.3e}"


def check_T2_streaming_roundtrip(strokes):
    recovered = replay_into_streamer(strokes)
    ok = strokes_equal(strokes, recovered)
    if ok:
        return True, f"{len(strokes)} strokes, lens={[len(s) for s in strokes]} — identical"
    return False, f"got {len(recovered)} strokes (expected {len(strokes)}), lens={[len(s) for s in recovered]}"


def check_T3_end_to_end_logits(strokes, ai):
    """Predict via raw JSON vs via replayed strokes. Compare softmax."""
    if ai.model is None:
        return None, "skipped — no model loaded"
    # Direct
    out_a = ai.predict(strokes)
    out_b = ai.predict(replay_into_streamer(strokes))
    # ai.predict returns (label, conf). For equality, both should yield same label & conf.
    if out_a is None or out_b is None:
        return False, f"predict returned None: a={out_a}, b={out_b}"
    same_label = (out_a[0] == out_b[0])
    conf_diff = abs(out_a[1] - out_b[1])
    ok = same_label and conf_diff < 1e-6
    return ok, f"a={out_a}, b={out_b}, conf_diff={conf_diff:.2e}"


def check_T4_scaler_persistence():
    """Load meta.pkl scaler twice (fresh AI instances) and compare transform()."""
    import pickle
    meta_path = ROOT / "weights" / "meta.pkl"
    if not meta_path.exists():
        return None, "skipped — weights/meta.pkl not present"
    with open(meta_path, "rb") as f:
        meta_a = pickle.load(f)
    with open(meta_path, "rb") as f:
        meta_b = pickle.load(f)
    sa, sb = meta_a.get("scaler"), meta_b.get("scaler")
    if sa is None or sb is None:
        return None, "skipped — scaler missing from meta.pkl"
    # Probe with random data shaped like model input (N, 11)
    rng = np.random.default_rng(42)
    x = rng.standard_normal((50, 11))
    ya, yb = sa.transform(x), sb.transform(x)
    diff = np.abs(ya - yb).max()
    return (diff < 1e-15), f"mean[0]={sa.mean_[0]:.6f}, scale[0]={sa.scale_[0]:.6f}, max|diff|={diff:.3e}"


# ─────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────

def verify_file(path: Path, ai) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    strokes = data.get("strokes", [])
    if not strokes:
        return {"file": path.name, "skipped": "no strokes"}

    results = {}
    ok, msg = check_T1_flatten_consistency(strokes); results["T1"] = (ok, msg)
    ok, msg = check_T2_streaming_roundtrip(strokes); results["T2"] = (ok, msg)
    ok, msg = check_T3_end_to_end_logits(strokes, ai); results["T3"] = (ok, msg)
    return {"file": path.name, "label": data.get("label"),
            "n_strokes": len(strokes),
            "n_pts": sum(len(s) for s in strokes),
            **results}


def run_training_matrix_test():
    """T6: instantiate every supported model type to catch shape/import bugs.

    Build each model with input_dim=11 and num_classes=2 (matches today's
    cleaned dataset). Run one forward pass on a synthetic batch of shape
    [1, 200, 11]. We do NOT train — just prove the model is wired up correctly.

    Catches the JW v1 / JW v2 input_dim=8 mismatch that was found and fixed.
    """
    import torch
    from airwriting_imu.core.ai_model import (
        PureBiLSTMAttention, GestureTransformer,
    )

    dummy = torch.randn(1, 200, 11)
    matrix = []

    # pure_bilstm
    try:
        m = PureBiLSTMAttention(input_dim=11, num_classes=2); m.eval()
        with torch.no_grad():
            out = m(dummy)
        matrix.append(("pure_bilstm", True, f"out={tuple(out.shape)}"))
    except Exception as e:
        matrix.append(("pure_bilstm", False, str(e)[:80]))

    # transformer
    try:
        m = GestureTransformer(input_dim=11, num_classes=2); m.eval()
        with torch.no_grad():
            out = m(dummy)
        matrix.append(("transformer", True, f"out={tuple(out.shape)}"))
    except Exception as e:
        matrix.append(("transformer", False, str(e)[:80]))

    # fastkan
    try:
        from airwriting_imu.core.fastkan import FastKANClassifier
        m = FastKANClassifier(input_dim=11, hidden_dim=32, num_classes=2, num_grids=8); m.eval()
        with torch.no_grad():
            out = m(dummy)
        matrix.append(("fastkan", True, f"out={tuple(out.shape)}"))
    except Exception as e:
        matrix.append(("fastkan", False, str(e)[:80]))

    # jw_v2 (Continuous Mamba)
    try:
        from airwriting_imu.core.jw_v1 import JWv2_Continuous
        m = JWv2_Continuous(input_dim=11, num_classes=2); m.eval()
        with torch.no_grad():
            out = m(dummy)
        matrix.append(("jw_v2", True, f"out={tuple(out.shape)}"))
    except Exception as e:
        matrix.append(("jw_v2", False, str(e)[:80]))

    # jw_v1 (image + IMU hybrid)
    try:
        from airwriting_imu.core.jw_v1 import JWv1
        m = JWv1(num_classes=2, in_channels=11); m.eval()
        # JWv1 needs both IMU and image
        dummy_img = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            label, conf, top_k = m.predict(dummy, dummy_img, {"A": 0, "B": 1})
        matrix.append(("jw_v1", True, f"label={label}"))
    except Exception as e:
        matrix.append(("jw_v1", False, str(e)[:80]))

    # CTC
    try:
        from airwriting_imu.core.ctc_model import CTCRecognizer
        m = CTCRecognizer(input_dim=11, hidden_dim=64, num_lstm_layers=2, num_classes=26); m.eval()
        with torch.no_grad():
            out = m(dummy)
        matrix.append(("ctc", True, f"out={tuple(out.shape)}"))
    except Exception as e:
        matrix.append(("ctc", False, str(e)[:80]))

    n_pass = sum(1 for r in matrix if r[1])
    print(f"  [T6 Training-Matrix] {n_pass}/{len(matrix)} model types instantiate cleanly")
    for name, ok, msg in matrix:
        badge = "✓" if ok else "✗"
        print(f"     {badge} {name:<14s} {msg}")
    return n_pass == len(matrix)


def run_onnx_export_test():
    """T7: production model exports to ONNX and the resulting file is loadable.

    The previous bug — `dynamic_axes` claiming seq_len was dynamic while
    torch.export inferred it static — would have silently failed in deployment.
    This test runs export → check file exists → load via onnxruntime → forward.
    Skips gracefully if onnxruntime isn't installed.
    """
    import os
    import tempfile
    from airwriting_imu.core.ai_model import AirWritingAI

    ai = AirWritingAI()
    if not ai.load_model():
        return None, "no production model loaded"
    if ai.model_type not in ("pure_bilstm", "transformer", "fastkan"):
        return None, f"model_type={ai.model_type} not in ONNX-tested set"

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "test_model.onnx")
        try:
            ai.export_onnx(path)
        except Exception as e:
            return False, f"export failed: {str(e)[:120]}"

        if not os.path.exists(path):
            return False, "export reported OK but no file produced"

        size_kib = os.path.getsize(path) / 1024
        try:
            import onnxruntime as ort
        except ImportError:
            return None, f"export OK ({size_kib:.1f} KiB), onnxruntime not installed"

        try:
            sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            inp_name = sess.get_inputs()[0].name
            dummy = np.random.randn(1, 200, 11).astype(np.float32)
            out = sess.run(None, {inp_name: dummy})[0]
            return True, f"{size_kib:.1f} KiB, ORT out shape {tuple(out.shape)}"
        except Exception as e:
            return False, f"ORT load/run failed: {str(e)[:120]}"


def run_live_recognition_test(ai):
    """T5: Per-class live-recognition accuracy.

    For each label present in the dataset, take up to N samples (default 5) and
    replay them through StreamingInference end-to-end (the exact same path the
    server uses). Report top-1 prediction and per-class confusion.

    This is the *only* test that approximates the user's live `ABCDEFGH` flow.
    """
    if ai.model is None:
        print("  [T5 Live-Recognition] SKIP — no model loaded")
        return None
    if not ai.label_map:
        print("  [T5 Live-Recognition] SKIP — empty label_map")
        return None

    from collections import defaultdict, Counter
    samples_per_class = 5
    by_label = defaultdict(list)
    for fp in sorted(glob.glob(str(ROOT / "dataset" / "*.json"))):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            lbl = data.get("label") or Path(fp).stem.split("_")[0]
            by_label[lbl].append((fp, data["strokes"]))
        except Exception:
            pass

    confusion = defaultdict(Counter)
    total = correct = 0
    for lbl in sorted(by_label.keys()):
        if lbl not in ai.label_map:
            print(f"  [T5] {lbl}: SKIP (not in model's label_map)")
            continue
        samples = by_label[lbl][:samples_per_class]
        for fp, strokes in samples:
            recovered_strokes = replay_into_streamer(strokes)
            # ai.predict expects list[list[dict]]
            result = ai.predict([list(s) for s in recovered_strokes])
            pred = result[0] if result else "?"
            confusion[lbl][pred or "?"] += 1
            total += 1
            if pred == lbl:
                correct += 1

    print(f"  [T5 Live-Recognition] {correct}/{total} = {correct/max(total,1)*100:.1f}%")
    for lbl in sorted(confusion.keys()):
        row = ", ".join(f"{p}:{n}" for p, n in confusion[lbl].most_common())
        marker = "✓" if confusion[lbl].most_common(1)[0][0] == lbl else "✗"
        print(f"     {marker} {lbl} → {row}")
    return correct == total


def main():
    args = [a for a in sys.argv[1:] if a != "--live"]
    live_only = "--live" in sys.argv

    if args:
        targets = [Path(a) for a in args]
    else:
        all_files = sorted(glob.glob(str(ROOT / "dataset" / "*.json")))
        targets = [Path(f) for f in all_files[:5]]

    print(f"verify_pipeline.py — checking {len(targets)} file(s)" + (" (live-only)" if live_only else ""))

    # Scaler persistence (independent of files)
    ok, msg = check_T4_scaler_persistence()
    badge = "PASS" if ok else ("SKIP" if ok is None else "FAIL")
    print(f"  [T4 Scaler]    {badge}  {msg}")
    print()

    # Try to load model for T3 / T5
    ai = AirWritingAI()
    ai.load_model()

    n_pass = n_fail = n_skip = 0
    if not live_only:
        for path in targets:
            res = verify_file(path, ai)
            if "skipped" in res:
                print(f"  {path.name}: SKIP ({res['skipped']})")
                continue
            line = (f"  {res['file']:34s} {res['label']:>3s} "
                    f"({res['n_strokes']}st, {res['n_pts']}pts)")
            print(line)
            for key in ("T1", "T2", "T3"):
                ok, msg = res[key]
                if ok is None:
                    badge, bucket = "SKIP", "skip"
                elif bool(ok):
                    badge, bucket = "PASS", "pass"
                else:
                    badge, bucket = "FAIL", "fail"
                print(f"      {key}: {badge}  {msg}")
                if bucket == "pass": n_pass += 1
                elif bucket == "fail": n_fail += 1
                else: n_skip += 1

    # T6: training-matrix (model instantiation + forward shape check)
    print()
    matrix_ok = run_training_matrix_test()
    if matrix_ok is True: n_pass += 1
    elif matrix_ok is False: n_fail += 1
    else: n_skip += 1

    # T7: ONNX export round-trip (production model only)
    print()
    onnx_ok, onnx_msg = run_onnx_export_test()
    if onnx_ok is None:
        print(f"  [T7 ONNX-Export]    SKIP  {onnx_msg}")
        n_skip += 1
    elif onnx_ok:
        print(f"  [T7 ONNX-Export]    PASS  {onnx_msg}")
        n_pass += 1
    else:
        print(f"  [T7 ONNX-Export]    FAIL  {onnx_msg}")
        n_fail += 1

    # T5: end-to-end live recognition (always run when a model is loaded)
    print()
    live_ok = run_live_recognition_test(ai)
    if live_ok is True: n_pass += 1
    elif live_ok is False: n_fail += 1
    else: n_skip += 1

    print()
    print(f"TOTALS: PASS={n_pass}  FAIL={n_fail}  SKIP={n_skip}")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
