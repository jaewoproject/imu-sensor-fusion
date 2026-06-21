# AirWriting Autonomous Development Plan

> Updated each cron-fire. Read top-to-bottom; pick the highest-priority item that is not blocked, execute, then update LOG and re-rank.

## Current state (2026-05-17)

- **Dataset:** 86 files (A=45, B=41). 711 corrupt 1-stroke files (C-H) deleted. User will re-record.
- **Pipeline integrity:** `verify_pipeline.py` T1/T2/T3/T4 all PASS (15/15) — flatten/streaming/scaler are byte-identical between training and inference paths.
- **Code hygiene:** 36/36 .py files compile clean. 3 actual runtime bugs fixed (undefined `log()` in cnn_bilstm fallback, missing `global _diag_frame_counter`, broken multiline f-string in test_ws.py). All 5 stroke→feature flattening duplications collapsed into one `extract_features()` helper in `ai_model.py`.
- **Training smoke:** `train_now.py` runs cleanly on 86-file A/B corpus → 100% val acc in 8 epochs. Weights + meta.pkl saved.
- **Model baseline (pure_bilstm, 2 classes):** PyTorch raw model 11.66 ms/inference CPU, end-to-end `predict()` 15.67 ms. ONNX export 218.5 KiB, ONNX-runtime CPU 9.93 ms (~15% faster than PyTorch). PyTorch dynamic INT8 quantization is **a trap** for LSTM — 12x slower per call due to per-step pack/unpack. For Jetson, use TensorRT FP16 from the existing ONNX.
- **Hardware ground truth:** user's live demo before cleanup → `ABCDEFGH` → recognized as `A EGEEE HE` (only A and E correct). Root cause: 1-stroke training data vs multi-stroke inference input. Mitigation: `DEBOUNCE_FRAMES` 6→2 in `main.py:124`. Re-test required after user re-records C-H.
- **Autonomous loop:** session-only cron `4af72c02` firing at :07/:37 each hour with `<<autonomous-loop>>` sentinel. 7-day auto-expiry.

## Priorities (ranked)

| # | Item | Status | Blocked by |
|---|------|--------|------------|
| 1 | Project-wide static error sweep | ✅ done — 3 real bugs fixed; 36/36 compile | — |
| 2 | Unify `extract_features` across `ai_model.py` × 4 + `train_now.py` × 1 | ✅ done — single helper, verify_pipeline 15/15 | — |
| 3 | Streaming concurrency audit | ✅ done — only `_current_sentence` race exists, see Risks below; not worth fixing | — |
| 4 | Smoke-test `train_now.py` end-to-end | ✅ done — 100% val acc on A/B/8 epochs | — |
| 5 | CTC direction (continuous text recognition) | pending — write a dev plan section below | user re-records >2 letters before any code work |
| 6 | ONNX export + latency baseline | ✅ done — see baseline numbers above | — |
| 7 | Retrain pure_bilstm + live re-test once user re-records | pending | user re-records C-H |
| 8 | Add held-out 80/20 validation split inside `ai_model._train_sequence` + save on best Val (not train) | ✅ done — seed=42 deterministic split, val pass each epoch, weights saved on val improvement | — |
| 9 | Wire `_is_stroke_start` field in streamer/recorder OR remove the dead fallback | ✅ done as side effect of P2 — no Python code references it anymore | — |
| 10 | Add data augmentation to `_train_sequence` (currently only `train_now.py` augments) | pending | — |
| 11 | T5 live-recognition simulation in verify_pipeline.py | ✅ done — 10/10 on A/B; ready as regression gate for re-records | — |
| 12 | Cosmetic pyflakes sweep on ML files (algorithm modules excluded per hands-off policy) | ✅ done — 6 unused imports cleaned, 36/36 compile, 16/16 verify | — |
| 13 | ~~Audit IMU filter modules~~ | ❌ off-limits per `feedback_algorithm_hands_off.md` — production filters are tuned-against-hardware, do not modify | n/a |
| 14 | Per-stroke resample (50 pts/stroke, concatenate) to make `is_new_stroke` more discriminative for variable stroke counts | pending — research | post-#7 |

## Research notes

Items here are ideas under evaluation. Move to Priorities when concrete.

- **Stroke-boundary embedding**: `is_new_stroke` is a binary feature today. An alternative is a learned embedding (one of two vectors, summed into the position embedding). Cleaner gradient flow, but adds parameters. Decide after #7.
- **CTC for continuous words**: `ctc_model.py` already exists. The training synth (`CTCDataset` with `mode="word"`) generates pseudo-words from single-character samples. With only A and B currently, synth words are useless — CTC dev is blocked until dataset covers more letters.
- **Per-stroke time-warping**: each stroke could be resampled independently (e.g., 50 points per stroke) so number-of-strokes is preserved but per-stroke length is normalized. Today's `_resample_imu` resamples the full sequence which compresses 3-stroke E vs 1-stroke E differently.
- **Validation split**: `_train_sequence` reports train accuracy only. No held-out split → overfitting unmeasured. Add a 20% held-out validation set.

## Risks / known issues

- `streaming.py:128-133`: `loop.run_in_executor(None, ...)` from inside a non-async caller — works because of fallback try/except, but the `_inference_lock` is held only inside `_do_predict_and_emit`, not around `_inference_running` flag toggles. Possible double-fire under heavy load.
- `ai_model.py:818, 941`: `pt.get("_is_stroke_start", pi == 0)` reads a field that is never written by the recorder or streamer. Effectively dead code, falls back to `pi == 0`. Either wire it in `streaming.py` (recommended) or remove.
- `main.py:62`: `streamer = StreamingInference(ai, char_timeout=0.8, ...)` overrides the constructor default of 0.6. Inconsistent with `StreamingInference.__init__` doc. Pick one.

## Iteration LOG

### 2026-05-17 ~03:50 (initial)
- Read main.py / streaming.py / ai_model.py / sample dataset.
- Diagnosed: training data has 1-stroke C-H (DEBOUNCE bug), inference has 3-stroke. Distribution mismatch — model collapses to nearest 1-stroke shape.
- Scanned all 797 files. A=46, B=41, C=67, D=42, E=188, F=98, G=127, H=188. Confirmed C-H are 100% single-stroke.

### 2026-05-17 ~04:00 (cleanup + verify)
- Deleted 711 corrupt files. 7.3 MB freed. 86 files remain (A=45, B=41).
- Lowered DEBOUNCE_FRAMES 6 → 2.
- Wrote `verify_pipeline.py` with T1 (flatten consistency), T2 (streaming roundtrip), T3 (end-to-end logits), T4 (scaler persistence).
- Ran on 5 A-samples: 15/15 PASS, 1/1 PASS on T4. Pipeline integrity proven.

### 2026-05-17 ~13:07 (iteration-7: cron-fire — dt audit, dataset unchanged)
- Cron `3c861f1b` fired with `<<autonomous-loop>>` sentinel. Verified system still healthy: verify_pipeline **18/18 PASS**, dataset still 86 files (A=45, B=41) — user has not re-recorded yet.
- **`dt` field audit:** `main.py:883` saves `dt` (actual packet-interval) into every recorded point. `extract_features` does NOT consume `dt` — only x/y/dx/dy/is_new_stroke + 6 IMU channels. `data_augmentor._speed_variation` mutates dt as part of stroke-dict-level augmentation but the trained model never sees that channel. So `dt` is implicitly encoded in `dx`/`dy` (user writes fast → larger dx/dy, slow → smaller). Not a bug; documented for future research.
- **Research note (not actionable yet):** adding `dt` as a 12th feature channel could give CTC explicit speed signal (helps "blank vs stroke" distinction). Trade-off: weights would need retraining, ONNX shape change, T1/T2/T3 of `verify_pipeline.py` would need updating. **Defer until C-H come back** — can't A/B test with 2 classes.
- No code changes this iteration — system in a steady state pending user re-record.

### 2026-05-17 ~07:30 (iteration-6: T7 ONNX gate + self-test the scanner)
- **T7 ONNX export added to verify_pipeline.** Round-trip: PyTorch model → `ai.export_onnx(tmp)` → file size check → `onnxruntime.InferenceSession.run(...)` → verify output shape. Passes for production `pure_bilstm`: 218.6 KiB on disk, ORT produces `(1, 2)`. **`verify_pipeline.py` now 18/18 PASS.**
- **`dataset_quality_scan.py` self-tested:** inject a corrupted JSON into a tmp dir (1-stroke E with 4 points) and confirm the scanner flags both the multi-stroke violation and the too-few-points violation. Scanner correctness verified.

### 2026-05-17 ~07:00 (iteration-5: deadcode + sub-module audit)
- **Deadcode identified, NOT deleted:**
  - `airwriting_imu/core/foundation_model.py` — TartanIMU-inspired model, **zero imports anywhere** in the project. Compiles cleanly. Kept for future re-experimentation; documented here so future iterations don't waste time auditing it as part of an active path.
  - `airwriting_imu/core/contrastive_learning.py` — auxiliary text branch, **zero imports anywhere**. Same disposition as above.
- **Used sub-modules verified:**
  - `iam_dataset.IAMDatasetLoader` — gracefully reports `available=False` with download URL when `data/iam/` is missing. JW v1 / studio_init paths fall back without it.
  - `trajectory_renderer.TrajectoryRenderer(size=64)` — renders one JSON sample to a 64×64 float32 grid (range 0.0–0.976); `render_dataset('dataset')` processes all 86 files cleanly.

### 2026-05-17 ~06:30 (iteration-4: training-path matrix)
- **Smoke-tested ALL 6 model types** end-to-end on the cleaned A/B dataset:
  - `pure_bilstm` (production): Val 100%
  - `transformer`: Val 100% in 16.5s
  - `fastkan`: Val 64.7% in 0.9s (extremely fast, weaker accuracy)
  - `jw_v2` (Continuous Mamba): Val 100% in 195s (slow on CPU but works)
  - `jw_v1` (image+IMU hybrid): 84.9% in 42.8s
  - `ctc`: 5 batches in 13.6s (~2.7s/batch — full epoch ~130s)
- **Found and fixed 2 real bugs**:
  - `_train_sequence` instantiated `JWv2_Continuous(num_classes=...)` without passing `input_dim`. JW v2 defaulted to `input_dim=8` but `extract_features` produces 11. Now passes `input_dim=11`. (`ai_model.py:601-604`)
  - `JWv1.DualModalVQTokenizer` hardcoded `MotionEncoder(in_channels=8, ...)` and `MotionDecoder(out_channels=8, ...)`. Added `in_channels` parameter (default 8 for back-compat), `_train_jw_v1` now passes 11. (`jw_v1.py:210-214, 416-428` + `ai_model.py:362-369`)
- **6th `extract_features` duplication killed** — `ctc_dataset.py:_extract_features` was a near-identical copy. Now it imports from `ai_model.extract_features` and converts to float32. Single source of truth across the entire codebase. CTC dataset still loads 86 samples + 10 synthetic words.
- **Added T6 Training-Matrix test** to `verify_pipeline.py`. Instantiates each model type with `input_dim=11, num_classes=2` and runs one forward pass on a synthetic `[1, 200, 11]` batch. **All 6/6 instantiate cleanly.** This is the regression gate that would have caught the input_dim bugs.
- **Production `pure_bilstm` restored** via `train_now.py` after smoke runs overwrote `meta.pkl` and weights. `verify_pipeline.py` now **17/17 PASS** (T1×5, T2×5, T3×5, T4 scaler, T5 live A/B 100%, T6 training-matrix 6/6).
- **DataAugmentor verified** — `aug.augment_sample(strokes, n=3)` produces 3 valid augmented stroke-dict structures from a real JSON sample.

### 2026-05-17 ~05:30 (iteration-3 wrap-up)
- **dataset_quality_scan.py** added at repo root. Reports per-label stroke-count and point-count distribution, flags suspicious files (too few/many points, multi-stroke letters saved as 1 stroke, sensor-frozen strokes where ≥60% of consecutive (x,y) pairs are identical). Runs clean on the 86 cleaned files. **Use after re-recording** to catch any new corruption before training.
- **P10 augmentation in `_train_sequence`:** evaluated, deferred. Reason: `train_now.py` (CLI) already augments and is the canonical training path; `_train_sequence` (websocket) handles class imbalance via inverse-frequency CrossEntropyLoss weights. Adding stroke-level augmentation here would create a cross-module dependency on `train_now.py`'s functions or duplicate them. Net value too low to justify.
- **Algorithm hands-off policy applied retroactively** — review of `eskf_filter.py` and other IMU modules removed from the priorities table (item 13 struck through). Pyflakes warnings in those files are now treated as expected noise per memory.

### 2026-05-17 ~05:00 (P8 + iteration-2 burndown)
- **P8 validation split:** `_train_sequence` now uses an 80/20 held-out split (seed=42, deterministic). Per-class accuracy is reported as `Tx/Vy` (train/val). Best weights are saved on best val-acc, not train-acc — overfitting blocker. 5-epoch smoke on A/B → Train 100% / Val 100% (split was 69 train / 17 val).
- **Hands-off memory entry** added (`feedback_algorithm_hands_off.md`) listing modules whose internals are off-limits even under static-analysis hints: madgwick, eskf, calibration, yaw_stabilizer, one_euro, ray_caster, bio_kinematics, packet_parser, time_sync. Two earlier touches (deleting Madgwick's `f1/f2/f3` unused locals, removing calibration's redundant `Rotation` import) were reverted.
- **ML-side cosmetic pyflakes pass** (algorithm modules untouched): removed unused imports in `data_augmentor.py`, `iam_dataset.py`, `config_loader.py`, `contrastive_learning.py`, `foundation_model.py`, `jw_v1.py`, `ctc_model.py`. 36/36 compile, verify_pipeline still PASS.
- **Augmentation audit:** `data_augmentor.DataAugmentor` (stroke-dict level, used by `_train_jw_v1`) and `train_now.py` augment_* functions (feature-matrix level) operate at different pipeline stages. Not duplication — left as-is.
- **verify_pipeline.py T5 — live-recognition test:** new test mode that replays each JSON through `StreamingInference` and runs `ai.predict` end-to-end (the same code path the live server uses). Reports per-class confusion. **A:5/5 ✓, B:5/5 ✓, 10/10 = 100%** on the 2-class model. Once C-H come back, T5 will be the regression gate for "ABCDEFGH live test".

### 2026-05-17 ~04:15 (P1-P6 burndown)
- **P1:** Compiled all 36 .py files. Pyflakes flagged 3 real bugs (`ai_model.py:765,769` undefined `log()`, `main.py:606-608` missing `global _diag_frame_counter`, `test_ws.py` broken f-string). All fixed.
- **P2:** Added `extract_features()` at `ai_model.py:30-62` and replaced 4 call sites (`GestureDataset._load_data`, `_predict_legacy`, `_predict_jw_v1`, `_predict_ctc`) plus the 5th copy in `train_now.py`. Single source of truth. verify_pipeline.py still 15/15 PASS.
- **P3:** Audited `streaming.py:62-188`. `_inference_running` race is not real (process_frame is single-threaded asyncio). The one real race is `_current_sentence` mutated from worker thread (`_do_predict_and_emit` → `_emit_char`) AND main asyncio handlers (ws_handler's erase / process_frame's space). Impact: 1-character drift on simultaneous erase+emit. Practically benign — deferred.
- **P4:** `train_now.py` runs end-to-end on 86 A/B samples → Best Val 100.0% in 8 epochs (24s on CPU). Augmentation expands 86 → 240 samples. New weights saved; verify_pipeline.py still 15/15 (T3 conf dropped 0.998 → 0.687 because model now only knows 2 classes, expected).
- **P6:** Fixed `export_onnx` — the original `dynamic_axes={"imu_sequence": {0: "batch", 1: "seq_len"}}` conflicted with torch.export's static shape inference on the LSTM/MHA stack. Set seq_len to fixed 200 (matches `_resample_imu`), only batch dynamic. ONNX = 218.5 KiB FP32. Latency: PyTorch raw 11.66 ms, PyTorch end-to-end 15.67 ms, ONNX-runtime CPU 9.93 ms. PyTorch `quantize_dynamic({LSTM, Linear}, qint8)` gives 12x slowdown (148 ms) — anti-pattern for small LSTM. For Jetson, recommend TensorRT FP16 from `weights/model.onnx`.

## CTC dev plan (P5, blocked until >2 letters)

`ctc_model.py` already exposes `CTCRecognizer(input_dim=11, hidden_dim=128, num_lstm_layers=3, num_classes=26)`. `ctc_dataset.py:CTCDataset(mode="word", synth_samples=500)` synthesizes pseudo-words by concatenating per-letter samples. Today with A and B only, synth words like "ABBA" / "BAB" are not useful training signal for real handwriting variability.

Concrete gating:
1. User re-records C-H at minimum (ideally I-Z).
2. Each letter ≥ 30 samples after the DEBOUNCE_FRAMES=2 fix so synthesis has enough variation to interpolate between.
3. Inside `_train_ctc`: add a held-out word split (synthetic words from samples user never saw) and report Character Error Rate (CER), not just per-batch accuracy.
4. Inference path needs a "continuous mode" toggle in `streaming.py` so the buffer flushes to CTC instead of the per-letter classifier — today `predict` dispatches by `self.model_type`, so loading a CTC model would route correctly, but `char_timeout` semantics break (CTC wants the entire word, not per-letter).

Defer all CTC code work until item 1 is satisfied. For now the plan section above is the spec.

<!-- Future iterations append below this line -->
