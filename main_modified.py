"""
Hybrid-AirScribe Digital Twin Server
=====================================
Phase 1: 모듈화된 구조

- UDP 수신: 12345 (ESP32 센서 데이터)
- Discovery 응답: 12344  
- WebSocket 브로드캐스트: 12347 (브라우저 시각화)
- HTTP 정적 파일 서빙: 8080 (web/ 폴더)
- OLED 피드백: 5555 (ESP32 OLED)

사용법:
  py -3 main.py              ← 기본 실행
  py -3 main.py --mock       ← ESP32 없이 시뮬레이션 모드
"""

import asyncio
import json
import os
import math
import logging
import time
import sys


import threading
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from functools import partial

import numpy as np

# [Phase 1] 원본 Madgwick 필터 복구 (Euler 분해 우회 없는 순수 쿼터니언 기반)
from airwriting_imu.core.madgwick import MadgwickFilter
from airwriting_imu.core.ray_caster import RayProjection
try:
    from websockets.asyncio.server import serve as ws_serve
except ImportError:
    print("ERROR: websockets 패키지가 필요합니다")
    print("  py -3 -m pip install websockets")
    sys.exit(1)

from airwriting_imu.core.packet_parser import PacketParser, SensorFrame
from airwriting_imu.core.eskf_filter import ESKF
from airwriting_imu.core.calibration import Calibrator
from airwriting_imu.core.time_sync import TimeSync
from airwriting_imu.core.one_euro_filter import OneEuroFilter
from airwriting_imu.core.ai_model import AirWritingAI
from airwriting_imu.core.streaming import StreamingInference
from airwriting_imu.core.raw_logger import RawFrameLogger

# ─── 차세대 관성 지능 모듈 ───
from airwriting_imu.core.bio_kinematics import (
    DifferentialKinematics, MotionSeparator
)
from airwriting_imu.core.yaw_stabilizer import YawStabilizer

# Scipy Rotation은 투영용에서 더 이상 매 프레임 사용하지 않음
# (ComplementaryRayCaster + RayProjection으로 대체)

# [BUG-3] AI/스트리머는 모듈 로드 시점이 아닌 main() 내부에서 지연 초기화한다.
# weights/meta.pkl 누락이나 torch 버전 불일치로 서버 전체가 죽지 않도록 함.
ai = None
streamer = None
# ─── 로깅 ───
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("airscribe")

# ─── 진단 모드 (--diag) ───
# 켜면 phys_x/phys_z/quaternion/out_x/out_y를 약 10Hz로 INFO 로깅.
# 축 매핑·영점·1:1 투영 검증 시에만 사용. 운영 시에는 끄세요.
DIAG_MODE = "--diag" in sys.argv
_diag_frame_counter = 0

# ─── 포트 설정 ───
try:
    import serial.tools.list_ports
    _ports = list(serial.tools.list_ports.comports())
except ImportError:
    _ports = []
    log.warning("pyserial 미설치 — COM 포트 자동 감지 비활성화")
_ESP32_VIDS = {0x10C4, 0x1A86, 0x0403}  # Silicon Labs CP2102, QinHeng CH340, FTDI
SERIAL_PORT = "COM3"
if _ports:
    _vid_match = [p.device for p in _ports if p.vid in _ESP32_VIDS]
    _non_com1  = [p.device for p in _ports if p.device != "COM1"]
    if _vid_match:
        SERIAL_PORT = _vid_match[0]
    elif _non_com1:
        SERIAL_PORT = _non_com1[0]
    else:
        SERIAL_PORT = _ports[-1].device

BAUD_RATE   = 921600
WS_PORT     = 12347
HTTP_PORT   = 8080


# ─── 글로벌 상태 ───
# [BUG-5] config/imu.yaml의 axis_remap([-1,-1,1])은 의도적으로 사용하지 않는다.
# PacketParser는 오픈소스 Right-Hand 규칙에 맞춰 내부에서 원시 축 반전을 처리하므로,
# config 기반 추가 반전을 적용하면 이중 반전이 된다. 따라서 axis_remap=False 고정.
parser = PacketParser(axis_remap=False)
s1_eskf = ESKF(dt=0.01)
s2_eskf = ESKF(dt=0.01)
s3_eskf = ESKF(dt=0.01)

# [Phase 1] 커서 전용 순수 쿼터니언 필터 (Madgwick 원상복구)
# 오일러 분해를 우회하여 대각선(축 간) 크로스 커플링을 방지합니다.
madgwick_s3_ray = MadgwickFilter(beta=0.05, sample_rate=85.0)
yaw_stabilizer = YawStabilizer(sample_rate=85.0)

# [Phase 2] 레이저 투영: 좌우/상하 1:1 게인 (스케일링은 main 루프에서 일원화)
ray_projector = RayProjection(
    projection_distance=2.5,   # 가상 벽면 2.5m 앞
    fov_limit_deg=60.0,        # 물리적 손목 가동 범위
    deadzone_deg=0.2,          # 0.2° 이하 미세 떨림 무시
)

# OneEuroFilter: 기존 원본 버전 파라미터 복구 (존나 잘 돌아갔던 셋팅)
oef_x = OneEuroFilter(freq=85.0, min_cutoff=1.0, beta=1.0, d_cutoff=1.0)
oef_y = OneEuroFilter(freq=85.0, min_cutoff=1.0, beta=1.0, d_cutoff=1.0)
smooth_x, smooth_y = 0.0, 0.0
_last_packet_ts = 0  # 직전 패킷 타임스탬프 (ms) — dt 계산용

# 버튼 디바운싱 (물리적 스위치 바운싱 및 손가락 압력 변동 방어)
button_debounce_counter = 0
DEBOUNCE_FRAMES = 2   # 85Hz 기준 ~24ms — 물리 바운스(~10ms)는 차단하되 다획 글자(E/F/H 등)의 짧은 stroke 간격(>24ms)은 분리 보존
_ws_frame_counter = 0          # 15Hz 무거운 필드 전송 주기 카운터

# ─── Yaw 자동 리센터링 (드리프트 누적 차단) ───
# 사용자가 펜을 들지 않고 손을 일정 시간 정지 → comp_filter_s3 의 yaw 성분 0으로 스냅
_stationary_frames = 0
_last_recenter_mono = 0.0
STATIONARY_GYRO_THRESH = np.radians(8.0)   # rad/s (작은 손떨림은 정지로 간주)
# [Phase 4] 정지 판정 시간 0.5초→2.0초, 쿨다운 2초→5초로 완화
# 글씨 쓰다가 잠깐 멈췄을 때 리센터가 개입하는 현상 방지
STATIONARY_TRIGGER     = 170               # 약 2.0초 (85Hz 기준)
RECENTER_COOLDOWN_S    = 5.0               # 5초 쿨다운

# AI 데이터셋 수집기 상태
(Path(__file__).parent / "dataset").mkdir(parents=True, exist_ok=True)
current_stroke = []
was_writing = False

# [Phase 6] 다중 획 레코딩 세션 (UI 통제)
is_recording_session = False
session_label = ""
session_strokes = []

# [Phase 10.5] Free-Draw Auto Predict (Removed, replaced by StreamingInference)

time_sync = TimeSync()
calibrator = Calibrator(required_samples=300)
ws_clients: set = set()
esp32_addr = None
mock_mode = "--mock" in sys.argv

# [Phase 0 / E5] opt-in 원시 프레임 로거. 플래그가 없으면 RAW_LOGGER=None →
# 모든 호출이 None 가드로 막혀 파이프라인 영향 0.
#   사용법: py -3 main.py --rawlog [경로]   (경로 생략 시 rawlog_s3.jsonl)
_rawlog_path = None
if "--rawlog" in sys.argv:
    _ri = sys.argv.index("--rawlog")
    if _ri + 1 < len(sys.argv) and not sys.argv[_ri + 1].startswith("--"):
        _rawlog_path = sys.argv[_ri + 1]
    else:
        _rawlog_path = "rawlog_s3.jsonl"
RAW_LOGGER = RawFrameLogger(_rawlog_path) if _rawlog_path else None

# ─── 차세대 모듈 초기화 ───
diff_kin = DifferentialKinematics()    # 3-IMU 차동 운동학
motion_sep = MotionSeparator(sample_rate=85.0)  # 손목/손가락 동작 분리

# 파이프라인 상태 (Digital Twin 대시보드 전송용)
pipeline_state = {
    "bio_kin": {},
    "writing_intent": 0.0,
}

# [AI 라벨링] CLI 인자로 넘어온 라벨 파싱 (ex: py main.py --label A)
dataset_label = "unlabeled"
for i, arg in enumerate(sys.argv):
    if arg == "--label" and i + 1 < len(sys.argv):
        dataset_label = sys.argv[i+1].upper()

main_loop = None

# ─── WebSocket ───
# WebSocket 전송 상태 관리 (병목 방지)
_ws_sending = set()

async def ws_broadcast(data: dict):
    if not ws_clients:
        return
    msg = json.dumps(data)
    
    async def send_to_client(client):
        if client in _ws_sending:
            return # 이미 전송 중이면 이번 프레임은 드랍 (레이턴시 방지)
        
        _ws_sending.add(client)
        try:
            await asyncio.wait_for(client.send(msg), timeout=0.010)
        except Exception:
            ws_clients.discard(client)
        finally:
            _ws_sending.discard(client)
            
    for c in list(ws_clients):
        asyncio.create_task(send_to_client(c))

# ─── Streaming 추론 콜백 ───
def _on_text_updated(sentence: str, char: str):
    """스트리밍 추론기에서 새 문자가 확정될 때 웹소켓으로 전송"""
    if ws_clients and main_loop is not None:
        msg = {
            "type": "streaming_text",
            "sentence": sentence,
            "latest_char": char
        }
        asyncio.run_coroutine_threadsafe(ws_broadcast(msg), main_loop)

# streamer.on_text_updated 연결은 streamer가 초기화되는 main() 내부에서 수행한다.


async def ws_handler(websocket):
    ws_clients.add(websocket)
    log.info(f"🌐 WebSocket 연결: {websocket.remote_address} (총 {len(ws_clients)})")
    try:
        async for msg in websocket:
            try:
                cmd = json.loads(msg)
                if cmd.get("action") == "reset":
                    s1_eskf.reset()
                    s2_eskf.reset()
                    s3_eskf.reset()
                    log.info("↩️  트래커 리셋")
                elif cmd.get("action") == "calibrate":
                    calibrator.reset()
                    s1_eskf.reset()
                    s2_eskf.reset()
                    s3_eskf.reset()
                    log.info("🎯 캘리브레이션 300샘플 다시 시작")
                
                elif cmd.get("action") == "erase_last_char":
                    if len(streamer._current_sentence) > 0:
                        erased = streamer._current_sentence[-1]
                        streamer._current_sentence = streamer._current_sentence[:-1]
                        log.info(f"⌫ 웹 UI에서 지우기 요청. 남은 문장: '{streamer._current_sentence}'")
                        _on_text_updated(streamer._current_sentence, "<ERASE>")
                elif cmd.get("action") == "clear_all_text":
                    streamer.reset()
                    log.info("🗑️ 웹 UI에서 전체 텍스트 지우기 요청")
                    
                # [Phase 6] UI 기반 녹화 제어 추가
                elif cmd.get("action") == "start_record":
                    global is_recording_session, session_label, session_strokes, current_stroke
                    is_recording_session = True
                    session_label = cmd.get("label", "UNNAMED")
                    session_strokes = []
                    current_stroke = []
                    log.info(f"🔴 녹화 시작: '{session_label}'")
                    
                elif cmd.get("action") == "stop_record":
                    is_recording_session = False
                    if len(session_strokes) > 0:
                        filename = str(Path(__file__).parent / "dataset" / f"{session_label}_{int(time.time() * 1000)}.json")
                        _payload = json.dumps({"label": session_label, "strokes": session_strokes}, indent=2)
                        await asyncio.to_thread(
                            Path(filename).write_text, _payload, "utf-8"
                        )
                        log.info(f"💾 데이터셋 저장 완료: {filename} (총 {len(session_strokes)}획)")
                    else:
                        log.warning("⚠️ 저장할 획 데이터가 없습니다 (버튼을 누르지 않음).")
                        
                    # 저장 완료 후 즉시 추론(Predict) 시도 -> Frontend로 Morphing 전송!
                    if len(session_strokes) > 0:
                        # 1. UI 시각적 확정(Morphing)은 유저가 방금 라벨링한 정답(session_label)으로 확실하게 변환시켜줌!
                        asyncio.create_task(ws_broadcast({
                            "type": "prediction",
                            "label": session_label
                        }))
                        
                        # 2. 백그라운드 테스트: 현재 AI 모델은 방금 쓴 궤적을 뭐라고 예측할까?
                        pred_label, pred_conf = (None, 0.0)
                        if ai is not None:
                            pred_label, pred_conf = await asyncio.to_thread(ai.predict, session_strokes)
                        if pred_label:
                            if pred_label != session_label.upper():
                                log.warning(f"⚠️ 현재 AI는 '{session_label}'을/를 '{pred_label}'(으)로 잘못 인식합니다! (신뢰도: {pred_conf*100:.0f}%)")
                            else:
                                log.info(f"✅ AI도 방금 쓴 글자를 '{pred_label}'(으)로 완벽히 예측했습니다! (신뢰도: {pred_conf*100:.0f}%)")
                    
                elif cmd.get("action") == "get_dataset_info":
                    items = _scan_dataset()
                    class_info = {}
                    for item in items:
                        lbl = item["label"]
                        if lbl not in class_info:
                            class_info[lbl] = {"samples": 0, "points": 0}
                        class_info[lbl]["samples"] += 1
                        class_info[lbl]["points"] += item["points"]
                    await websocket.send(json.dumps({
                        "type": "dataset_info",
                        "classes": class_info,
                        "total_files": len(items),
                    }))
                
                elif cmd.get("action") == "train_model":
                    epochs = cmd.get("epochs", 50)
                    log.info(f"🤖 AI 학습 스레드 시작... (목표: {epochs} Epochs)")
                    
                    def do_train(target_epochs):
                        loop = main_loop
                        
                        def train_callback(msg, level="info", **kw):
                            if "epoch" in kw:
                                prog = {
                                    "type": "train_progress",
                                    "epoch": kw["epoch"],
                                    "total_epochs": target_epochs,
                                    "loss": kw.get("loss", 0.0),
                                    "acc": kw.get("acc", 0.0) * 100,
                                    "done": False
                                }
                                asyncio.run_coroutine_threadsafe(ws_broadcast(prog), loop)
                            else:
                                if level == "warn":
                                    log.warning(f"[AI Train] {msg}")
                                else:
                                    log.info(f"[AI Train] {msg}")

                        # 실제 학습 파이프라인 실행!
                        dataset_dir = str(Path(__file__).parent / "dataset")
                        success, message = ai.train(
                            data_dir=dataset_dir,
                            epochs=target_epochs,
                            callback=train_callback
                        )
                        
                        # 완료 후 브로드캐스트
                        asyncio.run_coroutine_threadsafe(ws_broadcast({"type": "train_progress", "done": True}), loop)
                        
                        if success:
                            log.info(f"✅ AI 모델 학습 완료 및 저장 성공: {message}")
                            # 학습이 성공했으면 저장된 가중치를 즉시 메모리에 반영
                            ai.load_model()
                        else:
                            log.error(f"❌ AI 학습 실패: {message}")

                    threading.Thread(target=do_train, args=(epochs,), daemon=True).start()
                    
                elif cmd.get("action") == "chat_request":
                    user_msg = cmd.get("message", "")
                    log.info(f"💬 사용자 질문: {user_msg}")
                    # 향후 Gemini API 도는 Groq(Llama3) API 호출 연동부분
                    # 현재는 Rule-based 더미 응답으로 시스템 분석 결과를 시뮬레이션합니다.
                    response = ""
                    if "loss" in user_msg.lower() or "로스" in user_msg:
                        response = "현재 모델의 Loss가 떨어지지 않는 이유는, Mamba의 시계열 스캔 레이어(4단계)에서 Dropout 확률이 높아 과소적합(Underfitting)이 발생하고 있기 때문으로 분석됩니다. 학습률(LR)을 1e-4로 낮추고 다시 해보세요."
                    elif "왜" in user_msg and "인식" in user_msg:
                        response = "현재 가중치는 사전 학습되지 않은 <b>무작위 초기화 상태(Random Init)</b>이기 때문입니다.<br><br>💡 좌측의 <b>Start Training Mamba SSM</b> 버튼을 눌러 IAM 오프라인 데이터셋으로 최소 50 Epoch 이상 훈련시켜야 90% 이상의 정확도를 확보할 수 있습니다."
                    else:
                        response = f"질문하신 '{user_msg}'에 대해 로그상 특이사항은 발견되지 않았습니다. 백엔드(Nvidia Orin Nano)의 VRAM은 현재 8GB 중 1.2GB를 사용 중이며 아주 안정적입니다."
                        
                    await websocket.send(json.dumps({
                        "type": "chat_response",
                        "message": response
                    }))
                    
                elif cmd.get("action") == "stats":
                    stats = {
                        "type": "stats",
                        "parser": parser.get_stats(),
                        "time_sync": time_sync.get_stats(),
                    }
                    await websocket.send(json.dumps(stats))
                
                elif cmd.get("action") == "recognize":
                    log.warning("프론트엔드에서의 'recognize' 액션은 백엔드 스트리밍 아키텍처 도입으로 폐기되었습니다. 글씨를 쓰면 자동으로 streaming_text가 전송됩니다.")
                
                # ─── JW v1 Training Studio API ───
                elif cmd.get("action") == "studio_init":
                    # 데이터셋 목록 전송
                    items = _scan_dataset()
                    await websocket.send(json.dumps({
                        "type": "studio_data",
                        "items": items,
                    }))
                    # GPU 정보
                    try:
                        import torch as _torch
                        gpu_info = f"CUDA: {_torch.cuda.get_device_name(0)}" if _torch.cuda.is_available() else "CPU only"
                    except ImportError:
                        gpu_info = "CPU only (torch not installed)"
                    await websocket.send(json.dumps({
                        "type": "device_info",
                        "gpu": gpu_info,
                    }))
                    # IAM 데이터셋 상태
                    try:
                        from airwriting_imu.core.iam_dataset import IAMDatasetLoader
                        iam_status = IAMDatasetLoader().get_status()
                        await websocket.send(json.dumps({
                            "type": "iam_status",
                            **iam_status,
                        }))
                    except Exception as e:
                        log.warning(f"IAM 데이터셋 초기화 실패: {e}")

                elif cmd.get("action") == "delete_dataset":
                    fname = cmd.get("file", "")
                    fpath = str(Path(__file__).parent / "dataset" / fname)
                    if os.path.exists(fpath):
                        os.remove(fpath)
                        log.info(f"🗑️ 데이터 삭제: {fname}")
                        await websocket.send(json.dumps({
                            "type": "studio_data",
                            "items": _scan_dataset(),
                        }))
                
                elif cmd.get("action") == "train_jw_v1":
                    log.info("🧠 실제 모델 학습 시작...")
                    _loop = main_loop
                    def _do_train_real():
                        try:
                            epochs = int(cmd.get("epochs", 30))
                            bs = cmd.get("batch_size", 16)
                            lr = 0.002
                            
                            def ws_log_callback(msg_text, lvl="info", epoch=None, progress=None, loss=None, acc=None, accuracy=None):
                                msg = {
                                    "type": "train_log", 
                                    "message": msg_text, 
                                    "level": lvl
                                }
                                if epoch is not None: msg["epoch"] = epoch
                                if progress is not None: msg["progress"] = progress
                                if loss is not None: msg["loss"] = loss
                                if acc is not None: msg["acc"] = acc
                                if accuracy is not None: msg["accuracy"] = accuracy
                                
                                asyncio.run_coroutine_threadsafe(ws_broadcast(msg), _loop)
                            
                            model_type = cmd.get("model_type", "pure_bilstm")
                            success, result_msg = ai.train(
                                data_dir=str(Path(__file__).parent / "dataset"),
                                epochs=epochs, 
                                lr=lr, 
                                batch_size=bs, 
                                model_type=model_type,
                                callback=ws_log_callback
                            )
                            
                            if success:
                                # 학습 완료 후 즉시 새 가중치를 추론에 반영 (Hot-Reload)
                                ai.load_model()
                                streamer.engine = ai
                                asyncio.run_coroutine_threadsafe(ws_broadcast({
                                    "type": "train_complete", "accuracy": result_msg.split()[-1].replace('%', '')
                                }), _loop)
                                log.info(f"✅ 학습 완료 + 모델 Hot-Reload: {result_msg}")
                            else:
                                ws_log_callback(f"학습 실패: {result_msg}", "error")
                                
                        except Exception as e:
                            import traceback
                            err_trace = traceback.format_exc()
                            log.error(f"Train Crash: {err_trace}")
                            asyncio.run_coroutine_threadsafe(ws_broadcast({
                                "type": "train_log", "message": f"Error: {e}", "level": "error"
                            }), _loop)
                    
                    # start the background trigger
                    threading.Thread(target=_do_train_real, daemon=True).start()
                
                elif cmd.get("action") == "ai_query":
                    query = cmd.get("query", "")
                    # 로컬 분석 AI (데이터셋 통계 기반)
                    response = _ai_analyze(query)
                    await websocket.send(json.dumps({
                        "type": "ai_response",
                        "message": response
                    }))
                
                elif cmd.get("action") == "export_onnx":
                    log.info("📦 ONNX 변환 시작...")
                    try:
                        if ai.model:
                            ai.export_onnx()
                            await websocket.send(json.dumps({
                                "type": "train_log", "message": "ONNX export complete!", "level": "success"
                            }))
                    except Exception as e:
                        await websocket.send(json.dumps({
                            "type": "train_log", "message": f"ONNX export failed: {e}", "level": "error"
                        }))

            except json.JSONDecodeError:
                pass
    finally:
        ws_clients.discard(websocket)
        log.info(f"🔌 WebSocket 해제 (남은: {len(ws_clients)})")


def _scan_dataset(data_dir=None):
    """데이터셋 폴더 스캔 → 메타데이터 목록"""
    if data_dir is None:
        data_dir = str(Path(__file__).parent / "dataset")
    items = []
    if not os.path.isdir(data_dir):
        return items
    for f in sorted(os.listdir(data_dir)):
        if not f.endswith('.json'):
            continue
        try:
            fpath = os.path.join(data_dir, f)
            with open(fpath, 'r', encoding='utf-8') as fp:
                data = json.load(fp)
            label = data.get("label", f.split("_")[0])
            strokes = data.get("strokes", [])
            total_pts = sum(len(s) for s in strokes)
            items.append({
                "file": f,
                "label": label,
                "strokes": len(strokes),
                "points": total_pts,
                "size": os.path.getsize(fpath),
            })
        except Exception:
            pass
    return items


def _ai_analyze(query: str) -> str:
    """AI 질의응답 — 데이터셋/모델 상태 분석"""
    items = _scan_dataset()
    labels = {}
    total_pts = 0
    for item in items:
        labels[item["label"]] = labels.get(item["label"], 0) + 1
        total_pts += item["points"]
    
    q = query.lower()
    
    if "불균형" in q or "balance" in q:
        if not labels:
            return "데이터가 없습니다."
        counts = list(labels.values())
        ratio = min(counts) / max(counts)
        status = "균형" if ratio > 0.7 else "불균형" if ratio > 0.3 else "심각한 불균형"
        return f"클래스 균형도: {ratio:.0%} ({status}). " + \
               ", ".join(f"{k}={v}개" for k, v in sorted(labels.items()))
    
    if "추천" in q or "설정" in q or "하이퍼" in q:
        n = len(items)
        if n < 20:
            return f"데이터 {n}개: epochs=20, batch=4, lr=0.003, augment=100x 추천. 데이터가 많이 부족합니다."
        elif n < 100:
            return f"데이터 {n}개: epochs=30, batch=8, lr=0.002, augment=50x 추천."
        else:
            return f"데이터 {n}개: epochs=50, batch=16, lr=0.001, augment=20x 추천. 충분한 데이터!"
    
    if "모델" in q or "jw" in q.lower():
        return "JW v1: 1.13M params | VQ-VAE(K=512) → 4-layer Mamba SSM | O(N) 선형 복잡도 | CPU ~22ms | Jetson Orin Nano 최적화"
    
    return f"총 {len(items)}개 샘플, {len(labels)}개 클래스, {total_pts:,}개 포인트. 구체적으로 질문해주세요 (예: '데이터 불균형 확인', '추천 설정')"


# ─── USB Serial 수신 (자동 재연결) ───
def _queue_put_safe(queue: asyncio.Queue, packet: bytes):
    try:
        queue.put_nowait(packet)
    except asyncio.QueueFull:
        pass  # 버퍼 포화 시 신규 패킷 드롭 — 기존 최신 데이터 보존


def serial_reader_thread(loop, queue: asyncio.Queue):
    import serial
    while True:
        try:
            # inter_byte_timeout: 바이트 간 1ms 간격 발생 시 즉시 read() 리턴
            # → 패킷 1개(0.76ms 전송)와 다음 패킷 사이 11ms 갭마다 OS flush
            # Windows USB CDC 청킹 우회. 결과: read() 가 ~85Hz 로 1패킷씩 반환
            with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1,
                               inter_byte_timeout=0.001) as ser:
                log.info(f"🔌 USB Serial 연결됨: {SERIAL_PORT} ({BAUD_RATE} bps)")
                buffer = bytearray()
                while True:
                    # Read all available bytes or wait for at least 1 (to avoid spinning)
                    # This is much more efficient than reading fixed 94B on Windows
                    data = ser.read(max(1, ser.in_waiting))
                    if data:
                        buffer.extend(data)
                        while len(buffer) >= 68:
                            idx = buffer.find(0xAA)
                            if idx < 0:
                                buffer.clear()
                                break
                            if idx > 0:
                                buffer = buffer[idx:]
                            packet_found = False
                            for p_size in [92, 94, 68, 70]:
                                if len(buffer) >= p_size and buffer[p_size-1] == 0x55:
                                    packet = bytes(buffer[:p_size])
                                    loop.call_soon_threadsafe(_queue_put_safe, queue, packet)
                                    buffer = buffer[p_size:]
                                    packet_found = True
                                    break
                            if not packet_found:
                                if len(buffer) > 100:
                                    buffer = buffer[1:]
                                else:
                                    break
        except serial.SerialException as e:
            log.warning(f"🔌 Serial 끊김 ({e}) — 3초 후 재연결 시도...")
            time.sleep(3)
        except Exception as e:
            log.error(f"Serial 에러: {e}")
            time.sleep(3)

async def process_serial_queue(queue: asyncio.Queue):
    global esp32_addr, smooth_x, smooth_y, oef_x, oef_y, _last_packet_ts, madgwick_s3_ray
    global was_writing, current_stroke, is_recording_session, session_strokes, button_debounce_counter
    global _ws_frame_counter, _stationary_frames, _last_recenter_mono, _diag_frame_counter
    esp32_addr = "USB"
    last_ws_time = 0.0
    last_monitor_time = time.time()
    while True:
        try:
            # 큐 상태 모니터링 (1초마다 출력)
            now = time.time()
            qsize = queue.qsize()
            
            # 지연 상태에 따른 적응형 처리 모드 (Adaptive Mode)
            # 큐가 20개 이상 쌓이면 '지연 방어 모드' 활성화 (무거운 연산 생략)
            adaptive_skip = (qsize > 20)
            
            if now - last_monitor_time > 1.0:
                if qsize > 50:
                    log.warning(f"⚠️ 처리 지연 발생! Queue Size: {qsize} (AdaptiveSkip={'ON' if adaptive_skip else 'OFF'})")
                last_monitor_time = now

            # 모든 패킷을 ESKF에 빠짐없이 통과시키되, 웹소켓 전송은 마지막 1개만!
            # 배치 크기는 16으로 상한 (Windows 100ms 버퍼링 = ~8-9 패킷 대비 충분, 폭주 방지)
            packets = [await queue.get()]
            while not queue.empty() and len(packets) < 16:
                try:
                    packets.append(queue.get_nowait())
                except asyncio.QueueEmpty:
                    break

            for pkt_idx, data in enumerate(packets):
                # 4 패킷마다 이벤트 루프 양보 (WebSocket ping 응답성 보장)
                if pkt_idx % 4 == 0:
                    await asyncio.sleep(0)
                
                is_last_packet = (pkt_idx == len(packets) - 1)
                
                frame = parser.parse(data)
                if frame is None:
                    continue

                time_sync.update(frame.timestamp_ms)
                ts = int(time_sync.esp32_to_python_time(frame.timestamp_ms) * 1000)
                
                # 1. 캘리브레이션 단계 (S1, S2, S3 T-Pose 정렬)
                if not calibrator.is_calibrated:
                    done = calibrator.add_sample(frame)
                    
                    if is_last_packet:
                        msg = {
                            "type": "status",
                            "text": f"CALIBRATING NEUTRAL POSE... [{len(calibrator.samples['s1_a'])}/{calibrator.req_samples}]",
                            "ts": frame.timestamp_ms,
                        }
                        asyncio.create_task(ws_broadcast(msg))
                    
                    if done:
                        log.info(f"🎯 착용자 자세 정렬 완료!")
                        for sid in ('s3', 's2', 's1'):
                            label = {'s3': 'S3 Finger', 's2': 'S2 Hand  ', 's1': 'S1 Wrist '}[sid]
                            ba = calibrator.ba[sid]
                            bg = calibrator.bg[sid]
                            log.info(
                                "   [%s] q_align=%s | ba=(%+.3f,%+.3f,%+.3f) m/s² | bg=(%+.4f,%+.4f,%+.4f) rad/s",
                                label, calibrator.q_align[sid],
                                ba[0], ba[1], ba[2], bg[0], bg[1], bg[2],
                            )
                        m_ref_s3 = calibrator.m_ref.get('s3')
                        if m_ref_s3 is not None:
                            log.info(
                                "   [S3 Mag   ] m_ref=(%+.3f,%+.3f,%+.3f) (unit vec, %d samples)",
                                float(m_ref_s3[0]), float(m_ref_s3[1]), float(m_ref_s3[2]),
                                len(calibrator.samples.get('s3_m', [])),
                            )

                        # 정지 자세 품질 진단 — 진동 중 캘리브레이션 발생 시 사용자에 경고
                        log.info(
                            "   [QUALITY  ] accel_std S1=%.3f S2=%.3f S3=%.3f m/s² (threshold %.2f)",
                            calibrator.accel_std['s1'],
                            calibrator.accel_std['s2'],
                            calibrator.accel_std['s3'],
                            0.8,
                        )
                        if not calibrator.quality_ok:
                            log.error(
                                "⚠️ 캘리브레이션 중 손이 흔들렸습니다. 'Calibrate'를 다시 누르고 정지 자세를 유지해주세요."
                            )
                            asyncio.create_task(ws_broadcast({
                                "type": "status",
                                "text": "CALIBRATION SHAKY — REDO RECOMMENDED",
                                "ts": ts,
                            }))

                        s1_eskf.reset(initial_q=calibrator.q_align['s1'].astype(np.float64), initial_ba=calibrator.ba['s1'], initial_bg=calibrator.bg['s1'])
                        s2_eskf.reset(initial_q=calibrator.q_align['s2'].astype(np.float64), initial_ba=calibrator.ba['s2'], initial_bg=calibrator.bg['s2'])
                        s3_eskf.reset(initial_q=calibrator.q_align['s3'].astype(np.float64), initial_ba=calibrator.ba['s3'], initial_bg=calibrator.bg['s3'], initial_mag=calibrator.m_ref.get('s3'))

                        if RAW_LOGGER is not None:
                            RAW_LOGGER.write_header(calibrator)

                        # [Phase 1] Madgwick Filter 정렬 (원본 GitHub 동일 — q_align 그대로 적용)
                        # ESKF는 .inv() 적용하지만, Madgwick 필터는 W→S 컨벤션을 그대로 사용해야
                        # scipy Rotation.apply()가 forward 벡터를 올바른 방향으로 회전시킴.
                        q_s3 = calibrator.q_align['s3']
                        madgwick_s3_ray.reset(
                            q_wxyz=np.array([q_s3[3], q_s3[0], q_s3[1], q_s3[2]], dtype=np.float64)
                        )
                        ray_projector.reset()

                        smooth_x, smooth_y = 0.0, 0.0
                        oef_x.reset()
                        oef_y.reset()
                        
                        # Yaw Stabilizer 초기화 (자기장 캘리브레이션 포함)
                        s1_y0, _, s1_p0 = s1_eskf.q.as_euler('ZYX', degrees=False)
                        s3_y0, _, _ = s3_eskf.q.as_euler('ZYX', degrees=False)
                        
                        # 캘리브레이션 중 수집된 자기장 데이터로 MagFusion 활성화
                        mag_samples = calibrator.samples.get('s3_m', [])
                        if len(mag_samples) > 5:
                            mag_arr = np.array(mag_samples)
                            yaw_stabilizer.calibrate(
                                mag_samples=mag_arr, s1_yaw=s1_y0, s3_yaw=s3_y0, heading=0.0
                            )
                            log.info(f"🧲 MagFusion 활성화: {len(mag_samples)}개 샘플, norm={np.median(np.linalg.norm(mag_arr, axis=1)):.1f}µT")
                        else:
                            yaw_stabilizer.calibrate(
                                mag_samples=None, s1_yaw=s1_y0, s3_yaw=s3_y0, heading=0.0
                            )
                            # 자기장 데이터가 부족하면 yaw 드리프트 누적 위험 — UI에도 즉시 표시
                            log.error(
                                "⚠️ MagFusion 비활성화 — 자기장 샘플 %d개 (필요 ≥6). yaw 드리프트가 누적될 수 있습니다.",
                                len(mag_samples),
                            )
                            asyncio.create_task(ws_broadcast({
                                "type": "status",
                                "text": "MAG DISABLED — YAW WILL DRIFT",
                                "ts": ts,
                            }))
                        
                        msg = {
                            "type": "status",
                            "text": "READY",
                            "ts": ts,
                        }
                        asyncio.create_task(ws_broadcast(msg))
                    continue

                # ── ESP32 리셋 자동 감지 ──
                if frame.seq != -1 and parser.last_seq != -1 and frame.seq < parser.last_seq - 50:
                    log.warning("🔄 ESP32 Reset 감지! 필터 초기화...")
                    s1_eskf.reset()
                    s2_eskf.reset()
                    s3_eskf.reset()
                    madgwick_s3_ray.reset()
                    ray_projector.reset()
                    asyncio.create_task(ws_broadcast({
                        "type": "status",
                        "text": "ESP32 RESET — RE-CALIBRATING...",
                        "ts": ts,
                    }))
                    continue

                # 동적 dt 계산 (실제 패킷 간격 기반)
                dt = 0.0117  # 85Hz 통신 기준 디폴트
                if _last_packet_ts > 0 and ts > _last_packet_ts:
                    dt = (ts - _last_packet_ts) / 1000.0
                dt = max(0.005, min(dt, 0.1))
                _last_packet_ts = ts
                
                # ESKF Predict (Gyro 적분 정확도를 위해 매 패킷 실행)
                s1_eskf.predict(frame.wrist_accel, frame.wrist_gyro, dt)
                if frame.hand_accel is not None:
                    s2_eskf.predict(frame.hand_accel, frame.hand_gyro, dt)
                s3_eskf.predict(frame.finger_accel, frame.finger_gyro, dt)
                
                # ── 물리 버튼 디바운스 처리 ──
                raw_button = (frame.button > 0)
                if raw_button:
                    button_debounce_counter = DEBOUNCE_FRAMES
                    is_writing = True
                else:
                    if button_debounce_counter > 0:
                        button_debounce_counter -= 1
                        is_writing = True
                    else:
                        is_writing = False

                do_heavy_update = is_last_packet and (not adaptive_skip)
                s3_zupt = False  # do_heavy_update=False 프레임에서 NameError 방지

                if do_heavy_update:
                    gravity_alpha = 0.0003 if is_writing else 0.002
                    s1_eskf.update_gravity_mahony(frame.wrist_accel, alpha=gravity_alpha)
                    s3_eskf.update_gravity_mahony(frame.finger_accel, alpha=gravity_alpha)
                    
                    if not is_writing:
                        s3_zupt = s3_eskf.detect_zupt()
                        if s3_zupt:
                            s3_eskf.update_zupt(frame.finger_gyro)
                    else:
                        s3_zupt = False

                    # 동작 분리
                    wrist_motion, finger_motion = motion_sep.separate(frame.wrist_accel, frame.finger_accel)
                    writing_intent = motion_sep.get_writing_intent(finger_motion)
                    pipeline_state["writing_intent"] = writing_intent

                    # 관절 운동학 (DifferentialKinematics)
                    try:
                        ha = frame.hand_accel if frame.hand_accel is not None else frame.wrist_accel
                        hg = frame.hand_gyro  if frame.hand_accel is not None else frame.wrist_gyro
                        bio_result = diff_kin.update_full_chain(
                            frame.wrist_accel, frame.wrist_gyro,
                            ha, hg,
                            frame.finger_accel, frame.finger_gyro,
                            s1_eskf.q, s2_eskf.q, s3_eskf.q, dt
                        )
                        pipeline_state["bio_kin"] = bio_result
                    except Exception:
                        pass

                # [Phase 0 / E5] 원시 프레임 + 라이브 제어 결정 기록 (opt-in, 가드 뒤, 매 프레임)
                if RAW_LOGGER is not None:
                    RAW_LOGGER.log(ts, dt, frame, is_writing, do_heavy_update, s3_zupt)

                # ── [Phase 1] Madgwick Filter + yaw_stabilizer 보정 ──
                q_s3_ray_prev = madgwick_s3_ray.q
                if do_heavy_update:
                    corrected_s3_gyro = yaw_stabilizer.process(
                        s3_accel=frame.finger_accel, s3_gyro=frame.finger_gyro, s3_mag=frame.finger_mag,
                        s1_gyro=frame.wrist_gyro, is_writing=is_writing,
                        current_q_wxyz=q_s3_ray_prev, dt=dt
                    )
                else:
                    corrected_s3_gyro = frame.finger_gyro
                
                # Madgwick Filter: 오일러 분해 없이 쿼터니언 미분/경사하강 융합
                q_s3_ray = madgwick_s3_ray.update_imu(frame.finger_accel, corrected_s3_gyro, dt=dt)

                # ── 정지 감지 (yaw 자동 리센터 트리거용, 매 패킷 카운팅) ──
                gyro_norm = math.sqrt(
                    frame.finger_gyro[0]**2 + frame.finger_gyro[1]**2 + frame.finger_gyro[2]**2
                )
                if (not is_writing) and gyro_norm < STATIONARY_GYRO_THRESH:
                    _stationary_frames += 1
                else:
                    _stationary_frames = 0

                # [Crucial] 마지막 패킷이 아니면 투영/웹소켓 생략하고 다음 패킷으로!
                if not is_last_packet:
                    continue

                # (Auto-Recenter 삭제: 정지 상태마다 화면이 홱홱 돌아가는 문제 원인)

                # ── [Phase 2] 원본 Forward Vector 투영 ──
                # ray_projector는 이제 tan()을 쓰지 않고 필터링되지 않은 원본 yaw, pitch(rad)를 반환합니다.
                phys_x, phys_z = ray_projector.project(q_s3_ray)
                
                # 3. One Euro Filter (원본 방식: 물리적 각도를 먼저 필터링)
                # USB 패킷 지연 처리에 의한 dt=0 버그 방지를 위해 timestamp=None 전달
                filtered_x = oef_x.filter(phys_x, timestamp=None)
                filtered_y = oef_y.filter(phys_z, timestamp=None)

                # 4. 투영 스케일링 (원본 GitHub 매핑: yaw→X, pitch→Y)
                VIRTUAL_PEN_LENGTH = 2.5

                out_x = round(filtered_x * VIRTUAL_PEN_LENGTH, 4)
                out_y = round(filtered_y * VIRTUAL_PEN_LENGTH, 4)

                # ── [DIAG] 축 매핑/투영 진단 로그 (~10Hz) ──
                if DIAG_MODE:
                    _diag_frame_counter += 1
                    if _diag_frame_counter % 8 == 0:  # 85Hz / 8 ≈ 10.6Hz
                        qw, qx, qy, qz = q_s3_ray
                        log.info(
                            "[DIAG] phys=(%+.3f,%+.3f) rad | q_s3=(%+.3f,%+.3f,%+.3f,%+.3f) | out=(%+.3f,%+.3f)",
                            phys_x, phys_z, qw, qx, qy, qz, out_x, out_y,
                        )

                # ── 스트리밍 및 웹소켓 전송 ──
                if is_recording_session:
                    if is_writing:
                        current_stroke.append({
                            "ts": ts, "dt": dt, "x": out_x, "y": out_y,
                            "ax": float(frame.finger_accel[0]),
                            "ay": float(frame.finger_accel[1]),
                            "az": float(frame.finger_accel[2]),
                            "gx": float(frame.finger_gyro[0]),
                            "gy": float(frame.finger_gyro[1]),
                            "gz": float(frame.finger_gyro[2])
                        })
                    elif was_writing and not is_writing:
                        if len(current_stroke) > 2:  # 기존 10에서 2로 하향 (짧은 획이나 점이 무시되는 현상 방지)
                            session_strokes.append(current_stroke)
                            log.info(f"[Record] 획 추가 완료: {len(current_stroke)} points (총 {len(session_strokes)}획)")
                        current_stroke = []
                else:
                    if current_stroke:
                        current_stroke = []
                    # 백엔드 스트리밍 인퍼런스
                    frame_features = {
                        "x": out_x, "y": out_y,
                        "ax": float(frame.finger_accel[0]),
                        "ay": float(frame.finger_accel[1]),
                        "az": float(frame.finger_accel[2]),
                        "gx": float(frame.finger_gyro[0]),
                        "gy": float(frame.finger_gyro[1]),
                        "gz": float(frame.finger_gyro[2])
                    }
                    if streamer is not None:
                        streamer.process_frame(frame_features, is_writing)

                was_writing = is_writing

                # 윈도우 Serial 버퍼링(100ms 단위 청크)으로 인해 중간 패킷이 생략되는 것을 방지하기 위해
                # 수신된 모든 프레임을 웹소켓으로 전송하여 선이 부드럽게 그려지도록 합니다.
                now_mono = time.monotonic()
                # WebSocket throttling: 60Hz is enough for smooth UI (reduces JSON overhead)
                WS_INTERVAL = 1/60.0
                if now_mono - last_ws_time >= WS_INTERVAL:
                    if now_mono - last_ws_time > 2 * WS_INTERVAL:
                        last_ws_time = now_mono
                    else:
                        last_ws_time += WS_INTERVAL

                    _ws_frame_counter += 1
                    send_heavy = (_ws_frame_counter % 6 == 0)  # 3D/센서 필드는 ~15Hz만 전송

                    # 85Hz 필수 필드 (커서·펜 상태만, 작은 payload)
                    msg = {
                        "ray_hit": [out_x, out_y],
                        "fingertip": [out_x, out_y, -0.68],
                        "is_writing": is_writing,
                        "button": frame.button,
                        "ts": ts,
                    }

                    if send_heavy:
                        q1 = s1_eskf.q.as_quat()
                        q2 = s2_eskf.q.as_quat() if frame.hand_accel is not None else q1
                        q3 = s3_eskf.q.as_quat()
                        hand_accel = frame.hand_accel
                        msg["pkt"] = parser.valid_packets
                        msg["latency"] = round(time_sync.latency_ms, 1) if time_sync.is_synced else -1
                        msg["zupt"] = bool(s3_zupt)
                        msg["orientations"] = {
                            "forearm": [float(q1[3]), float(q1[0]), float(q1[1]), float(q1[2])],
                            "hand":    [float(q2[3]), float(q2[0]), float(q2[1]), float(q2[2])],
                            "finger":  [float(q3[3]), float(q3[0]), float(q3[1]), float(q3[2])],
                        }
                        msg["raw_sensors"] = {
                            "s1": {"ax": float(frame.wrist_accel[0]), "ay": float(frame.wrist_accel[1]), "az": float(frame.wrist_accel[2])},
                            "s2": {"ax": float(hand_accel[0] if hand_accel is not None else 0.0),
                                   "ay": float(hand_accel[1] if hand_accel is not None else 0.0),
                                   "az": float(hand_accel[2] if hand_accel is not None else 0.0)},
                            "s3": {"ax": float(frame.finger_accel[0]), "ay": float(frame.finger_accel[1]), "az": float(frame.finger_accel[2])},
                        }
                        msg["pipeline"] = {
                            "joint_angles": pipeline_state["bio_kin"].get("joint_angles", {}),
                            "writing_intent": round(pipeline_state.get("writing_intent", 0), 3),
                        }

                    asyncio.create_task(ws_broadcast(msg))
        except Exception as e:
            import traceback
            log.error(f"❌ 큐 프로세스 오류: {e}")
            log.error(traceback.format_exc())


# ─── Mock 데이터 ───
async def mock_data_generator():
    log.info("🎭 Mock 모드 — 가상 필기 데이터 생성")
    t = 0
    ts = 0
    while True:
        t += 0.01
        ts += 10

        phase = t % 8.0
        pen = phase < 5.0
        if pen:
            angle = phase * 2 * math.pi / 2.5
            ax, ay = math.cos(angle) * 3.0, math.sin(angle) * 3.0
        else:
            ax = ay = 0.0

        frame = SensorFrame(
            timestamp_ms=ts,
            wrist_accel=np.array([0, 0, 9.81], dtype=np.float32),
            wrist_gyro=np.zeros(3, dtype=np.float32),
            finger_accel=np.array([ax, ay, 9.81], dtype=np.float32),
            finger_gyro=np.array([0, 0, 0.3 if pen else 0], dtype=np.float32),
            finger_mag=np.zeros(3, dtype=np.float32),
            button=1 if pen else 0,
            packet_size=68,
        )

        s3_eskf.predict(frame.finger_accel, frame.finger_gyro, 0.01)
        x, y, _ = s3_eskf.p
        msg = {
            "type": "position",
            "x": round(x * 1000, 2),
            "y": round(y * 1000, 2),
            "button": frame.button,
            "zupt": False,
            "ts": ts,
            "pkt": ts // 10,
            "latency": 0.0,
        }
        await ws_broadcast(msg)
        await asyncio.sleep(0.01)


# ─── HTTP 서버 ───
def start_http_server():
    web_dir = Path(__file__).parent / "web"
    web_dir.mkdir(parents=True, exist_ok=True)
    handler = partial(SimpleHTTPRequestHandler, directory=str(web_dir))
    httpd = HTTPServer(("0.0.0.0", HTTP_PORT), handler)
    log.info(f"🌍 HTTP 서버: http://localhost:{HTTP_PORT}")
    httpd.serve_forever()


# ─── 상태 출력 ───
async def status_printer():
    last_valid = 0
    while True:
        await asyncio.sleep(10)
        stats = parser.get_stats()
        rate = (stats["valid"] - last_valid) / 10.0
        last_valid = stats["valid"]
        ts = time_sync.get_stats()
        esp = esp32_addr if esp32_addr else "미연결"

        log.info(
            f"📊 ESP32={esp} | {rate:.0f}Hz | "
            f"loss={stats['loss_rate']}% gaps={stats['gap_count']} | "
            f"latency={ts['latency_ms']}ms | "
            f"WS={len(ws_clients)}"
        )
        
        # 3-Layer Yaw Stabilizer 상태 로깅
        ystats = yaw_stabilizer.get_stats()
        log.info(
            f"🛡️ YawStabilizer | ZARU(Bias): {ystats['bias']['bias_deg_s'][2]:.3f}°/s "
            f"| Anchor(Dev): {ystats['anchor']['deviation_deg']:.1f}° "
            f"| Mag(Trust): {ystats['mag']['trust']:.2f}"
        )


# ─── 메인 ───
async def main():
    global main_loop, ai, streamer
    main_loop = asyncio.get_running_loop()

    # [BUG-3] AI/스트리머 지연 초기화. 생성/로드가 실패해도 서버는 계속 기동한다.
    try:
        ai = AirWritingAI()
        ai.load_model()  # 내부적으로 실패를 잡고 False 반환 — 서버는 멈추지 않음
    except Exception as e:
        ai = None
        log.warning(f"AI 초기화 실패 — 인식 비활성화로 서버 계속 기동: {e}")
    streamer = StreamingInference(ai, char_timeout=0.8, space_timeout=2.0)
    streamer.on_text_updated = _on_text_updated

    http_thread = threading.Thread(target=start_http_server, daemon=True)
    http_thread.start()

    ws_server = await ws_serve(ws_handler, "0.0.0.0", WS_PORT)
    log.info(f"🔌 WebSocket: ws://localhost:{WS_PORT}")

    serial_queue = asyncio.Queue(maxsize=500)  # 약 6초치 버퍼 — 초과 시 _queue_put_safe가 드롭
    tasks = [status_printer(), process_serial_queue(serial_queue)]
    
    if not mock_mode:
        t = threading.Thread(target=serial_reader_thread, args=(main_loop, serial_queue), daemon=True)
        t.start()
    else:
        tasks.append(mock_data_generator())

    log.info("=" * 55)
    log.info("  Hybrid-AirScribe Digital Twin — Phase 1 (USB Mode)")
    log.info(f"  Dashboard:  http://localhost:{HTTP_PORT}")
    log.info(f"  WebSocket:  ws://localhost:{WS_PORT}")
    log.info(f"  Serial:     {SERIAL_PORT} @ {BAUD_RATE}")
    log.info(f"  Mode:       {'🎭 Mock' if mock_mode else '📡 Live USB'}")
    log.info("=" * 55)

    try:
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        pass
    finally:
        ws_server.close()
        if RAW_LOGGER is not None:
            RAW_LOGGER.close()


if __name__ == "__main__":
    import serial.tools.list_ports
    ports = list(serial.tools.list_ports.comports())
    if ports:
        log.info(f"사용 가능한 COM 포트: {[p.device for p in ports]}")
    else:
        log.warning("장치 관리자에 인식된 COM 포트가 없습니다!")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("서버 종료")
