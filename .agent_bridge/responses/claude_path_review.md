# [Claude] Windows 환경 경로 및 오류 검증 결과

**작성자:** Claude (claude-sonnet-4-6)  
**작성일:** 2026-05-03  
**대상 프로젝트:** `airwriting_imu_only` (`C:\Users\USER\airwriting_imu_only`)  
**요청 파일:** `.agent_bridge/requests/windows_path_review.md`

---

## 검증 범위

전체 Python 소스 (`airwriting_imu/core/*.py`, `test_ws.py`) 및 설정 파일을 대상으로 아래 항목을 검증했습니다.

1. 시리얼 포트(COM3) 하드코딩 여부
2. 파일 경로 구분자 문제 (슬래시 혼용)
3. 소켓/포트 충돌 및 Windows 방화벽
4. 기타 Windows 특이적 런타임 버그

---

## 발견된 문제 목록 (우선순위 순)

| # | 심각도 | 파일 | 문제 요약 |
|---|--------|------|-----------|
| 1 | 🔴 Critical | `ai_model.py` | 상대경로(`weights/`, `dataset/`) — CWD 의존 |
| 2 | 🔴 Critical | `ai_model.py` | `torch.load()` `weights_only` 파라미터 누락 |
| 3 | 🔴 Critical | `config/` 디렉토리 | `imu.yaml`, `system.yaml` 파일 삭제됨 |
| 4 | 🔴 Critical | 미래 `main.py` | Windows ProactorEventLoop + UDP 비동기 불호환 |
| 5 | 🟡 Medium | `streaming.py` | 스레드 레이스 컨디션 (락 없는 공유 상태) |
| 6 | 🟡 Medium | `test_ws.py` | UTF-16 인코딩 → Python 파싱 실패 |
| 7 | 🟡 Medium | OS 설정 | Windows 방화벽 포트 차단 가능성 |
| 8 | ✅ 문제없음 | 전체 | COM3/시리얼 하드코딩 — 현재 코드에 없음 |
| 9 | ✅ 문제없음 | 전체 | 경로 구분자 — `os.path.join`, `pathlib` 올바르게 사용 중 |

---

## 상세 분석 및 수정 코드

---

### Issue 1 🔴 `ai_model.py` — 상대경로로 인한 FileNotFoundError

**위치:** `airwriting_imu/core/ai_model.py` 라인 25, 210, 389–394, 477–486, 493–535

**원인:**  
`"weights/meta.pkl"`, `"weights/pure_bilstm.pt"`, `"dataset"` 등이 모두 CWD(현재 작업 디렉토리) 기준 상대경로입니다.  
프로젝트 루트가 아닌 다른 위치에서 `python`을 실행하거나, IDE에서 파일을 직접 실행하면 즉시 `FileNotFoundError`가 발생합니다.

**수정 — `ai_model.py` 상단 import 블록에 추가:**

```python
from pathlib import Path

# 프로젝트 루트 절대경로 (airwriting_imu/core/ai_model.py 기준 3단계 위)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_WEIGHTS_DIR = _PROJECT_ROOT / "weights"
_DATASET_DIR = _PROJECT_ROOT / "dataset"
```

**수정 — GestureDataset 기본 인자 (라인 25):**

```python
# ❌ 기존
def __init__(self, data_dir="dataset", max_seq_len=200):
    self.data_dir = data_dir

# ✅ 수정
def __init__(self, data_dir=None, max_seq_len=200):
    if data_dir is None:
        data_dir = str(_DATASET_DIR)
    self.data_dir = data_dir
```

**수정 — JW v1 학습 저장 (라인 389–394):**

```python
# ❌ 기존
os.makedirs("weights", exist_ok=True)
torch.save(self.model.state_dict(), "weights/jw_v1.pt")
with open("weights/meta.pkl", "wb") as f:
    pickle.dump({...}, f)

# ✅ 수정
_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
torch.save(self.model.state_dict(), str(_WEIGHTS_DIR / "jw_v1.pt"))
with open(_WEIGHTS_DIR / "meta.pkl", "wb") as f:
    pickle.dump({...}, f)
```

**수정 — 시퀀스 모델 저장 (라인 477–486):**

```python
# ❌ 기존
os.makedirs("weights", exist_ok=True)
model_filename = f"{self.model_type}.pt"
torch.save(self.model.state_dict(), f"weights/{model_filename}")
with open("weights/meta.pkl", "wb") as f:
    pickle.dump({...}, f)

# ✅ 수정
_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
model_filename = f"{self.model_type}.pt"
torch.save(self.model.state_dict(), str(_WEIGHTS_DIR / model_filename))
with open(_WEIGHTS_DIR / "meta.pkl", "wb") as f:
    pickle.dump({...}, f)
```

**수정 — 모델 로드 (라인 493–535):**

```python
# ❌ 기존
with open("weights/meta.pkl", "rb") as f:
    meta = pickle.load(f)

# ✅ 수정
with open(_WEIGHTS_DIR / "meta.pkl", "rb") as f:
    meta = pickle.load(f)
```

**수정 — ONNX export (라인 639, 645):**

```python
# ❌ 기존
def export_onnx(self, path="weights/model.onnx"):
    self.model.export_onnx("weights/jw_v1.onnx")

# ✅ 수정
def export_onnx(self, path=None):
    if path is None:
        path = str(_WEIGHTS_DIR / "model.onnx")
    self.model.export_onnx(str(_WEIGHTS_DIR / "jw_v1.onnx"))
```

---

### Issue 2 🔴 `ai_model.py` — `torch.load()` `weights_only` 파라미터 누락

**위치:** `airwriting_imu/core/ai_model.py` 라인 506, 514, 519, 525, 531, 535

**원인:**  
- PyTorch **2.4+**: `FutureWarning` 발생  
- PyTorch **2.6+**: `weights_only=False`를 명시하지 않으면 **오류** 발생  
Windows에서 `pip install torch` 하면 최신 버전이 설치되므로 즉시 실행 불가.

**수정 — 6개 `torch.load()` 호출 전체:**

```python
# ❌ 기존
torch.load("weights/jw_v1.pt", map_location=self.device)
torch.load("weights/pure_bilstm.pt", map_location=self.device)
torch.load("weights/jw_v2.pt", map_location=self.device)
torch.load("weights/fastkan.pt", map_location=self.device)
torch.load("weights/cnn_bilstm.pt", map_location=self.device)
torch.load("weights/transformer.pt", map_location=self.device)

# ✅ 수정 (weights_only=True: state_dict 전용, 보안상 권장)
torch.load(str(_WEIGHTS_DIR / "jw_v1.pt"), map_location=self.device, weights_only=True)
torch.load(str(_WEIGHTS_DIR / "pure_bilstm.pt"), map_location=self.device, weights_only=True)
torch.load(str(_WEIGHTS_DIR / "jw_v2.pt"), map_location=self.device, weights_only=True)
torch.load(str(_WEIGHTS_DIR / "fastkan.pt"), map_location=self.device, weights_only=True)
torch.load(str(_WEIGHTS_DIR / "cnn_bilstm.pt"), map_location=self.device, weights_only=True)
torch.load(str(_WEIGHTS_DIR / "transformer.pt"), map_location=self.device, weights_only=True)
```

> **주의:** `meta.pkl`은 `pickle.load()`로 읽으므로 이 이슈와 무관합니다.

---

### Issue 3 🔴 `config/` 디렉토리 — 필수 파일 삭제됨

**원인:**  
`git status`에서 다음 파일들이 `D`(Deleted) 상태로 표시됩니다:
- `config/imu.yaml`
- `config/system.yaml`
- `config/system.local.example.yaml`

`config_loader.py:68`의 `ConfigLoader.__init__`은 이 파일들이 없으면 즉시 `FileNotFoundError("config/ not found")`를 던집니다.

**복원 명령:**

```bash
# 직전 커밋에서 config 파일 복원
git checkout HEAD~1 -- config/imu.yaml config/system.yaml config/system.local.example.yaml

# 또는 특정 커밋 해시로 복원
git log --oneline  # 커밋 해시 확인
git checkout <commit-hash> -- config/
```

복원이 불가능하다면, `config/imu.yaml`과 `config/system.yaml`을 프로젝트 요구사항에 맞게 새로 작성해야 합니다.

---

### Issue 4 🔴 미래 `main.py` — Windows asyncio ProactorEventLoop + UDP 불호환

**원인:**  
Python 3.8+ Windows에서 asyncio 기본 이벤트 루프는 `ProactorEventLoop`입니다.  
`ProactorEventLoop`는 `loop.create_datagram_endpoint()`(UDP 비동기 수신)을 **지원하지 않습니다**.  
`websockets` 서버(포트 12347)와 UDP 수신을 같은 asyncio 루프에서 돌리면 `NotImplementedError`가 발생합니다.

**수정 — `main.py` 재작성 시 파일 최상단에 반드시 추가:**

```python
import sys
import asyncio

# ✅ Windows에서 UDP 비동기 지원을 위해 SelectorEventLoop 강제 설정
# ProactorEventLoop(Windows 기본값)은 create_datagram_endpoint() 미지원
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# 이후에 asyncio.run(main()) 호출
```

> **참고:** `asyncio.WindowsSelectorEventLoopPolicy()`는 Python 3.7+부터 사용 가능합니다.

---

### Issue 5 🟡 `streaming.py` — 스레드 레이스 컨디션

**위치:** `airwriting_imu/core/streaming.py` 라인 88–108

**원인:**  
`_do_predict_and_emit`이 `loop.run_in_executor()`로 별도 스레드에서 실행되면서,  
메인 스레드에서도 읽는 `_last_emitted_char`과 `_last_emit_time`을 락 없이 수정합니다.  
Windows 스케줄러 특성상 Linux보다 스레드 컨텍스트 스위치가 더 자주 발생하여 같은 글자가 연달아 두 번 출력되는 증상이 생깁니다.

**수정 코드:**

```python
# streaming.py 상단 import에 추가
import threading

class StreamingInference:
    def __init__(
        self,
        ai_engine,
        debounce_time: float = 0.8,
        char_timeout: float = 0.6,
        space_timeout: float = 2.0,
    ):
        # ... 기존 코드 ...
        self._inference_lock = threading.Lock()  # ✅ 이 줄 추가

    def _do_predict_and_emit(self, session_strokes):
        predicted_word = self.engine.predict(session_strokes)

        if predicted_word:
            char = predicted_word
            now = time.time()

            # ✅ 공유 상태 접근을 락으로 보호
            with self._inference_lock:
                is_duplicate = (
                    char == self._last_emitted_char and
                    (now - self._last_emit_time) < self.debounce_time
                )
                if not is_duplicate:
                    self._last_emitted_char = char
                    self._last_emit_time = now

            # 락 밖에서 emit (I/O 작업이므로 락 유지 불필요)
            if not is_duplicate:
                self._emit_char(char)
```

---

### Issue 6 🟡 `test_ws.py` — UTF-16 인코딩

**원인:**  
`test_ws.py` 파일이 UTF-16 LE(BOM: `FF FE`)로 저장되어 있습니다.  
Python은 소스 코드를 UTF-8로 읽으므로 `python test_ws.py` 실행 시 `SyntaxError: (unicode error)` 또는 `IndentationError`가 발생합니다.

**수정 — PowerShell에서 UTF-8로 변환:**

```powershell
$content = Get-Content "test_ws.py" -Encoding Unicode
$content | Set-Content "test_ws.py" -Encoding UTF8NoBOM
```

또는 VS Code에서 파일 열고 우하단 인코딩 클릭 → **"UTF-8로 저장"** 선택.

---

### Issue 7 🟡 Windows 방화벽 포트 허용

**원인:**  
Windows Defender 방화벽이 기본적으로 인바운드 UDP 및 TCP를 차단합니다.  
ESP32 패킷 수신과 WebSocket 클라이언트 연결이 차단될 수 있습니다.

**설정 — PowerShell (관리자 권한으로 실행):**

```powershell
# WebSocket 서버 포트 12347 허용 (프론트엔드 → Python 서버)
New-NetFirewallRule -DisplayName "AirWriting WebSocket" `
    -Direction Inbound -Protocol TCP -LocalPort 12347 -Action Allow

# UDP 패킷 수신 허용 (ESP32 → PC)
# 실제 사용 포트 번호는 config/system.yaml의 network.ports에서 확인
New-NetFirewallRule -DisplayName "AirWriting UDP Receive" `
    -Direction Inbound -Protocol UDP -LocalPort 12345-12350 -Action Allow

# OLED 상태 전송 허용 (PC → ESP32, oled_sender.py 기본 포트 5555)
New-NetFirewallRule -DisplayName "AirWriting OLED UDP" `
    -Direction Outbound -Protocol UDP -RemotePort 5555 -Action Allow
```

---

### Issue 8 ✅ COM3 / 시리얼 포트 — 문제 없음

**확인 결과:**  
전체 Python 소스에서 `COM`, `serial`, `pyserial`, `UART` 키워드를 검색한 결과 **매칭 없음**.  
`windows_path_review.md`에서 언급한 `imu/serial_receiver.py`와 `main.py`의 COM3 코드는  
`git status`에서 `D main.py`로 확인되듯이 이미 삭제된 상태입니다.  
현재 코드베이스는 전체가 UDP 소켓(`socket.SOCK_DGRAM`)만 사용합니다.

향후 `main.py`를 재작성할 때 시리얼을 사용할 계획이라면 하드코딩 대신 자동 탐색 방식 사용:

```python
import serial.tools.list_ports

def find_esp32_port() -> str:
    """ESP32 연결 COM 포트 자동 탐색 (CP2102 / CH340 / FTDI 칩셋)"""
    KNOWN_VIDS = {0x10C4, 0x1A86, 0x0403}  # Silicon Labs, QinHeng, FTDI
    for p in serial.tools.list_ports.comports():
        if p.vid in KNOWN_VIDS:
            return p.device
    raise RuntimeError(
        "ESP32 포트를 찾을 수 없습니다. USB 연결 및 드라이버를 확인하세요."
    )
```

---

### Issue 9 ✅ 파일 경로 구분자 (`/` vs `\\`) — 문제 없음

**확인 결과:**  
검토한 모든 파일에서 경로 조작 시 다음 안전한 방법을 올바르게 사용 중입니다:
- `os.path.join()` — `ai_model.py`, `trajectory_renderer.py`
- `pathlib.Path` — `config_loader.py`, `iam_dataset.py`
- `open(filepath, encoding="utf-8")` — 인코딩 명시 일관

별도 수정 불필요.

---

## 수정 우선순위 및 실행 계획

```
[즉시] Step 1: config 파일 복원
  git checkout HEAD~1 -- config/imu.yaml config/system.yaml

[즉시] Step 2: test_ws.py UTF-8 변환
  PowerShell: $content = Get-Content "test_ws.py" -Encoding Unicode
              $content | Set-Content "test_ws.py" -Encoding UTF8NoBOM

[코드 수정] Step 3: ai_model.py 상단에 _PROJECT_ROOT, _WEIGHTS_DIR, _DATASET_DIR 상수 추가
           → 모든 상대경로 문자열을 절대경로로 교체

[코드 수정] Step 4: ai_model.py의 torch.load() 6개 호출에 weights_only=True 추가

[코드 수정] Step 5: streaming.py에 threading.Lock() 추가

[신규 작성] Step 6: main.py 재작성 시 파일 최상단에 asyncio Windows 정책 설정 추가

[OS 설정] Step 7: 방화벽 규칙 추가 (PowerShell 관리자 권한)
```

---

## 다음 에이전트(Gemini)를 위한 참고사항

- **삭제된 파일:** `main.py`, `config/imu.yaml`, `config/system.yaml`, `web/studio.html`은 현재 워킹트리에 없습니다. 코드 참조 시 이전 커밋 (`git show HEAD:main.py` 등)을 확인하세요.
- **가중치 파일:** `weights/pure_bilstm.pt`는 존재합니다. `weights/meta.pkl`은 `git status`상 삭제 표시이므로 재학습이 필요할 수 있습니다.
- **데이터셋:** `dataset/*.json` (약 180개) 파일은 정상 존재합니다.
- **핵심 수정 대상 파일:** `airwriting_imu/core/ai_model.py`가 가장 많은 수정이 필요합니다.
- **아키텍처:** ESP32 → UDP → Python 서버 → WebSocket → 브라우저 파이프라인입니다. 시리얼 포트는 사용하지 않습니다.
