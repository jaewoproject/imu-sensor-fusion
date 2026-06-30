"""
streaming.py — 연속 문장 스트리밍 인식 로직

프론트엔드 의존 없이, 백엔드에서 센서 데이터를 실시간으로 모아
글자를 판별하고 문장을 이어나가는 StreamingInference 모듈입니다.

[핵심 수정] 버퍼를 flat list → list of strokes 구조로 변경.
녹화 데이터와 동일한 획 분리(is_new_stroke 마킹)를 보장하여
학습/추론 데이터 불일치 문제를 해결합니다.

[v2 Fix] 인식 씹힘 버그 3건 수정:
  1. _inference_running 중 요청 → pending 큐로 재시도 보장
  2. 버퍼를 executor에 복사한 후 비우기 (경쟁 조건 제거)
  3. 같은 글자 연속 입력 허용 (디바운스 0.8s → 0.3s)
"""

import time
import logging
import threading
from typing import Optional, Callable, List, Dict

logger = logging.getLogger(__name__)

class StreamingInference:
    def __init__(
        self,
        ai_engine,
        debounce_time: float = 0.3,  # [v2] 0.8s → 0.3s: 같은 글자 연속 입력(LL, OO 등) 허용
        char_timeout: float = 1.0,
        space_timeout: float = 2.0,
    ):
        """
        Args:
            ai_engine: AirWritingAI 인스턴스 (predict 메서드 보유)
            debounce_time: 동일 문자의 중복 출력을 무시하는 최소 시간 (초)
            char_timeout: is_writing=False가 유지되면 한 글자로 판정할 시간 (초)
            space_timeout: is_writing=False가 유지되면 띄어쓰기로 판정할 시간 (초)
        """
        self.engine = ai_engine
        self.debounce_time = debounce_time
        self.char_timeout = char_timeout
        self.space_timeout = space_timeout

        # [핵심] 획 단위 버퍼: 녹화 데이터와 동일한 [[획1], [획2], ...] 구조
        self._strokes: List[List[Dict]] = []
        self._frame_count = 0
        self._max_frames = 300  # 총 프레임 수 제한 (약 3.5초)

        # State tracking
        self._last_emitted_char: Optional[str] = None
        self._last_emit_time: float = 0.0
        self._last_writing_time: float = time.time()
        self._space_emitted = False
        self._last_frame_is_writing = False
        
        self._tentative_char: Optional[str] = None
        self._inference_running = False  # 추론 중복 실행 방지 플래그
        self._pending_inference = False  # [v2] 추론 중 새 요청이 들어왔을 때 재시도 플래그

        self.on_text_updated: Optional[Callable[[str, str, float], None]] = None
        self._current_sentence = ""
        self._inference_lock = threading.Lock()

    def _total_frames(self) -> int:
        """현재 버퍼에 쌓인 총 프레임 수"""
        return sum(len(s) for s in self._strokes)

    def process_frame(self, frame_data: dict, is_writing: bool):
        """매 프레임 호출되어 텍스트 스트리밍을 제어합니다."""
        now = time.time()

        if is_writing:
            self._last_writing_time = now
            self._space_emitted = False
            self._frame_count += 1

            # [핵심] is_writing 전환 시 새 획 시작 — 녹화와 동일한 구조
            if not self._last_frame_is_writing:
                # 이전에 쓰고 있지 않았으므로 새 획 시작
                self._strokes.append([])
            
            # 현재 획에 프레임 추가
            self._strokes[-1].append(frame_data)
            self._last_frame_is_writing = True
            
            # 총 프레임 수 제한 (오래된 획부터 제거)
            while self._total_frames() > self._max_frames and len(self._strokes) > 1:
                self._strokes.pop(0)

        else:    # 글씨를 안 쓰고 있을 때 타임아웃 판정
            self._last_frame_is_writing = False
            duration_since_writing = now - self._last_writing_time
            
            # 1. 단일 글자 완성 판정 (char_timeout)
            if duration_since_writing > self.char_timeout and len(self._strokes) > 0:
                total = self._total_frames()
                if total > 3:  # 최소 프레임 수
                    if self._inference_running:
                        # [v2 Fix] 추론 진행 중 → pending 플래그 설정, 추론 완료 시 재시도
                        self._pending_inference = True
                    else:
                        self._run_inference(is_partial=False)
                        # [v2 Fix] 버퍼 비우기는 _run_inference 내부에서 복사 후 수행
                else:
                    self._strokes = []
                    self._tentative_char = None

            # 2. 띄어쓰기 판정 (space_timeout)
            if duration_since_writing > self.space_timeout and not self._space_emitted:
                if len(self._current_sentence) > 0 and not self._current_sentence.endswith(" "):
                    self._emit_char(" ")
                self._space_emitted = True
                self._last_emitted_char = None # 새 단어이므로 중복방지 해제

    def _run_inference(self, is_partial: bool = False):
        """엔진을 돌려 문자를 가져오고 디바운싱합니다. (Asyncio 이벤트 루프 차단 방지)"""
        if not self.engine or len(self._strokes) == 0:
            return
        if self._inference_running:  # 이전 추론이 아직 실행 중이면 스킵 (태스크 누적 방지)
            self._pending_inference = True  # [v2] 나중에 재시도
            return

        self._inference_running = True
        self._pending_inference = False
        
        # [v2 Fix] 버퍼를 먼저 복사한 후 비움 (경쟁 조건 제거)
        # 녹화 경로와 정합: 최소 3포인트 미만 획 제거 (main.py:892)
        session_strokes = [list(s) for s in self._strokes if len(s) > 2]
        self._strokes = []  # 복사 완료 후 버퍼 비움
        self._tentative_char = None
        
        if not session_strokes:
            self._inference_running = False
            return
        
        n_strokes = len(session_strokes)
        n_frames = sum(len(s) for s in session_strokes)
        logger.info(f"🔍 Inference: {n_strokes} strokes, {n_frames} frames")

        import asyncio
        try:
            loop = asyncio.get_running_loop()
            loop.run_in_executor(None, self._do_predict_and_emit, session_strokes, is_partial)
        except RuntimeError:
            self._do_predict_and_emit(session_strokes, is_partial)

    def _do_predict_and_emit(self, session_strokes, is_partial: bool):
        try:
            result = self.engine.predict(session_strokes)

            if not result:
                return

            if isinstance(result, tuple) and len(result) == 2:
                predicted_word, conf = result
            else:
                predicted_word = result
                conf = 1.0

            if not predicted_word:
                return

            if is_partial and conf < 0.65:  # 부분 추론 임계 완화 (반응성↑)
                return

            char = predicted_word
            now = time.time()

            chars_to_emit = []

            with self._inference_lock:
                if is_partial:
                    if self._tentative_char != char:
                        if self._tentative_char is not None:
                            chars_to_emit.append("<ERASE>")
                        self._tentative_char = char
                        chars_to_emit.append(char)
                        self._last_emitted_char = char
                        self._last_emit_time = now
                else:
                    if self._tentative_char is not None:
                        if self._tentative_char != char:
                            chars_to_emit.append("<ERASE>")
                            chars_to_emit.append(char)
                        self._tentative_char = None
                        self._last_emitted_char = char
                        self._last_emit_time = now
                    else:
                        is_duplicate = (char == self._last_emitted_char) and \
                                       ((now - self._last_emit_time) < self.debounce_time)
                        if not is_duplicate:
                            self._last_emitted_char = char
                            self._last_emit_time = now
                            chars_to_emit.append(char)
                        else:
                            logger.debug(f"⏭️ Debounce skip: '{char}' (last emit {now - self._last_emit_time:.2f}s ago)")

            # 락 외부에서 I/O 발생 (데드락 방지)
            for c in chars_to_emit:
                self._emit_char(c, conf)
        finally:
            self._inference_running = False
            
            # [v2 Fix] 추론 완료 후 pending 요청이 있으면 즉시 재시도
            if self._pending_inference and len(self._strokes) > 0:
                self._pending_inference = False
                total = self._total_frames()
                if total > 3:
                    logger.info("🔄 Pending inference retry (was blocked during previous inference)")
                    self._run_inference(is_partial=False)

    def _emit_char(self, char: str, confidence: float = 0.0):
        if char == "<ERASE>":
            if len(self._current_sentence) > 0:
                self._current_sentence = self._current_sentence[:-1]
            logger.info(f"🔙 Stream Erased -> Sentence: '{self._current_sentence}'")
        else:
            self._current_sentence += char
            logger.info(f"📝 Stream Emitted: '{char}' ({confidence:.0%}) -> Sentence: '{self._current_sentence}'")
            
        if self.on_text_updated:
            self.on_text_updated(self._current_sentence, char, confidence)

    def reset(self):
        self._strokes = []
        self._frame_count = 0
        self._current_sentence = ""
        self._last_emitted_char = None
        self._last_emit_time = 0.0
        self._last_writing_time = time.time()
        self._space_emitted = False
        self._last_frame_is_writing = False
        self._tentative_char = None
        self._inference_running = False
        self._pending_inference = False
        if self.on_text_updated:
            self.on_text_updated("", "", 0.0)
