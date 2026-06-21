# [에러 보고] main.py 파일 누락 및 재작성 요청

현재 워크스페이스에 `main.py`가 삭제되어 서버를 구동할 수 없습니다. (`python main.py` 실행 시 `[Errno 2] No such file or directory` 발생)
기존 시스템 아키텍처와 방금 우리가 패치한 윈도우 호환성 환경을 바탕으로 `main.py`를 완벽하게 재작성해 주세요.

## 📌 시스템 현황 및 요구사항

1. **Windows 비동기 환경 대응**:
   - `asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())`가 파일 최상단에 반드시 포함되어야 합니다. (이전에 당신이 지적한 UDP 호환성 이슈)

2. **포트 및 네트워크 설정 (`config/system.yaml` 기반)**:
   - `esp_receive`: 12345 (UDP, ESP32 센서 패킷 수신)
   - `dashboard`: 12347 (WebSocket, 웹 프론트엔드 텍스트 전송)

3. **핵심 AI 모듈 및 스트리밍 연동**:
   - `airwriting_imu.core.ai_model.AirWritingAI` 인스턴스화 및 `.load_model()` 호출.
   - `airwriting_imu.core.streaming.StreamingInference`를 초기화할 때 앞서 만든 AI 엔진을 주입.
   - 비동기 UDP 루프에서 수신한 센서 패킷을 파싱하여 `streaming.process_frame(data, is_writing)`으로 전달.
   - 스트리밍에서 문자가 완성되었을 때 호출되는 `on_text_updated` 콜백을 통해 WebSocket 클라이언트들에게 `{"type": "text", "sentence": "...", "char": "..."}` JSON 포맷으로 브로드캐스트.

4. **로깅 및 예외 처리**:
   - 터미널에 포트 오픈 상태 및 실시간 AI 인식 글자를 깔끔하게 출력.
   - `Ctrl+C` (KeyboardInterrupt) 입력 시 비동기 루프 및 소켓들이 깔끔하게 종료되도록 Graceful Shutdown 구현.

위 스펙을 모두 만족하는 완전한 `main.py` 파이썬 코드를 작성해 주세요.
