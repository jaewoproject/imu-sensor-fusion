import os
import time
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - AIOpsDaemon - %(levelname)s - %(message)s')
logger = logging.getLogger("RealTimeMonitor")

BRIDGE_DIR = os.path.dirname(os.path.abspath(__file__))
REQUESTS_DIR = os.path.join(BRIDGE_DIR, "requests")
os.makedirs(REQUESTS_DIR, exist_ok=True)

def monitor_log_file(log_path, chunk_size=100, error_keywords=None):
    if error_keywords is None:
        error_keywords = ["Exception", "Error", "Warning", "Traceback", "Fail", "Timeout"]

    logger.info(f"Starting real-time monitoring on: {log_path}")
    logger.info("IDE Panel Collaboration Mode Enabled. Anomalies will be dumped to requests folder.")
    
    # Wait for file to exist
    while not os.path.exists(log_path):
        time.sleep(2)
        
    with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
        f.seek(0, os.SEEK_END)
        buffer = []
        
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.5)
                continue
                
            buffer.append(line)
            has_error = any(keyword.lower() in line.lower() for keyword in error_keywords)
            
            # If there's an error or we collected enough lines
            if has_error or len(buffer) >= chunk_size:
                log_chunk = "".join(buffer)
                
                # Dump only if it's an error to not spam Claude
                if has_error:
                    logger.warning("🚨 Anomaly or error detected! Creating request file for IDE Claude...")
                    timestamp = int(time.time())
                    request_file = os.path.join(REQUESTS_DIR, f"anomaly_trigger_{timestamp}.md")
                    
                    with open(request_file, "w", encoding="utf-8") as rf:
                        rf.write(f"## 실시간 런타임 이상 감지 리포트\n\n")
                        rf.write(f"**@Claude** 이 로그 청크를 분석하고 문제의 원인과 코드 개선안을 찾아주세요.\n")
                        rf.write(f"분석이 끝나면 답변을 `.agent_bridge/responses/claude_insight_{timestamp}.md` 파일로 저장해 주세요. Gemini가 읽고 코드를 수정할 예정입니다.\n\n")
                        rf.write(f"### [로그 내용]\n```text\n{log_chunk}\n```\n")
                        
                    logger.info(f"✅ Dumped to {request_file}. You can now point IDE Claude to this file!")
                
                buffer = [] # clear buffer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time Log Monitor for IDE Agents")
    parser.add_argument("--log", type=str, required=True, help="Path to the log file to monitor")
    args = parser.parse_args()
    
    try:
        monitor_log_file(args.log)
    except KeyboardInterrupt:
        logger.info("Monitoring stopped by user.")
