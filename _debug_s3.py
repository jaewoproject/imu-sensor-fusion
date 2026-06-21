"""S3 진단 — packet_parser가 S3 자리에서 실제로 읽는 raw 값 dump.

사용:
  1. main.py 먼저 종료 (COM 포트 점유 충돌 방지)
  2. py -3 _debug_s3.py
  3. 30 패킷 출력 후 자동 종료
"""
import sys, serial, time
sys.stdout.reconfigure(encoding="utf-8")

from airwriting_imu.core.packet_parser import PacketParser

# main.py의 SERIAL_PORT 자동 감지 로직 동일
import serial.tools.list_ports
ports = list(serial.tools.list_ports.comports())
ESP32_VIDS = {0x10C4, 0x1A86, 0x0403}
SERIAL_PORT = "COM3"
if ports:
    vid_match = [p.device for p in ports if p.vid in ESP32_VIDS]
    non_com1 = [p.device for p in ports if p.device != "COM1"]
    if vid_match:
        SERIAL_PORT = vid_match[0]
    elif non_com1:
        SERIAL_PORT = non_com1[0]

print(f"[debug-s3] opening {SERIAL_PORT} @ 921600...", flush=True)
parser = PacketParser(axis_remap=False)
ser = serial.Serial(SERIAL_PORT, 921600, timeout=1, inter_byte_timeout=0.001)

buf = bytearray()
count = 0
size_counts = {92: 0, 94: 0, 68: 0, 70: 0}

while count < 30:
    data = ser.read(max(1, ser.in_waiting))
    if not data:
        continue
    buf.extend(data)
    while len(buf) >= 68:
        idx = buf.find(0xAA)
        if idx < 0:
            buf.clear()
            break
        if idx > 0:
            buf = buf[idx:]
        found = False
        for psize in [92, 94, 68, 70]:
            if len(buf) >= psize and buf[psize - 1] == 0x55:
                packet = bytes(buf[:psize])
                buf = buf[psize:]
                size_counts[psize] += 1
                frame = parser.parse(packet)
                if frame is not None:
                    s1 = frame.wrist_accel
                    s2 = frame.hand_accel if frame.hand_accel is not None else (0, 0, 0)
                    s3a = frame.finger_accel
                    s3g = frame.finger_gyro
                    s3m = frame.finger_mag if hasattr(frame, "finger_mag") and frame.finger_mag is not None else (0, 0, 0)
                    print(
                        f"#{count:02d} size={psize} btn={frame.button} | "
                        f"S1.a=({s1[0]:+.2f},{s1[1]:+.2f},{s1[2]:+.2f}) | "
                        f"S2.a=({s2[0]:+.2f},{s2[1]:+.2f},{s2[2]:+.2f}) | "
                        f"S3.a=({s3a[0]:+.2f},{s3a[1]:+.2f},{s3a[2]:+.2f}) | "
                        f"S3.g=({s3g[0]:+.3f},{s3g[1]:+.3f},{s3g[2]:+.3f}) | "
                        f"S3.m=({s3m[0]:+.2f},{s3m[1]:+.2f},{s3m[2]:+.2f})",
                        flush=True,
                    )
                    count += 1
                found = True
                break
        if not found:
            if len(buf) > 100:
                buf = buf[1:]
            else:
                break

ser.close()
print(f"\n[debug-s3] packet sizes received: {size_counts}", flush=True)

# Diagnose
print("\n=== diagnosis ===", flush=True)
