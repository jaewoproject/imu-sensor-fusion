"""diag_serial.py — 펌웨어 수정 검증용 시리얼 진단 (일회성)

목적:
  - 94B PacketV4가 정상 수신/체크섬 통과하는지
  - finger_mag(자기장)가 **움직이는지**(=ST2 수정으로 frozen 버그 해소됐는지)
  - accel/gyro가 살아있는지, 실효 패킷레이트가 몇 Hz인지

사용:
  py -3 diag_serial.py            # COM 포트 자동 감지
  py -3 diag_serial.py COM5       # 포트 지정

손목을 천천히 돌리면서 mag std 값을 보세요:
  - mag std 가 0에 가깝게 고정 → 아직 frozen (ST2 수정 안 먹음/구버전 플래시)
  - mag std 가 회전 시 뚜렷이 커짐 → ✅ 수정 정상 (mag 살아남)
"""
import sys
import time
import collections
import numpy as np

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    print("pyserial 필요: py -3 -m pip install pyserial")
    sys.exit(1)

from airwriting_imu.core.packet_parser import PacketParser

BAUD = 921600  # main.py와 동일 (S3 네이티브 USB-CDC면 baud 무시됨)
_ESP32_VIDS = {0x10C4, 0x1A86, 0x0403}


def pick_port():
    for a in sys.argv[1:]:
        if a.startswith("-") or a.isdigit():
            continue  # 옵션이나 baud 숫자는 포트가 아님
        return a
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        return None
    vid = [p.device for p in ports if p.vid in _ESP32_VIDS]
    if vid:
        return vid[0]
    non_com1 = [p.device for p in ports if p.device != "COM1"]
    return (non_com1 or [ports[-1].device])[0]


def pick_baud():
    # argv에서 5자리 이상 숫자를 baud로 인식 (예: py diag_serial.py COM16 115200)
    for a in sys.argv[1:]:
        if a.isdigit() and int(a) >= 9600:
            return int(a)
    return BAUD


def main():
    port = pick_port()
    baud = pick_baud()
    if not port:
        print("COM 포트를 찾지 못했습니다. 장치 연결 확인 후 포트를 인자로 주세요.")
        return 2
    print(f"[diag] 포트={port} baud={baud} — Ctrl+C로 종료\n")

    parser = PacketParser(axis_remap=False)  # main.py와 동일
    buf = bytearray()
    PKT = 94

    mag_hist = collections.deque(maxlen=60)   # 최근 ~60프레임 mag
    last_print = time.time()
    n_valid = 0
    n_total = 0
    n_bytes = 0
    t0 = time.time()
    last_mag = None
    frozen_run = 0
    max_frozen_run = 0

    try:
        ser = serial.Serial(port, baud, timeout=0.1)
    except Exception as e:
        print(f"포트 열기 실패: {e}")
        return 2

    print("[diag] 데이터 수신 대기중... 보드가 포트 열 때 리셋될 수 있으니 5~10초 기다리세요.\n")
    try:
        while True:
            data = ser.read(512)
            if data:
                buf.extend(data)
                n_bytes += len(data)
            # 0xAA 헤더 + 94B + 0x55 푸터로 패킷 슬라이싱
            while len(buf) >= PKT:
                if buf[0] != 0xAA:
                    buf.pop(0)
                    continue
                if buf[PKT - 1] != 0x55:
                    buf.pop(0)
                    continue
                pkt = bytes(buf[:PKT])
                del buf[:PKT]
                n_total += 1
                frame = parser.parse(pkt)
                if frame is None:
                    continue
                n_valid += 1
                m = np.asarray(frame.finger_mag, dtype=float)
                mag_hist.append(m)
                if last_mag is not None and np.array_equal(m, last_mag):
                    frozen_run += 1
                    max_frozen_run = max(max_frozen_run, frozen_run)
                else:
                    frozen_run = 0
                last_mag = m

            now = time.time()
            if now - last_print >= 0.5:
                rate = n_total / max(now - t0, 1e-6)
                if mag_hist:
                    arr = np.array(mag_hist)
                    mstd = arr.std(axis=0)
                    mnow = arr[-1]
                    print(
                        f"rate={rate:5.1f}Hz valid={n_valid}/{n_total} cks_err={parser.checksum_errors} "
                        f"| mag=({mnow[0]:+7.1f},{mnow[1]:+7.1f},{mnow[2]:+7.1f}) "
                        f"std=({mstd[0]:5.2f},{mstd[1]:5.2f},{mstd[2]:5.2f}) "
                        f"| 연속동일mag최대={max_frozen_run}"
                    )
                else:
                    # 유효 패킷이 아직 0 → 무엇이 들어오는지 진단
                    msg = (f"[수신중] 원시바이트={n_bytes} 패킷후보={n_total} "
                           f"valid=0 cks_err={parser.checksum_errors} buf={len(buf)}")
                    if n_bytes == 0:
                        msg += "  ← 바이트가 0 (보드 리셋 대기/포트/케이블 확인)"
                    elif len(buf) >= 16:
                        hexs = " ".join(f"{b:02X}" for b in bytes(buf[:16]))
                        msg += f"\n   첫16바이트: {hexs}"
                        if 0xAA not in bytes(buf[:64]):
                            msg += "\n   ← 0xAA 헤더가 안 보임: baud 불일치(텍스트면 부팅로그) 가능"
                    print(msg)
                last_print = now
    except KeyboardInterrupt:
        print("\n[diag] 종료")
        print(f"  원시바이트={n_bytes} 패킷후보={n_total} valid={n_valid} cks_err={parser.checksum_errors}")
        if mag_hist:
            arr = np.array(mag_hist)
            tot_std = float(arr.std(axis=0).sum())
            print(f"  최근 mag 총 std={tot_std:.3f}, 최대 연속 동일 mag={max_frozen_run} 프레임")
            if max_frozen_run > 30:
                print("  ⚠️ mag가 길게 고정됨 → 아직 frozen 의심 (구버전 플래시? ST2 수정 확인)")
            else:
                print("  ✅ mag가 갱신되고 있음 (회전 시 std가 커지면 정상)")
        elif n_bytes == 0:
            print("  ❌ 바이트가 전혀 안 들어옴 → 포트 점유(다른 프로그램)/케이블/보드 확인")
        else:
            print("  ⚠️ 바이트는 오는데 유효 94B 패킷이 0 → baud 불일치 또는 패킷 포맷 상이 의심")
    finally:
        ser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
