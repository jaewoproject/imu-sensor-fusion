"""
노이즈 측정 & 비교 도구
========================
1. main.py 캘리브레이션 시 자동으로 노이즈 데이터 저장
2. 빵판 vs PCB 비교 차트 자동 생성

사용법:
  [자동] main.py 서버 실행 → 캘리브레이션 완료 시 자동 저장
  [비교] python tools/noise_compare.py

저장 위치: tools/noise_logs/
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime

NOISE_LOG_DIR = Path(__file__).parent / "noise_logs"
NOISE_LOG_DIR.mkdir(exist_ok=True)


def save_noise_snapshot(calibrator, label: str = "breadboard"):
    """
    캘리브레이션 완료 시 호출하여 노이즈 통계를 JSON으로 저장.
    
    Args:
        calibrator: Calibrator 인스턴스 (samples 딕셔너리 보유)
        label: "breadboard" 또는 "pcb"
    """
    stats = {}
    
    for sid, accel_key, gyro_key in [
        ('s1', 's1_a', 's1_g'),
        ('s2', 's2_a', 's2_g'),
        ('s3', 's3_a', 's3_g'),
    ]:
        accel_samples = calibrator.samples.get(accel_key, [])
        gyro_samples = calibrator.samples.get(gyro_key, [])
        
        if len(accel_samples) < 10:
            continue
            
        acc = np.array(accel_samples)
        gyr = np.array(gyro_samples) if len(gyro_samples) > 0 else np.zeros((len(acc), 3))
        
        # 축별 표준편차 (노이즈 RMS)
        acc_std = np.std(acc, axis=0)
        gyr_std = np.std(gyr, axis=0)
        
        # 축별 평균 (바이어스)
        acc_mean = np.mean(acc, axis=0)
        gyr_mean = np.mean(gyr, axis=0)
        
        # 종합 norm
        acc_std_norm = float(np.linalg.norm(acc_std))
        gyr_std_norm = float(np.linalg.norm(gyr_std))
        
        # Peak-to-Peak (최대-최소 범위)
        acc_ptp = np.ptp(acc, axis=0)
        gyr_ptp = np.ptp(gyr, axis=0)
        
        stats[sid] = {
            "accel": {
                "std_xyz": acc_std.tolist(),      # [m/s²] 축별 노이즈 RMS
                "std_norm": acc_std_norm,          # [m/s²] 종합 노이즈
                "mean_xyz": acc_mean.tolist(),     # [m/s²] 축별 평균 (바이어스)
                "ptp_xyz": acc_ptp.tolist(),       # [m/s²] 축별 peak-to-peak
                "mean_norm": float(np.linalg.norm(acc_mean)),
            },
            "gyro": {
                "std_xyz_dps": np.degrees(gyr_std).tolist(),  # [°/s] 축별 노이즈 RMS
                "std_norm_dps": float(np.degrees(gyr_std_norm)),
                "mean_xyz_dps": np.degrees(gyr_mean).tolist(),
                "ptp_xyz_dps": np.degrees(gyr_ptp).tolist(),
                "std_xyz_rads": gyr_std.tolist(),  # [rad/s] 원본 단위
            },
            "n_samples": len(accel_samples),
        }
    
    # 자기장 데이터 (S3만)
    mag_samples = calibrator.samples.get('s3_m', [])
    if len(mag_samples) > 5:
        mag = np.array(mag_samples)
        stats["s3"]["mag"] = {
            "std_xyz": np.std(mag, axis=0).tolist(),
            "mean_xyz": np.mean(mag, axis=0).tolist(),
            "mean_norm": float(np.linalg.norm(np.mean(mag, axis=0))),
        }
    
    # 원본 시계열 저장 (파형 비교용, S3 가속도 Z축 100샘플만)
    s3_accel = calibrator.samples.get('s3_a', [])
    if len(s3_accel) > 50:
        arr = np.array(s3_accel)
        stats["waveform_s3_az"] = arr[:100, 2].tolist()  # Z축 100샘플
        stats["waveform_s3_ax"] = arr[:100, 0].tolist()
        stats["waveform_s3_ay"] = arr[:100, 1].tolist()
    
    # 메타데이터
    snapshot = {
        "label": label,
        "timestamp": datetime.now().isoformat(),
        "sample_rate_hz": 85,
        "sensors": stats,
    }
    
    # 파일명: label_날짜시간.json
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = NOISE_LOG_DIR / f"{label}_{ts}.json"
    filename.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding='utf-8')
    
    # 요약 출력
    if 's3' in stats:
        s3 = stats['s3']
        print(f"📏 [{label.upper()}] 노이즈 기록 완료 → {filename.name}")
        print(f"   가속도 σ: X={s3['accel']['std_xyz'][0]:.4f} Y={s3['accel']['std_xyz'][1]:.4f} Z={s3['accel']['std_xyz'][2]:.4f} m/s² (norm={s3['accel']['std_norm']:.4f})")
        print(f"   자이로  σ: X={s3['gyro']['std_xyz_dps'][0]:.4f} Y={s3['gyro']['std_xyz_dps'][1]:.4f} Z={s3['gyro']['std_xyz_dps'][2]:.4f} °/s (norm={s3['gyro']['std_norm_dps']:.4f})")
    
    return str(filename)


def get_hardware_label():
    """현재 하드웨어 라벨을 반환 (설정 파일 기반)."""
    label_file = NOISE_LOG_DIR / "current_hardware.txt"
    if label_file.exists():
        return label_file.read_text().strip()
    return "breadboard"  # 기본값


def set_hardware_label(label: str):
    """하드웨어 라벨 변경 (pcb로 바꿀 때 호출)."""
    label_file = NOISE_LOG_DIR / "current_hardware.txt"
    label_file.write_text(label, encoding='utf-8')
    print(f"🔧 하드웨어 라벨 변경: {label}")
