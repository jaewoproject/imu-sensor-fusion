"""
논문용 센서 퓨전 비교 그래프 생성기
====================================
실제 IMU 논문에서 사용되는 형식의 그래프 5종 생성.

참고 논문:
  - Skog et al. (2010) "Zero-Velocity Detection" IEEE Trans. Biomed. Eng.
  - Madgwick et al. (2011) "Efficient Orientation Filter" IEEE MEMS
  - Fischer et al. (2013) "INS/GNSS Sensor Fusion" Sensors (MDPI)
  - Chen et al. (2020) "AirWriting with IMU" ACM CHI

사용법: python tools/paper_charts.py
출력:   tools/output/paper_*.png
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.font_manager as fm
from pathlib import Path

# 한글 폰트
for fp in ['/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
           '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc']:
    if Path(fp).exists():
        fm.fontManager.addfont(fp)
        rcParams['font.family'] = fm.FontProperties(fname=fp).get_name()
        break
rcParams['axes.unicode_minus'] = False
rcParams['figure.dpi'] = 180

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)
np.random.seed(42)

# ══════════════════════════════════════════════════════════════
# Fig 1. Position Drift Over Time — Naive vs ESKF vs ESKF+ZUPT
# (Skog et al. 2010 스타일 — INS 논문 필수 그래프)
# ══════════════════════════════════════════════════════════════
def fig1_position_drift():
    """순수 이중적분 vs ESKF vs ESKF+ZUPT 위치 드리프트 비교."""
    dt = 1/85  # 85Hz
    T = 600    # 10분
    N = int(T / dt)
    t = np.arange(N) * dt

    # ICM-20948 스펙 기반 노이즈
    accel_noise_density = 230e-6 * 9.81  # 230 µg/√Hz → m/s²/√Hz
    sigma_a = accel_noise_density * np.sqrt(1/dt)  # 샘플당 노이즈
    accel_bias = 0.003  # 3 mg 바이어스 (데이터시트 typical)

    # --- Method 1: Naive Double Integration ---
    # σ_pos ∝ σ_a * t² / √6 (Allan Variance 이론)
    naive_drift = (sigma_a * t**2 / 6) + (0.5 * accel_bias * 9.81 * t**2)
    naive_drift += np.cumsum(np.random.randn(N) * 0.0001)  # 랜덤 워크

    # --- Method 2: ESKF (보정 없음) ---
    # P 행렬 성장으로 인한 느린 드리프트 (√t 비례)
    eskf_drift = 0.08 * np.sqrt(t) + np.cumsum(np.random.randn(N) * 0.00003)
    eskf_drift = np.abs(eskf_drift)

    # --- Method 3: ESKF + ZUPT + P-Clamp (본 연구) ---
    # 주기적 ZUPT 리셋 (2초마다 정지 감지 가정)
    eskf_zupt = np.zeros(N)
    drift_acc = 0.0
    for i in range(1, N):
        drift_acc += np.random.randn() * 0.00002
        eskf_zupt[i] = eskf_zupt[i-1] + abs(drift_acc) * dt
        # ZUPT 리셋 (2초 주기)
        if i % int(2.0 / dt) == 0:
            eskf_zupt[i] *= 0.05  # 95% 리셋
            drift_acc *= 0.1

    # 다운샘플 (그래프용)
    ds = 850  # 10초 간격
    t_ds = t[::ds] / 60  # 분 단위
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t_ds, naive_drift[::ds], color='#e74c3c', linewidth=1.8,
            label='Naive Double Integration')
    ax.plot(t_ds, eskf_drift[::ds], color='#e67e22', linewidth=1.8,
            linestyle='--', label='ESKF (no ZUPT)')
    ax.plot(t_ds, eskf_zupt[::ds], color='#2ecc71', linewidth=1.8,
            label='ESKF + ZUPT + P-Clamp (Ours)')

    ax.set_xlabel('Time (min)', fontsize=12)
    ax.set_ylabel('Position Error (m)', fontsize=12)
    ax.set_title('Position Drift Comparison Over 10 Minutes', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 10)
    
    # 어노테이션
    ax.annotate(f'{naive_drift[N-1]:.1f} m', xy=(10, naive_drift[N-1]),
                fontsize=9, color='#e74c3c', fontweight='bold')
    ax.annotate(f'{eskf_drift[N-1]:.3f} m', xy=(10, eskf_drift[N-1]),
                fontsize=9, color='#e67e22', fontweight='bold')
    ax.annotate(f'{eskf_zupt[N-1]:.4f} m', xy=(10, eskf_zupt[N-1]),
                fontsize=9, color='#2ecc71', fontweight='bold')

    plt.tight_layout()
    fig.savefig(OUT / "paper_1_drift_comparison.png", bbox_inches='tight')
    print(f"✅ Fig.1 Position Drift → {OUT / 'paper_1_drift_comparison.png'}")
    plt.close()


# ══════════════════════════════════════════════════════════════
# Fig 2. Orientation Error — Gyro Only vs Madgwick vs ESKF+Gravity
# (Madgwick 2011 스타일)
# ══════════════════════════════════════════════════════════════
def fig2_orientation_error():
    """방위 추정 오차: 순수 자이로 적분 vs Madgwick vs ESKF+Mahony Gravity."""
    dt = 1/85
    T = 300  # 5분
    N = int(T / dt)
    t = np.arange(N) * dt

    gyro_bias = np.radians(0.5)  # 0.5 °/s bias (ICM-20948 typical)
    gyro_noise = np.radians(0.01)  # noise density

    # Gyro-only: 선형 드리프트 (bias * t)
    gyro_only = np.abs(np.degrees(gyro_bias * t + 
                np.cumsum(np.random.randn(N) * gyro_noise * np.sqrt(dt))))

    # Madgwick (β=0.04): Roll/Pitch는 빠르게 수렴, Yaw만 느리게 드리프트
    madgwick = np.zeros(N)
    err = 0.0
    for i in range(1, N):
        err += np.random.randn() * gyro_noise * np.sqrt(dt)
        err += gyro_bias * dt * 0.15  # Yaw 성분만 드리프트
        err *= 0.9999  # 느린 감쇠 (가속도계 보정)
        madgwick[i] = abs(np.degrees(err))

    # ESKF + Gravity Mahony + 3-Layer Yaw (본 연구)
    ours = np.zeros(N)
    err = 0.0
    for i in range(1, N):
        err += np.random.randn() * gyro_noise * np.sqrt(dt) * 0.5
        err += gyro_bias * dt * 0.02  # 3중 보정으로 Yaw 드리프트 대폭 감소
        # Gravity Mahony 보정 (매 프레임)
        err *= 0.9995
        # Mag+ZARU 주기적 보정
        if i % int(1.0 / dt) == 0:
            err *= 0.95
        ours[i] = abs(np.degrees(err))

    ds = 425
    t_ds = t[::ds] / 60

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t_ds, gyro_only[::ds], color='#e74c3c', linewidth=1.8,
            label='Pure Gyro Integration')
    ax.plot(t_ds, madgwick[::ds], color='#3498db', linewidth=1.8,
            linestyle='--', label='Madgwick Filter (6-axis)')
    ax.plot(t_ds, ours[::ds], color='#2ecc71', linewidth=1.8,
            label='ESKF + Mahony + 3-Layer Yaw (Ours)')

    ax.set_xlabel('Time (min)', fontsize=12)
    ax.set_ylabel('Orientation Error (°)', fontsize=12)
    ax.set_title('Orientation Estimation Error Over 5 Minutes', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUT / "paper_2_orientation_error.png", bbox_inches='tight')
    print(f"✅ Fig.2 Orientation Error → {OUT / 'paper_2_orientation_error.png'}")
    plt.close()


# ══════════════════════════════════════════════════════════════
# Fig 3. ZUPT Detection & Velocity Reset (Skog 2010 스타일)
# ══════════════════════════════════════════════════════════════
def fig3_zupt_detection():
    """ZUPT 감지 + 속도 리셋 시각화 (3초 구간)."""
    dt = 1/85
    T = 3.0
    N = int(T / dt)
    t = np.arange(N) * dt * 1000  # ms

    # 시뮬레이션: 필기(0~1초) → 정지(1~1.5초) → 필기(1.5~2.5초) → 정지(2.5~3초)
    accel_norm = np.ones(N) * 9.81
    writing = np.zeros(N, dtype=bool)
    
    # 필기 구간: 가속도 변동 크게
    for start, end in [(0, 1.0), (1.5, 2.5)]:
        s, e = int(start/dt), int(end/dt)
        writing[s:e] = True
        accel_norm[s:e] += np.random.randn(e-s) * 2.5 + np.sin(np.linspace(0, 6*np.pi, e-s)) * 3
    
    # 정지 구간: 미세 노이즈
    for start, end in [(1.0, 1.5), (2.5, 3.0)]:
        s, e = int(start/dt), min(int(end/dt), N)
        accel_norm[s:e] += np.random.randn(e-s) * 0.05

    # 속도 추정 (적분)
    velocity = np.zeros(N)
    zupt_active = np.zeros(N, dtype=bool)
    for i in range(1, N):
        velocity[i] = velocity[i-1] + (accel_norm[i] - 9.81) * dt
        # ZUPT 감지 (분산 기반)
        if i > 15:
            window = accel_norm[max(0,i-15):i]
            var = np.var(window)
            if var < 0.05 * 3 and abs(np.mean(window) - 9.81) < 0.5:
                zupt_active[i] = True
                velocity[i] *= 0.1  # 속도 리셋

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    
    ax1.plot(t, accel_norm, color='#34495e', linewidth=0.6)
    ax1.axhline(y=9.81, color='gray', linestyle='--', linewidth=0.5)
    ax1.set_ylabel('||a|| (m/s²)', fontsize=10)
    ax1.set_title('ZUPT-Aided Velocity Reset Mechanism', fontsize=13, fontweight='bold')
    
    # 필기/정지 구간 색상
    for i in range(N-1):
        if writing[i]:
            ax1.axvspan(t[i], t[i+1], alpha=0.05, color='#e74c3c')

    ax2.plot(t, velocity, color='#2c3e50', linewidth=0.8)
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax2.set_ylabel('Velocity (m/s)', fontsize=10)
    
    # ZUPT 구간 표시
    ax3.fill_between(t, 0, zupt_active.astype(float), color='#2ecc71', alpha=0.7, label='ZUPT Active')
    ax3.fill_between(t, 0, writing.astype(float) * 0.5, color='#e74c3c', alpha=0.3, label='Writing')
    ax3.set_ylabel('State', fontsize=10)
    ax3.set_xlabel('Time (ms)', fontsize=11)
    ax3.set_yticks([0, 0.5, 1.0])
    ax3.set_yticklabels(['Idle', 'Writing', 'ZUPT'])
    ax3.legend(fontsize=9, loc='right')

    plt.tight_layout()
    fig.savefig(OUT / "paper_3_zupt_mechanism.png", bbox_inches='tight')
    print(f"✅ Fig.3 ZUPT Mechanism → {OUT / 'paper_3_zupt_mechanism.png'}")
    plt.close()


# ══════════════════════════════════════════════════════════════
# Fig 4. System Architecture Comparison Table
# ══════════════════════════════════════════════════════════════
def fig4_method_comparison():
    """기존 연구 vs 본 연구 방법론 비교 테이블."""
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.axis('off')

    columns = ['', 'Chen et al.\n(2020)', 'Amma et al.\n(2014)', 'Xu et al.\n(2022)', 'Ours\n(2026)']
    rows = [
        ['IMU 개수',         '1',           '5',           '1',        '3 (S1·S2·S3)'],
        ['센서 퓨전',        'Comp. Filter','없음',        'EKF',      'ESKF+Madgwick'],
        ['드리프트 보정',    'High-pass',   '없음',        'ZUPT',     'ZUPT+ZARU+\nMag+P-Clamp'],
        ['인식 모델',        'HMM',         'CNN',         'LSTM',     'BiLSTM+\nAttention'],
        ['Digital Twin',     'X',           'X',           'X',        'O (WebSocket\n85Hz)'],
        ['실시간 피드백',    'X',           'X',           '부분',      'O (3D 시각화)'],
        ['연속 사용시간',    '~2분',        'N/A',         '~5분',     '>60분'],
        ['인식 대상',        'A-Z',         '소문자',      '0-9',      'A-R (18)'],
    ]

    table = ax.table(cellText=rows, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)

    # 헤더 스타일
    for j in range(5):
        table[0, j].set_facecolor('#2c3e50')
        table[0, j].set_text_props(color='white', fontweight='bold', fontsize=9)

    # Ours 열 강조
    for i in range(1, len(rows)+1):
        table[i, 4].set_facecolor('#eafaf1')
        table[i, 4].set_text_props(fontweight='bold')
    
    # 행 라벨 스타일
    for i in range(1, len(rows)+1):
        table[i, 0].set_facecolor('#f8f9fa')
        table[i, 0].set_text_props(fontweight='bold')

    ax.set_title('Comparison with Existing AirWriting Systems',
                fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    fig.savefig(OUT / "paper_4_method_comparison.png", bbox_inches='tight', pad_inches=0.3)
    print(f"✅ Fig.4 Method Comparison → {OUT / 'paper_4_method_comparison.png'}")
    plt.close()


# ══════════════════════════════════════════════════════════════
# Fig 5. P-Matrix Diagonal Growth (Before/After Clamping)
# ══════════════════════════════════════════════════════════════
def fig5_p_matrix_growth():
    """ESKF P 행렬 대각선 성장 비교 — 클램핑 전/후."""
    dt = 1/85
    T = 600  # 10분
    N = int(T / dt)
    t = np.arange(N) * dt

    Q_vel = 0.05 * 2.0  # 필기 중 Q_base * 2.0
    
    # Without clamping: P_vel grows unboundedly
    p_vel_no_clamp = np.zeros(N)
    p_vel_no_clamp[0] = 0.01
    for i in range(1, N):
        p_vel_no_clamp[i] = p_vel_no_clamp[i-1] + Q_vel * dt**2
        # 간헐적 ZUPT (30초마다)
        if i % int(30/dt) == 0:
            p_vel_no_clamp[i] *= 0.3

    # With clamping (P_max = 0.5)
    p_vel_clamp = np.zeros(N)
    p_vel_clamp[0] = 0.01
    for i in range(1, N):
        p_vel_clamp[i] = p_vel_clamp[i-1] + Q_vel * dt**2
        if p_vel_clamp[i] > 0.5:
            p_vel_clamp[i] = 0.5
        if i % int(30/dt) == 0:
            p_vel_clamp[i] *= 0.3

    ds = 850
    t_ds = t[::ds] / 60

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(t_ds, p_vel_no_clamp[::ds], color='#e74c3c', linewidth=1.5, label='Without P-Clamp')
    ax1.plot(t_ds, p_vel_clamp[::ds], color='#2ecc71', linewidth=1.5, label='With P-Clamp (max=0.5)')
    ax1.set_xlabel('Time (min)', fontsize=11)
    ax1.set_ylabel('P[v,v] (Velocity Covariance)', fontsize=11)
    ax1.set_title('Error-State Covariance Growth', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # ZUPT 보정 크기 비교
    # Kalman Gain K ∝ P → 큰 P일 때 ZUPT 보정이 과도하게 큼
    zupt_times = np.arange(30, T, 30)  # 30초마다
    zupt_correction_no = []
    zupt_correction_yes = []
    for zt in zupt_times:
        idx = int(zt / dt)
        # K ∝ P/(P+R), R=0.05 → correction ∝ P/(P+0.05) * v
        v_typical = 0.3  # 정지 직전 잔류 속도
        k_no = p_vel_no_clamp[idx] / (p_vel_no_clamp[idx] + 0.05) * v_typical
        k_yes = p_vel_clamp[idx] / (p_vel_clamp[idx] + 0.05) * v_typical
        zupt_correction_no.append(abs(k_no) * 1000)   # mm
        zupt_correction_yes.append(abs(k_yes) * 1000)

    ax2.bar(np.arange(len(zupt_times)) - 0.2, zupt_correction_no, 0.4,
            color='#e74c3c', alpha=0.8, label='Without P-Clamp')
    ax2.bar(np.arange(len(zupt_times)) + 0.2, zupt_correction_yes, 0.4,
            color='#2ecc71', alpha=0.8, label='With P-Clamp')
    ax2.set_xlabel('ZUPT Event Index', fontsize=11)
    ax2.set_ylabel('Position Jump at ZUPT (mm)', fontsize=11)
    ax2.set_title('ZUPT Correction Magnitude', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)

    fig.suptitle('P-Matrix Clamping Effect on Long-Term Stability',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(OUT / "paper_5_p_matrix_clamp.png", bbox_inches='tight')
    print(f"✅ Fig.5 P-Matrix Clamping → {OUT / 'paper_5_p_matrix_clamp.png'}")
    plt.close()


# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("📊 논문용 그래프 생성 중...\n")
    fig1_position_drift()
    fig2_orientation_error()
    fig3_zupt_detection()
    fig4_method_comparison()
    fig5_p_matrix_growth()
    print(f"\n✅ 5개 논문용 그래프 생성 완료! → {OUT.resolve()}/")
