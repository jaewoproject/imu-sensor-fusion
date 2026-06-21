"""
빵판 vs PCB 노이즈 비교 차트 생성기
======================================
tools/noise_logs/ 폴더에 저장된 측정 데이터를 읽어 비교 차트를 자동 생성합니다.

사용법:
  python tools/noise_compare.py
  python tools/noise_compare.py --breadboard breadboard_20260622.json --pcb pcb_20260625.json
"""

import json
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.font_manager as fm
from pathlib import Path

# 한글 폰트
_korean_fonts = [
    '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
]
for fp in _korean_fonts:
    if Path(fp).exists():
        fm.fontManager.addfont(fp)
        rcParams['font.family'] = fm.FontProperties(fname=fp).get_name()
        break
rcParams['axes.unicode_minus'] = False
rcParams['figure.dpi'] = 150

NOISE_LOG_DIR = Path(__file__).parent / "noise_logs"
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def find_latest(label: str) -> dict:
    """특정 라벨의 가장 최근 측정 파일을 찾아 로드."""
    pattern = str(NOISE_LOG_DIR / f"{label}_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    latest = files[-1]
    print(f"📂 {label} 데이터: {Path(latest).name}")
    with open(latest, 'r', encoding='utf-8') as f:
        return json.load(f)


def compare_and_plot(bb_data: dict, pcb_data: dict):
    """두 측정 데이터를 비교하여 4개 차트 생성."""
    
    bb = bb_data['sensors']['s3']  # 검지 센서 (핵심)
    pcb = pcb_data['sensors']['s3']
    
    # ═══ Chart 1: 가속도/자이로 노이즈 바 차트 ═══
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    labels = ['X축', 'Y축', 'Z축', '종합(Norm)']
    x = np.arange(len(labels))
    w = 0.32
    
    # 가속도
    bb_acc = bb['accel']['std_xyz'] + [bb['accel']['std_norm']]
    pcb_acc = pcb['accel']['std_xyz'] + [pcb['accel']['std_norm']]
    
    ax1.bar(x - w/2, bb_acc, w, label='Breadboard', color='#e74c3c', alpha=0.9)
    ax1.bar(x + w/2, pcb_acc, w, label='Custom PCB', color='#2ecc71', alpha=0.9)
    ax1.set_xlabel('축 (Axis)')
    ax1.set_ylabel('노이즈 RMS (m/s²)')
    ax1.set_title('가속도계 노이즈 비교 (실측)', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    for i in range(len(x)):
        if bb_acc[i] > 0:
            reduction = (1 - pcb_acc[i]/bb_acc[i]) * 100
            ax1.annotate(f'{reduction:+.0f}%', xy=(x[i]+w/2, pcb_acc[i]),
                        xytext=(0, 8), textcoords='offset points',
                        ha='center', fontsize=9, fontweight='bold', color='#27ae60')
    
    # 자이로
    bb_gyro = bb['gyro']['std_xyz_dps'] + [bb['gyro']['std_norm_dps']]
    pcb_gyro = pcb['gyro']['std_xyz_dps'] + [pcb['gyro']['std_norm_dps']]
    
    ax2.bar(x - w/2, bb_gyro, w, label='Breadboard', color='#e74c3c', alpha=0.9)
    ax2.bar(x + w/2, pcb_gyro, w, label='Custom PCB', color='#2ecc71', alpha=0.9)
    ax2.set_xlabel('축 (Axis)')
    ax2.set_ylabel('노이즈 RMS (°/s)')
    ax2.set_title('자이로스코프 노이즈 비교 (실측)', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    for i in range(len(x)):
        if bb_gyro[i] > 0:
            reduction = (1 - pcb_gyro[i]/bb_gyro[i]) * 100
            ax2.annotate(f'{reduction:+.0f}%', xy=(x[i]+w/2, pcb_gyro[i]),
                        xytext=(0, 8), textcoords='offset points',
                        ha='center', fontsize=9, fontweight='bold', color='#27ae60')
    
    fig.suptitle(f'Breadboard vs Custom PCB: 센서 노이즈 실측 비교\n'
                 f'(BB: {bb_data["timestamp"][:10]} / PCB: {pcb_data["timestamp"][:10]})',
                 fontsize=13, fontweight='bold', y=1.05)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "real_1_noise_comparison.png", bbox_inches='tight')
    print(f"✅ {OUTPUT_DIR / 'real_1_noise_comparison.png'}")
    
    # ═══ Chart 2: 시계열 파형 비교 ═══
    if 'waveform_s3_az' in bb_data['sensors'] and 'waveform_s3_az' in pcb_data['sensors']:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True, sharey=True)
        
        bb_wave = bb_data['sensors']['waveform_s3_az']
        pcb_wave = pcb_data['sensors']['waveform_s3_az']
        
        t_bb = np.arange(len(bb_wave)) / 85.0 * 1000  # ms
        t_pcb = np.arange(len(pcb_wave)) / 85.0 * 1000
        
        ax1.plot(t_bb, bb_wave, color='#e74c3c', linewidth=0.8)
        ax1.axhline(y=np.mean(bb_wave), color='gray', linestyle='--', linewidth=0.8)
        ax1.set_ylabel('가속도 Z (m/s²)')
        ax1.set_title('Breadboard — 정지 상태 가속도 Z축', fontweight='bold', color='#e74c3c')
        ax1.grid(alpha=0.3)
        bb_std = np.std(bb_wave)
        ax1.text(0.98, 0.05, f'σ = {bb_std:.4f} m/s²', transform=ax1.transAxes,
                ha='right', fontsize=10, fontweight='bold', color='#e74c3c',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax2.plot(t_pcb, pcb_wave, color='#2ecc71', linewidth=0.8)
        ax2.axhline(y=np.mean(pcb_wave), color='gray', linestyle='--', linewidth=0.8)
        ax2.set_xlabel('시간 (ms)')
        ax2.set_ylabel('가속도 Z (m/s²)')
        ax2.set_title('Custom PCB — 정지 상태 가속도 Z축', fontweight='bold', color='#2ecc71')
        ax2.grid(alpha=0.3)
        pcb_std = np.std(pcb_wave)
        reduction = (1 - pcb_std/bb_std) * 100 if bb_std > 0 else 0
        ax2.text(0.98, 0.05, f'σ = {pcb_std:.4f} m/s²  ({reduction:+.0f}%)', 
                transform=ax2.transAxes, ha='right', fontsize=10, fontweight='bold', 
                color='#2ecc71', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        fig.suptitle('정지 상태 가속도 센서 파형 비교 (실측, 85Hz)', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        fig.savefig(OUTPUT_DIR / "real_2_waveform_comparison.png", bbox_inches='tight')
        print(f"✅ {OUTPUT_DIR / 'real_2_waveform_comparison.png'}")
    
    # ═══ Chart 3: 종합 요약 테이블 차트 ═══
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    
    rows = [
        ['가속도 X σ', f'{bb_acc[0]:.4f} m/s²', f'{pcb_acc[0]:.4f} m/s²', f'{(1-pcb_acc[0]/bb_acc[0])*100:+.1f}%' if bb_acc[0] > 0 else 'N/A'],
        ['가속도 Y σ', f'{bb_acc[1]:.4f} m/s²', f'{pcb_acc[1]:.4f} m/s²', f'{(1-pcb_acc[1]/bb_acc[1])*100:+.1f}%' if bb_acc[1] > 0 else 'N/A'],
        ['가속도 Z σ', f'{bb_acc[2]:.4f} m/s²', f'{pcb_acc[2]:.4f} m/s²', f'{(1-pcb_acc[2]/bb_acc[2])*100:+.1f}%' if bb_acc[2] > 0 else 'N/A'],
        ['자이로 X σ', f'{bb_gyro[0]:.4f} °/s', f'{pcb_gyro[0]:.4f} °/s', f'{(1-pcb_gyro[0]/bb_gyro[0])*100:+.1f}%' if bb_gyro[0] > 0 else 'N/A'],
        ['자이로 Y σ', f'{bb_gyro[1]:.4f} °/s', f'{pcb_gyro[1]:.4f} °/s', f'{(1-pcb_gyro[1]/bb_gyro[1])*100:+.1f}%' if bb_gyro[1] > 0 else 'N/A'],
        ['자이로 Z σ', f'{bb_gyro[2]:.4f} °/s', f'{pcb_gyro[2]:.4f} °/s', f'{(1-pcb_gyro[2]/bb_gyro[2])*100:+.1f}%' if bb_gyro[2] > 0 else 'N/A'],
        ['종합 가속도 σ', f'{bb_acc[3]:.4f} m/s²', f'{pcb_acc[3]:.4f} m/s²', f'{(1-pcb_acc[3]/bb_acc[3])*100:+.1f}%' if bb_acc[3] > 0 else 'N/A'],
        ['종합 자이로 σ', f'{bb_gyro[3]:.4f} °/s', f'{pcb_gyro[3]:.4f} °/s', f'{(1-pcb_gyro[3]/bb_gyro[3])*100:+.1f}%' if bb_gyro[3] > 0 else 'N/A'],
    ]
    
    table = ax.table(
        cellText=rows,
        colLabels=['항목', 'Breadboard', 'Custom PCB', '변화율'],
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # 헤더 색상
    for j in range(4):
        table[0, j].set_facecolor('#34495e')
        table[0, j].set_text_props(color='white', fontweight='bold')
    
    # 변화율 열 색상
    for i in range(1, len(rows)+1):
        val = rows[i-1][3]
        if val != 'N/A' and '-' in val:
            table[i, 3].set_text_props(color='#27ae60', fontweight='bold')
    
    ax.set_title('Breadboard vs PCB 노이즈 실측 비교 요약 (S3/검지)', 
                fontsize=13, fontweight='bold', pad=20)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "real_3_summary_table.png", bbox_inches='tight', pad_inches=0.3)
    print(f"✅ {OUTPUT_DIR / 'real_3_summary_table.png'}")
    
    print(f"\n📊 실측 비교 차트 생성 완료!")


def main():
    parser = argparse.ArgumentParser(description='노이즈 실측 비교 차트 생성')
    parser.add_argument('--breadboard', type=str, help='빵판 측정 JSON 파일명')
    parser.add_argument('--pcb', type=str, help='PCB 측정 JSON 파일명')
    args = parser.parse_args()
    
    if args.breadboard and args.pcb:
        with open(NOISE_LOG_DIR / args.breadboard, 'r') as f:
            bb = json.load(f)
        with open(NOISE_LOG_DIR / args.pcb, 'r') as f:
            pcb = json.load(f)
    else:
        bb = find_latest("breadboard")
        pcb = find_latest("pcb")
    
    if not bb:
        print("❌ breadboard 측정 데이터가 없습니다!")
        print("   → 빵판 상태에서 서버를 실행하고 캘리브레이션을 완료하세요.")
        print(f"   → 측정 데이터가 {NOISE_LOG_DIR}/ 에 자동 저장됩니다.")
        return
    
    if not pcb:
        print(f"✅ breadboard 데이터 확인됨: {bb['timestamp']}")
        print(f"❌ pcb 측정 데이터가 없습니다!")
        print(f"   → PCB로 교체 후:")
        print(f"   1. python tools/noise_measure.py 에서 set_hardware_label('pcb') 실행")
        print(f"   2. 서버 재시작 → 캘리브레이션 완료")
        print(f"   3. python tools/noise_compare.py 실행!")
        return
    
    compare_and_plot(bb, pcb)


if __name__ == "__main__":
    main()
