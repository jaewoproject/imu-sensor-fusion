"""
드리프트 비교 실험: Naive 속도 적분 vs ESKF 센서 퓨전
========================================================================
현재 ESKF 아키텍처를 채택한 근거를 정량적으로 제시하기 위한 비교 실험.

시나리오:
  (a) 정지 상태 드리프트 — 센서를 가만히 놔뒀을 때 위치가 얼마나 이탈하는가
  (b) 필기 후 속도 잔류 — 필기 동작 후 속도가 0으로 수렴하는가
  (c) 반복 필기 누적 드리프트 — 같은 글자를 반복할 때 원점이 밀리는 정도
"""

import json
import glob
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


# ═══════════════════════════════════════
# Naive Integration
# ═══════════════════════════════════════
def naive_process(accel_seq, gyro_seq, dt_seq):
    n_cal = min(10, len(accel_seq))
    g_est = np.mean(accel_seq[:n_cal], axis=0)
    
    positions = [np.zeros(3)]
    velocities = [np.zeros(3)]
    v = np.zeros(3)
    p = np.zeros(3)
    
    for acc, dt in zip(accel_seq, dt_seq):
        a_lin = acc - g_est
        v = v + a_lin * dt
        p = p + v * dt + 0.5 * a_lin * dt**2
        positions.append(p.copy())
        velocities.append(v.copy())
    
    return np.array(positions), np.array(velocities)


# ═══════════════════════════════════════
# ESKF Integration
# ═══════════════════════════════════════
class MiniESKF:
    def __init__(self):
        self.g = np.array([0, 0, 9.81])
        self.p = np.zeros(3)
        self.v = np.zeros(3)
        self.q = Rotation.from_quat([0, 0, 0, 1])
        self.a_b = np.zeros(3)
        self.w_b = np.zeros(3)
        self.window_size = 15
        self.a_win, self.g_win = [], []
    
    def detect_zupt(self):
        if len(self.a_win) < self.window_size: return False
        a_var = np.var(self.a_win, axis=0)
        g_var = np.var(self.g_win, axis=0)
        a_mean = np.mean(self.a_win, axis=0)
        return (np.sum(a_var) < 0.15 and np.sum(g_var) < 0.15 
                and abs(np.linalg.norm(a_mean) - 9.81) < 0.5)
    
    def step(self, accel, gyro, dt):
        gyro_t = gyro - self.w_b
        angle = np.linalg.norm(gyro_t) * dt
        if angle > 1e-8:
            self.q = self.q * Rotation.from_rotvec(gyro_t / np.linalg.norm(gyro_t) * angle)
        
        R = self.q.as_matrix()
        accel_world = R @ (accel - self.a_b) - self.g
        self.p += self.v * dt + 0.5 * accel_world * dt**2
        self.v += accel_world * dt
        
        self.a_win.append(accel.copy())
        self.g_win.append(gyro.copy())
        if len(self.a_win) > self.window_size:
            self.a_win.pop(0)
            self.g_win.pop(0)
        
        if self.detect_zupt():
            self.v *= 0.90
        else:
            self.v *= 0.995
        
        v_n = np.linalg.norm(self.v)
        if v_n > 2.0: self.v *= 2.0 / v_n
        return self.p.copy(), self.v.copy()


def eskf_process(accel_seq, gyro_seq, dt_seq):
    eskf = MiniESKF()
    positions = [np.zeros(3)]
    velocities = [np.zeros(3)]
    for a, g, dt in zip(accel_seq, gyro_seq, dt_seq):
        p, v = eskf.step(a, g, dt)
        positions.append(p.copy())
        velocities.append(v.copy())
    return np.array(positions), np.array(velocities)


# ═══════════════════════════════════════
# 시나리오 생성
# ═══════════════════════════════════════
def gen_stationary_data(seconds=10, hz=85):
    """정지 상태: 중력만 + 센서 노이즈 + 미세 바이어스"""
    n = int(seconds * hz)
    np.random.seed(42)
    
    # 실제 센서처럼 중력 + 노이즈 + 바이어스 드리프트
    bias_acc = np.array([0.03, -0.02, 0.01])  # 초기 가속도 바이어스
    bias_gyr = np.array([0.005, -0.003, 0.002])  # 초기 자이로 바이어스
    
    accel = np.tile([0, 0, 9.81], (n, 1)) + bias_acc
    accel += np.random.normal(0, 0.05, (n, 3))  # 가속도 센서 노이즈
    
    # 바이어스 랜덤 워크 (시간에 따라 서서히 변동)
    for i in range(1, n):
        bias_acc += np.random.normal(0, 0.0002, 3)
        accel[i] += bias_acc
    
    gyro = np.zeros((n, 3)) + bias_gyr
    gyro += np.random.normal(0, 0.003, (n, 3))
    
    dts = np.full(n, 1.0/hz)
    return accel, gyro, dts


def load_writing_data(dataset_dir, label='D'):
    """실제 필기 데이터 로드"""
    files = sorted(glob.glob(os.path.join(dataset_dir, f'{label}_*.json')))[:5]
    all_acc, all_gyr, all_dt = [], [], []
    
    for f in files:
        data = json.load(open(f))
        for stroke in data['strokes']:
            for frame in stroke:
                all_acc.append([frame['ax'], frame['ay'], frame['az']])
                all_gyr.append([frame['gx'], frame['gy'], frame['gz']])
                dt = frame['dt']
                all_dt.append(dt if 0 < dt < 0.1 else 0.012)
    
    return np.array(all_acc), np.array(all_gyr), np.array(all_dt)


def gen_write_then_stop(accel_w, gyro_w, dts_w, stop_seconds=3, hz=85):
    """필기 데이터 뒤에 정지 구간을 붙인 시퀀스"""
    n_stop = int(stop_seconds * hz)
    
    # 정지 구간: 마지막 프레임의 가속도를 기반으로 중력 방향 유지
    last_acc = accel_w[-1]
    stop_acc = np.tile(last_acc, (n_stop, 1)) + np.random.normal(0, 0.02, (n_stop, 3))
    stop_gyr = np.random.normal(0, 0.003, (n_stop, 3))
    stop_dt = np.full(n_stop, 1.0/hz)
    
    acc = np.vstack([accel_w, stop_acc])
    gyr = np.vstack([gyro_w, stop_gyr])
    dt = np.concatenate([dts_w, stop_dt])
    
    return acc, gyr, dt, len(accel_w)


def run_comparison():
    dataset_dir = os.path.join(os.path.dirname(__file__), 'dataset')
    
    # ─── 시나리오 A: 정지 상태 드리프트 (5초, 15초, 30초) ───
    print("[시나리오 A] 정지 상태 드리프트...")
    stat_results = {}
    for dur in [5, 15, 30]:
        acc, gyr, dts = gen_stationary_data(dur)
        n_pos, n_vel = naive_process(acc, gyr, dts)
        e_pos, e_vel = eskf_process(acc, gyr, dts)
        stat_results[dur] = {
            'naive_pos': n_pos, 'eskf_pos': e_pos,
            'naive_vel': n_vel, 'eskf_vel': e_vel,
        }
        nd = np.sqrt(n_pos[-1, 0]**2 + n_pos[-1, 1]**2 + n_pos[-1, 2]**2)
        ed = np.sqrt(e_pos[-1, 0]**2 + e_pos[-1, 1]**2 + e_pos[-1, 2]**2)
        print(f"  [{dur:2d}s] Naive: {nd:.4f}m | ESKF: {ed:.4f}m")
    
    # ─── 시나리오 B: 필기 후 속도 수렴 ───
    print("\n[시나리오 B] 필기 후 속도 수렴...")
    acc_w, gyr_w, dt_w = load_writing_data(dataset_dir, 'D')
    acc_ws, gyr_ws, dt_ws, split_idx = gen_write_then_stop(acc_w, gyr_w, dt_w, stop_seconds=3)
    
    n_pos_b, n_vel_b = naive_process(acc_ws, gyr_ws, dt_ws)
    e_pos_b, e_vel_b = eskf_process(acc_ws, gyr_ws, dt_ws)
    
    n_speed_b = np.sqrt(np.sum(n_vel_b**2, axis=1))
    e_speed_b = np.sqrt(np.sum(e_vel_b**2, axis=1))
    
    print(f"  필기 종료 시점 속도 - Naive: {n_speed_b[split_idx]:.4f}m/s | ESKF: {e_speed_b[split_idx]:.4f}m/s")
    print(f"  3초 정지 후 속도   - Naive: {n_speed_b[-1]:.4f}m/s | ESKF: {e_speed_b[-1]:.4f}m/s")
    
    # ─── 시나리오 C: 반복 필기 누적 드리프트 (5회 반복) ───
    print("\n[시나리오 C] 반복 필기 누적 드리프트...")
    n_repeats = 5
    # 같은 글자를 5번 반복
    acc_rep = np.tile(acc_w, (n_repeats, 1))
    gyr_rep = np.tile(gyr_w, (n_repeats, 1))
    dt_rep = np.tile(dt_w, n_repeats)
    
    n_pos_c, n_vel_c = naive_process(acc_rep, gyr_rep, dt_rep)
    e_pos_c, e_vel_c = eskf_process(acc_rep, gyr_rep, dt_rep)
    
    # 각 반복의 끝에서의 드리프트
    stride = len(acc_w)
    for rep in range(n_repeats):
        end_idx = (rep + 1) * stride
        nd = np.sqrt(np.sum(n_pos_c[end_idx]**2))
        ed = np.sqrt(np.sum(e_pos_c[end_idx]**2))
        print(f"  반복 {rep+1}회 후 - Naive: {nd:.4f}m | ESKF: {ed:.4f}m")
    
    # ═══════════════════════════════════════
    # 시각화 (3행 2열)
    # ═══════════════════════════════════════
    fig, axes = plt.subplots(3, 2, figsize=(16, 18), facecolor='white')
    fig.suptitle("Naive 이중적분 vs ESKF 센서 퓨전 — 드리프트 비교 실험\n"
                 "(ESKF 아키텍처 채택 근거)",
                 fontsize=15, fontweight='bold', color='#1a1a2e', y=0.99)
    
    # ── (a-1) 정지 상태: 위치 드리프트 시계열 ──
    ax = axes[0, 0]
    for dur in [5, 15, 30]:
        r = stat_results[dur]
        t = np.arange(len(r['naive_pos'])) / 85.0
        n_drift = np.sqrt(np.sum(r['naive_pos']**2, axis=1))
        e_drift = np.sqrt(np.sum(r['eskf_pos']**2, axis=1))
        ax.plot(t, n_drift, '-', lw=2, alpha=0.8, label=f'Naive {dur}s (final: {n_drift[-1]:.4f}m)')
        ax.plot(t, e_drift, '--', lw=2, alpha=0.8, label=f'ESKF {dur}s (final: {e_drift[-1]:.4f}m)')
    
    ax.set_title('(a) 정지 상태 위치 드리프트\n(센서를 고정한 채 방치)', fontsize=11, fontweight='bold', color='#2c3e50')
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Position Drift (m)', fontsize=10)
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # ── (a-2) 정지 상태: 속도 발산 ──
    ax = axes[0, 1]
    for dur in [5, 15, 30]:
        r = stat_results[dur]
        t = np.arange(len(r['naive_vel'])) / 85.0
        n_speed = np.sqrt(np.sum(r['naive_vel']**2, axis=1))
        e_speed = np.sqrt(np.sum(r['eskf_vel']**2, axis=1))
        ax.plot(t, n_speed, '-', lw=2, alpha=0.8, label=f'Naive {dur}s')
        ax.plot(t, e_speed, '--', lw=2, alpha=0.8, label=f'ESKF {dur}s')
    
    ax.axhline(y=0.0, color='#27ae60', linestyle=':', lw=1, alpha=0.5)
    ax.set_title('(a) 정지 상태 속도 발산\n(이상적으로 0이어야 함)', fontsize=11, fontweight='bold', color='#2c3e50')
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Speed (m/s)', fontsize=10)
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # ── (b-1) 필기 후 속도 수렴 ──
    ax = axes[1, 0]
    t_b = np.arange(len(n_speed_b)) / 85.0
    t_split = split_idx / 85.0
    
    ax.plot(t_b, n_speed_b, '-', color='#e74c3c', lw=2, label='Naive', alpha=0.8)
    ax.plot(t_b, e_speed_b, '-', color='#2980b9', lw=2, label='ESKF', alpha=0.8)
    ax.axvline(x=t_split, color='#e67e22', linestyle='--', lw=1.5, alpha=0.7, label='필기 종료')
    ax.axhline(y=0.0, color='#27ae60', linestyle=':', lw=1, alpha=0.5)
    
    # 주석
    ax.annotate(f'Naive: {n_speed_b[-1]:.2f} m/s\n(속도 잔류)', 
                xy=(t_b[-1], n_speed_b[-1]), fontsize=9, color='#e74c3c', fontweight='bold',
                ha='right', va='bottom')
    ax.annotate(f'ESKF: {e_speed_b[-1]:.4f} m/s\n(0 수렴)', 
                xy=(t_b[-1], e_speed_b[-1]+0.05), fontsize=9, color='#2980b9', fontweight='bold',
                ha='right', va='bottom')
    
    ax.set_title('(b) 필기 후 속도 수렴 비교\n(필기 종료 후 3초간 정지)', fontsize=11, fontweight='bold', color='#2c3e50')
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Speed (m/s)', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # ── (b-2) 필기 후 위치 궤적 ──
    ax = axes[1, 1]
    ax.plot(n_pos_b[:, 0], n_pos_b[:, 1], '-', color='#e74c3c', lw=1, alpha=0.6, label='Naive')
    ax.plot(e_pos_b[:, 0], e_pos_b[:, 1], '-', color='#2980b9', lw=1.5, alpha=0.8, label='ESKF')
    ax.plot(0, 0, 'o', color='#27ae60', markersize=10, zorder=5, label='Origin')
    ax.plot(n_pos_b[-1, 0], n_pos_b[-1, 1], 'x', color='#e74c3c', markersize=12, markeredgewidth=3, zorder=5)
    ax.plot(e_pos_b[-1, 0], e_pos_b[-1, 1], 'x', color='#2980b9', markersize=12, markeredgewidth=3, zorder=5)
    
    ax.set_title('(b) 필기+정지 궤적 (XY 평면)\n(정지 후에도 Naive는 계속 이동)', fontsize=11, fontweight='bold', color='#2c3e50')
    ax.set_xlabel('X (m)', fontsize=10)
    ax.set_ylabel('Y (m)', fontsize=10)
    ax.legend(fontsize=9)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # ── (c-1) 반복 필기 누적 드리프트 ──
    ax = axes[2, 0]
    t_c = np.arange(len(n_pos_c)) / 85.0
    n_drift_c = np.sqrt(np.sum(n_pos_c**2, axis=1))
    e_drift_c = np.sqrt(np.sum(e_pos_c**2, axis=1))
    
    ax.plot(t_c, n_drift_c, '-', color='#e74c3c', lw=2, label='Naive', alpha=0.8)
    ax.plot(t_c, e_drift_c, '-', color='#2980b9', lw=2, label='ESKF', alpha=0.8)
    
    # 반복 구간 표시
    for rep in range(n_repeats):
        t_mark = (rep + 1) * stride / 85.0
        ax.axvline(x=t_mark, color='#95a5a6', linestyle=':', lw=0.8, alpha=0.5)
        if rep < n_repeats:
            ax.text(t_mark, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1, 
                    f'#{rep+1}', fontsize=8, ha='center', color='#7f8c8d')
    
    ax.set_title('(c) 반복 필기 누적 드리프트 (D 글자 x5)\n(Naive: 2차 함수적 발산)', fontsize=11, fontweight='bold', color='#2c3e50')
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Cumulative Drift (m)', fontsize=10)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # ── (c-2) 종합 통계 바 차트 ──
    ax = axes[2, 1]
    
    categories = ['정지 5s\n위치 드리프트', '정지 30s\n위치 드리프트',
                  '필기 후\n잔류 속도', '5회 반복 후\n누적 드리프트']
    
    n5 = np.sqrt(np.sum(stat_results[5]['naive_pos'][-1]**2))
    e5 = np.sqrt(np.sum(stat_results[5]['eskf_pos'][-1]**2))
    n30 = np.sqrt(np.sum(stat_results[30]['naive_pos'][-1]**2))
    e30 = np.sqrt(np.sum(stat_results[30]['eskf_pos'][-1]**2))
    nv_res = n_speed_b[-1]
    ev_res = e_speed_b[-1]
    n_rep_final = np.sqrt(np.sum(n_pos_c[-1]**2))
    e_rep_final = np.sqrt(np.sum(e_pos_c[-1]**2))
    
    naive_vals = [n5, n30, nv_res, n_rep_final]
    eskf_vals = [e5, e30, ev_res, e_rep_final]
    
    x = np.arange(len(categories))
    w = 0.35
    
    bars_n = ax.bar(x - w/2, naive_vals, w, label='Naive Integration', color='#e74c3c', alpha=0.85, edgecolor='#c0392b')
    bars_e = ax.bar(x + w/2, eskf_vals, w, label='ESKF Pipeline', color='#2980b9', alpha=0.85, edgecolor='#2471a3')
    
    for bar, val in zip(bars_n, naive_vals):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold', color='#e74c3c')
    for bar, val in zip(bars_e, eskf_vals):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold', color='#2980b9')
    
    # 개선율 표시
    for i in range(len(categories)):
        if naive_vals[i] > 0:
            improvement = (1 - eskf_vals[i] / naive_vals[i]) * 100
            color = '#27ae60' if improvement > 0 else '#e74c3c'
            symbol = 'v' if improvement > 0 else '^'
            ax.text(x[i], max(naive_vals[i], eskf_vals[i]) * 1.15,
                    f'{improvement:+.1f}%', ha='center', fontsize=10, fontweight='bold', color=color)
    
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylabel('Value (m or m/s)', fontsize=10)
    ax.set_title('(d) 종합 비교 통계', fontsize=11, fontweight='bold', color='#2c3e50')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    out_path = os.path.join(os.path.dirname(__file__), 'web', 'drift_comparison.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"\n[OK] Saved: {out_path}")
    plt.close()
    
    # 콘솔 요약
    print("\n" + "="*60)
    print("  드리프트 비교 결과 요약 (논문 인용용)")
    print("="*60)
    for i, cat in enumerate(categories):
        cat_short = cat.replace('\n', ' ')
        n_v, e_v = naive_vals[i], eskf_vals[i]
        imp = (1 - e_v / n_v) * 100 if n_v > 0 else 0
        print(f"  {cat_short:<25} Naive: {n_v:>10.4f}  ESKF: {e_v:>10.4f}  ({imp:+.1f}%)")
    print("="*60)


if __name__ == '__main__':
    run_comparison()
