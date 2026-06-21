"""replay_eskf.py — ESKF E5 (ZUPT 후 P 미축소) 오프라인 결정론적 A/B 재생기

목적 (Phase 0 관측성 게이트):
  같은 입력 스트림에 두 변종을 통과시켜, E5 수정이 s3_eskf.q(= bio_kinematics로
  흘러가는 유일한 ESKF 출력)에 유의미한 차이를 만드는지 숫자로 판정한다.

  변종 A = 현행 (프로덕션 ESKF.update_zupt 그대로 호출 → 거동 100% 동일)
  변종 B = 수정 (Joseph form 공분산 축소 P=(I-KH)P(I-KH)ᵀ+KRKᵀ 추가)

[hands-off 보존] 변종 B는 ESKF를 서브클래싱한 ReplayESKF에만 존재한다.
프로덕션 airwriting_imu/core/eskf_filter.py는 수정하지 않는다.

사용법:
  py -3 replay_eskf.py rawlog_s3.jsonl          # main.py --rawlog 로 녹화한 실데이터
  py -3 replay_eskf.py --synthetic              # 하드웨어 없이 하네스 데모/검증
  py -3 replay_eskf.py rawlog_s3.jsonl --csv out.csv   # 프레임별 시계열 덤프
"""
import sys
import json
import numpy as np
from scipy.spatial.transform import Rotation

from airwriting_imu.core.eskf_filter import ESKF


class ReplayESKF(ESKF):
    """ESKF + 선택적 공분산 축소(reduce_cov). reduce_cov=False면 프로덕션과 동일."""

    def __init__(self, dt=0.01, reduce_cov=False):
        super().__init__(dt=dt)
        self.reduce_cov = reduce_cov

    def _joseph(self, K, H, R):
        """Joseph form 공분산 축소: P = (I-KH)P(I-KH)ᵀ + KRKᵀ (대칭/양정치 보존)."""
        I = np.eye(15)
        A = I - K @ H
        self.P = A @ self.P @ A.T + K @ R @ K.T

    def update_zupt(self, current_gyro=None):
        # 변종 A: 프로덕션 메서드를 그대로 호출 → 비트 동일성 보장
        if not self.reduce_cov:
            return super().update_zupt(current_gyro)

        # 변종 B: 원본 로직 + 각 측정 갱신 후 Joseph 공분산 축소
        if current_gyro is not None:
            H_zaru = np.zeros((3, 15))
            H_zaru[0:3, 12:15] = np.eye(3)
            z_g = current_gyro - self.w_b
            K_g = self.P @ H_zaru.T @ np.linalg.inv(H_zaru @ self.P @ H_zaru.T + self.R_zaru)
            K_g[6:9, :] = 0  # 자세 보호 마스킹 유지
            self._inject_error(K_g @ z_g)
            self._joseph(K_g, H_zaru, self.R_zaru)   # ← E5 수정 핵심

        H_zupt = np.zeros((3, 15))
        H_zupt[0:3, 3:6] = np.eye(3)
        z_v = np.zeros(3) - self.v
        K_v = self.P @ H_zupt.T @ np.linalg.inv(H_zupt @ self.P @ H_zupt.T + self.R_zupt)
        K_v[6:9, :] = 0  # 자세 보호 마스킹 유지
        self._inject_error(K_v @ z_v)
        self._joseph(K_v, H_zupt, self.R_zupt)       # ← E5 수정 핵심


def _reset_from_header(eskf, calib):
    q = calib.get("q_align_s3")
    ba = calib.get("ba_s3")
    bg = calib.get("bg_s3")
    m = calib.get("m_ref_s3")
    eskf.reset(
        initial_q=np.array(q, dtype=np.float64) if q else None,
        initial_ba=np.array(ba, dtype=np.float64) if ba else None,
        initial_bg=np.array(bg, dtype=np.float64) if bg else None,
        initial_mag=np.array(m, dtype=np.float64) if m else None,
    )


def load_jsonl(path):
    header, rows = None, []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("type") == "header":
                header = obj["calib"]
            else:
                rows.append(obj)
    if header is None:
        header = {}  # 캘리브 전 종료된 로그 — identity로 시작
    return header, rows


def make_synthetic(n_seg=8, seg_frames=200, dt=0.0117, seed=0):
    """하드웨어 없이 하네스를 검증하기 위한 합성 스트림.

    정지(ZUPT 발동) ↔ 필기(움직임) 구간을 번갈아 생성. 버튼은 필기 구간에서 1.
    heavy=1(모든 프레임을 last-packet으로 간주), zupt=-1(replay가 detect_zupt로 결정).
    """
    rng = np.random.default_rng(seed)
    rows = []
    ts = 0
    for s in range(n_seg):
        writing = (s % 2 == 1)
        for _ in range(seg_frames):
            ts += int(dt * 1000)
            if writing:
                # 손목 회전 위주의 필기 동작 + 약한 선형 가속
                fg = rng.normal(0, 0.6, 3)
                fa = np.array([0.0, 0.0, 9.81]) + rng.normal(0, 1.2, 3)
                btn = 1
            else:
                # 정지: 중력만 + 미세 노이즈 → detect_zupt True 기대
                fg = rng.normal(0, 0.01, 3)
                fa = np.array([0.0, 0.0, 9.81]) + rng.normal(0, 0.02, 3)
                btn = 0
            rows.append({
                "ts": ts, "dt": dt,
                "fa": fa.tolist(), "fg": fg.tolist(),
                "fm": None, "btn": btn,
                "writing": 1 if writing else 0, "heavy": 1, "zupt": -1,
            })
    return {}, rows


def _q_angle_deg(qa, qb):
    """두 scipy Rotation 사이의 각도 차(도)."""
    rel = qa.inv() * qb
    return float(np.degrees(np.linalg.norm(rel.as_rotvec())))


def run_ab(header, rows, csv_path=None):
    a = ReplayESKF(reduce_cov=False)   # 현행
    b = ReplayESKF(reduce_cov=True)    # 수정
    _reset_from_header(a, header)
    _reset_from_header(b, header)

    q_div, trace_a, trace_b = [], [], []
    nan_a = nan_b = 0
    zupt_fires = 0

    csv_f = open(csv_path, "w", encoding="utf-8") if csv_path else None
    if csv_f:
        csv_f.write("i,q_div_deg,traceP_vv_A,traceP_vv_B,vnorm_A,vnorm_B,wb_norm_A,wb_norm_B,zupt\n")

    for i, r in enumerate(rows):
        fa = np.array(r["fa"], dtype=np.float64)
        fg = np.array(r["fg"], dtype=np.float64)
        dt = float(r["dt"])
        writing = bool(r["writing"])
        heavy = bool(r["heavy"])

        a.predict(fa, fg, dt)
        b.predict(fa, fg, dt)

        z = 0
        if heavy:
            alpha = 0.0003 if writing else 0.002
            a.update_gravity_mahony(fa, alpha=alpha)
            b.update_gravity_mahony(fa, alpha=alpha)
            if not writing:
                # detect_zupt는 P와 무관(분산 기반) → A로 한 번 판정해 양쪽에 동일 적용
                logged = int(r.get("zupt", -1))
                z = a.detect_zupt() if logged < 0 else logged
                if z:
                    a.update_zupt(fg)
                    b.update_zupt(fg)
                    zupt_fires += 1

        ta = float(np.trace(a.P[3:6, 3:6]))
        tb = float(np.trace(b.P[3:6, 3:6]))
        if not np.isfinite(ta) or not np.all(np.isfinite(a.q.as_quat())):
            nan_a += 1
        if not np.isfinite(tb) or not np.all(np.isfinite(b.q.as_quat())):
            nan_b += 1

        d = _q_angle_deg(a.q, b.q)
        q_div.append(d)
        trace_a.append(ta)
        trace_b.append(tb)

        if csv_f:
            csv_f.write(f"{i},{d:.6f},{ta:.4f},{tb:.4f},"
                        f"{np.linalg.norm(a.v):.5f},{np.linalg.norm(b.v):.5f},"
                        f"{np.linalg.norm(a.w_b):.6f},{np.linalg.norm(b.w_b):.6f},{z}\n")

    if csv_f:
        csv_f.close()

    q_div = np.array(q_div)
    return {
        "n": len(rows),
        "zupt_fires": zupt_fires,
        "q_div_max": float(q_div.max()) if len(q_div) else 0.0,
        "q_div_rms": float(np.sqrt(np.mean(q_div ** 2))) if len(q_div) else 0.0,
        "traceP_A_final": trace_a[-1] if trace_a else 0.0,
        "traceP_A_max": max(trace_a) if trace_a else 0.0,
        "traceP_B_final": trace_b[-1] if trace_b else 0.0,
        "traceP_B_max": max(trace_b) if trace_b else 0.0,
        "nan_A": nan_a,
        "nan_B": nan_b,
        "csv": csv_path,
    }


def main(argv):
    csv_path = None
    if "--csv" in argv:
        ci = argv.index("--csv")
        csv_path = argv[ci + 1] if ci + 1 < len(argv) else "replay_out.csv"
        argv = argv[:ci] + argv[ci + 2:]

    if "--synthetic" in argv:
        print("[replay] 합성 스트림 생성 (하드웨어 없이 하네스 검증)")
        header, rows = make_synthetic()
    else:
        paths = [a for a in argv[1:] if not a.startswith("--")]
        if not paths:
            print(__doc__)
            print("ERROR: rawlog 경로 또는 --synthetic 필요")
            return 2
        print(f"[replay] 로드: {paths[0]}")
        header, rows = load_jsonl(paths[0])

    if not rows:
        print("ERROR: 프레임 0개")
        return 2

    res = run_ab(header, rows, csv_path)

    print("=" * 60)
    print("  ESKF E5 — 변종 A(현행) vs B(P 축소) 오프라인 재생 결과")
    print("=" * 60)
    print(f"  프레임 수            : {res['n']}")
    print(f"  ZUPT 발동 횟수       : {res['zupt_fires']}")
    print(f"  trace(P_vv) A 최종/최대: {res['traceP_A_final']:.1f} / {res['traceP_A_max']:.1f}")
    print(f"  trace(P_vv) B 최종/최대: {res['traceP_B_final']:.3f} / {res['traceP_B_max']:.3f}")
    print(f"  q 발산 (A vs B)  max  : {res['q_div_max']:.4f}°")
    print(f"  q 발산 (A vs B)  RMS  : {res['q_div_rms']:.4f}°")
    print(f"  NaN/Inf  A / B       : {res['nan_A']} / {res['nan_B']}")
    if res["csv"]:
        print(f"  프레임별 CSV         : {res['csv']}")
    print("-" * 60)

    # 판정 (Phase 0 관측성 게이트)
    GATE_DEG = 0.5
    if res["nan_A"] or res["nan_B"]:
        print(f"  ⚠️ 판정: NaN 발생 → 수치 불안정. 변종/입력 점검 필요.")
    elif res["q_div_max"] < GATE_DEG:
        print(f"  ✅ 판정: q 발산 max {res['q_div_max']:.4f}° < {GATE_DEG}° "
              f"→ E5는 s3_eskf.q에 사실상 INERT.")
        print(f"     trace(P_vv)는 A에서 {res['traceP_A_max']:.0f}까지 폭주하지만 "
              f"B에서 {res['traceP_B_max']:.2f}로 유계 — 그러나 q/하류 출력에는 영향 미미.")
        print(f"     → 권고: 수정하지 말고 현 상태 유지(회귀 위험만 추가). 근거 데이터 확보됨.")
    else:
        print(f"  ❗ 판정: q 발산 max {res['q_div_max']:.4f}° ≥ {GATE_DEG}° "
              f"→ E5가 q를 흔든다. Phase 1(정량) / Phase 2(하드웨어 A/B)로 진행.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
