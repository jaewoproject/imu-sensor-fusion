# ESKF E5 (ZUPT 후 P 미축소) — A/B 검증 계획

> 대상 이슈: `update_zupt()` / `_inject_error()`가 칼만 측정 갱신 후
> 공분산 축소 `P = (I − K H) P`를 하지 않아, 속도 공분산 P가 단조 폭주함
> (실측: trace(P_vv) 0.03 → 10,245 / 14초). [eskf_filter.py](airwriting_imu/core/eskf_filter.py#L178-L202)
>
> **이 문서는 "고친다"가 아니라 "고칠 가치가 있는지, 고치면 회귀가 없는지"를
> 코드만으로 판단할 수 없으므로 데이터로 판정**하기 위한 절차서입니다.
> hands-off 정책상 본 계획의 승인 전에는 알고리즘 거동을 바꾸지 않습니다.

---

## 0. 결론부터 (의사결정 트리)

```
Phase 0 관측성 게이트
  └─ E5가 "사용자가 보는 값"(커서/인식결과)에 유의미한 차이를 만드는가?
       ├─ NO  → 수정하지 말 것. 현 상태 유지 + 코드에 "inert" 주석만. 종료.
       └─ YES → Phase 1 (오프라인 결정론적 A/B)
                  └─ 오프라인에서 개선 + 무회귀 확인되면
                       → Phase 2 (하드웨어 A/B, 회귀 시나리오 집중)
                            └─ 합격 시에만 머지
```

**왜 Phase 0가 먼저인가:** 현재 파이프라인에서 **눈에 보이는 커서는 Madgwick
경로**(`madgwick_s3_ray` → `ray_projector` → `OneEuroFilter`)로 만들어진다
([main.py:846,865-876](main.py#L846)). E5가 사는 `s3_eskf`의 출력
(`v`, `p`, `a_b`, `w_b`)은 커서에 직접 들어가지 않는다. 유일한 간접 경로는:

```
ZARU(update_zupt의 자이로 보정) → w_b 변화 → predict의 gyro_true = gyro − w_b
   → s3_eskf.q 변화 → DifferentialKinematics/MotionSeparator (bio_kinematics)
```

즉 E5가 의미를 가지려면 (1) ZUPT/ZARU가 실제로 자주 발동하고, (2) 그 결과가
`w_b`/`q`를 통해 bio_kinematics 출력을 흔들고, (3) 그 출력이 사용자에게 보여야
한다. **이 3단계 중 하나라도 끊겨 있으면 E5는 inert이고 수정은 순손해(회귀
위험만 추가)다.** Phase 0가 이걸 판정한다.

---

## 1. 사전 준비: 결정론적 재생(replay) 하네스

물리적으로 "똑같은 손동작"을 두 번 재현하는 것은 불가능하므로, **원시 센서
스트림을 1회 녹화 → 동일 데이터에 두 변종(현행 A / 수정 B)을 오프라인으로 통과**
시켜 apples-to-apples 비교한다.

### 1-1. 원시 프레임 로거 (신규, 비침습)

현재 녹화([main.py:889-904](main.py#L889))는 `is_writing` 구간의 `out_x/out_y`와
finger accel/gyro만 저장한다. **ZUPT는 `not is_writing`일 때 발동**하므로 이
포맷으로는 부족하다. 다음을 만족하는 별도 로거가 필요하다:

- **연속 전체 스트림** 저장 (writing/non-writing 무관, 매 프레임)
- 필드: `ts, dt, finger_accel[3], finger_gyro[3], finger_mag[3], button`
  (+ 캘리브레이션 결과 `q_align/ba/bg/m_ref`를 헤더에 1회)
- 포맷: `.jsonl` (프레임당 1줄) 또는 `np.savez` (권장: float32 배열, 용량/정밀도)
- 트리거: `--rawlog <path>` CLI 플래그로 켜고, `serial_reader_thread` 또는
  `process_serial_queue` 진입부에서 frame을 그대로 덤프 (필터 통과 전 원시값)

> 구현 위치 후보: [main.py:786](main.py#L786) 직전(= s3_eskf.predict 호출 전)에
> 원시 frame 덤프. 이렇게 하면 재생이 라이브와 동일한 입력 순서를 보장.

### 1-2. 오프라인 재생기 (신규 스크립트 `replay_eskf.py`)

```
입력:  rawlog(.npz) + 캘리브레이션 헤더
처리:  ESKF 인스턴스 1개 생성 → reset(q_align, ba, bg, m_ref)
       for frame in stream:
           eskf.predict(accel, gyro, dt)
           eskf.update_gravity_mahony(accel, alpha=...)   # 라이브와 동일 조건
           if not is_writing and eskf.detect_zupt():
               eskf.update_zupt(gyro)
           # 매 프레임 관측치 기록 (아래 metrics)
출력:  프레임별 시계열 CSV/npz (variant 태그 A 또는 B)
```

라이브 분기(`is_writing` 판정, gravity_alpha 스케줄 등)를 **[main.py:786-811]과
비트 단위로 동일하게** 복제하는 것이 핵심. 분기 로직이 다르면 A/B가 무의미.

### 1-3. A/B 토글 (테스트 브랜치 한정)

`ESKF.__init__`에 `self.reduce_cov = False` 플래그 추가. `update_zupt`/`update_mag`
에서 K 계산 직후:

```python
# Variant B 전용: Joseph form 공분산 축소
if self.reduce_cov:
    I = np.eye(15)
    self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T
```

- **A = 현행** (`reduce_cov=False`, 코드 거동 변화 0 → 회귀 불가능)
- **B = 수정** (`reduce_cov=True`)
- Joseph form을 쓰는 이유: 단순 `(I−KH)P`보다 수치적으로 대칭/양정치 보존이
  강건. K[6:9]=0 마스킹은 **그대로 유지**(자세 보호 의도 보존).

> 이 토글은 **테스트 전용 브랜치**에만 존재. main 머지 시점에 플래그를 제거하고
> 채택된 변종만 남긴다.

---

## 2. Phase 0 — 관측성 게이트 (오프라인, 가장 먼저)

같은 rawlog에 A와 B를 돌리고, **사용자 관측 가능 신호의 차이**를 본다.

| 측정 | 정의 | 게이트 |
|---|---|---|
| ΔCursor | 커서는 Madgwick 경로라 ESKF와 독립 → **이론상 Δ=0** 확인용(반증 테스트) | Δ≠0 이면 가정 오류, 조사 |
| Δq(s3_eskf) | A/B 쿼터니언 각도 차 `angle(qA·qB⁻¹)` 시계열 max/RMS | < 0.5° 이면 **E5 inert 판정** |
| Δkinematics | bio_kinematics(DifferentialKinematics) 최종 출력 A/B 차 | < 인지 임계 이면 inert |

**판정:**
- Δq, Δkinematics가 모두 무시할 수준 → **E5는 inert. 수정 중단, 종료.**
  (현재 구조상 가장 가능성 높은 결과. 이 경우 결과물은 "수정 안 함 + 근거 문서".)
- 유의미한 Δ 발생 → Phase 1로.

---

## 3. Phase 1 — 오프라인 정량 A/B (개선 입증 + 무회귀)

### 3-1. 필터 건전성 메트릭 (A vs B)

| 메트릭 | A(현행) 예상 | B(수정) 기대 | 합격 기준 |
|---|---|---|---|
| `trace(P_vv)` 장기(>60s) | 단조 폭주(수천~수만) | 유계(수렴) | B가 유계 |
| `cond(H P Hᵀ + R)` 최대 | 점증 | 안정 | B가 더 작음 |
| ZUPT 후 `‖v‖` | (감쇠로 가려짐) | 더 일관 | NaN/발산 0건 |
| `w_b` 시계열 표준편차 | — | 과도 변동 없음 | B가 A 대비 ↑하지 않음 |
| `a_b` 드리프트(정지→이동 전환 구간) | 과보정 가능 | 완화 | 전환 직후 스파이크 ↓ |

### 3-2. 회귀 탐지 메트릭 (가장 중요)

작성자가 잡아둔 **"버튼 뗄 때 튐(button-release jump)"**이 B에서 재발하는지.

- rawlog에서 `button: True→False` 전환 시점들을 인덱싱
- 각 전환 직후 N프레임(예: 0.3s) 동안 `s3_eskf.q`의 **각속도 점프**
  `max |Δangle/dt|` 를 A/B 비교
- 추가로, q가 bio_kinematics를 거쳐 만든 출력의 전환 직후 스파이크 비교

| 회귀 메트릭 | 합격 기준 |
|---|---|
| 버튼 전환 직후 q 각속도 점프 max | B ≤ A (악화 금지) |
| 정지 구간 q 드리프트율(°/min) | B ≤ A × 1.1 |
| NaN/Inf 발생 | 0건 (필수) |

### 3-3. 데이터셋

- **최소 6개 세션**, 각 ≥ 60초: (a) 일반 필기 ABCDEFGH, (b) 빠른 필기,
  (c) 장시간(>3분) 연속 — P 폭주가 드러나는 핵심 케이스, (d) 자주 멈추는 필기
  (ZUPT 빈발), (e) 정지 후 급출발 반복(전환 스트레스), (f) 큰 팔 동작 포함.
- 각 세션 rawlog 1개로 A/B 모두 재생(동일 입력 보장).

---

## 4. Phase 2 — 하드웨어 A/B (Phase 1 통과 시에만)

오프라인이 "B가 P를 유계로 만들고 회귀 메트릭 악화 없음"을 보였을 때만 진행.

### 4-1. 절차
1. 테스트 브랜치에서 `reduce_cov`를 **런타임 토글**(WS 명령 또는 env)로 노출.
2. 동일 피험자가 **고정 스크립트** 수행 (블라인드: 어떤 변종인지 모름):
   - S1: ABCDEFGH 5회 — 인식 정확도/지연
   - S2: 정지 후 급출발 20회 — 튐 관찰(회귀 핵심)
   - S3: 5분 연속 필기 — yaw/pitch 드리프트 누적
   - S4: 버튼 뗐다 다시 쓰기 30회 — button-release jump 직접 표적
3. A→B→A→B 교차 반복(순서 효과 상쇄), 각 세션 화면 녹화 + rawlog 동시 저장.

### 4-2. 합격 기준 (전부 충족해야 머지)
- S2/S4에서 **육안 튐이 A 대비 증가하지 않음** (녹화 블라인드 채점)
- S1 인식 정확도 A 대비 **−2%p 이내** (저하 금지)
- S3 드리프트 A 이하 또는 동등
- 전 세션 NaN/프리즈 0건
- 지연(input→커서) 증가 < 2ms (Joseph form 추가 연산 비용 확인)

### 4-3. 롤백 기준
위 중 **하나라도 미달 → B 폐기, A 유지.** "이론적으로 맞다"는 채택 사유가
될 수 없음. 관측 가능한 개선이 없거나 회귀가 있으면 되돌린다.

---

## 5. 산출물 / 체크리스트

- [ ] `--rawlog` 원시 프레임 로거 (1-1)
- [ ] `replay_eskf.py` 결정론적 재생기 (1-2), 라이브 분기와 비트 동일 검증
- [ ] `ESKF.reduce_cov` 토글 + Joseph form (1-3, 테스트 브랜치)
- [ ] Phase 0 관측성 리포트 → **inert면 여기서 종료**
- [ ] Phase 1 오프라인 A/B 리포트 (건전성 + 회귀 메트릭)
- [ ] Phase 2 하드웨어 블라인드 A/B 리포트
- [ ] 최종 결정: 머지 / 폐기 / 보류 + 근거

---

## 6. 위험 요약 (왜 이렇게까지 하나)

- **방치 위험:** P 폭주로 ZUPT 게인 포화 → 측정 갱신이 "전부/전무"화,
  교차항으로 `a_b` 과보정 → 정지→이동 전환 시 잠재적 튐. 단, 현재는
  속도 감쇠(0.90/0.995)·클램프(2 m/s)·잦은 reset이 이를 가림.
- **수정 위험:** `P=(I−KH)P`가 K 크기를 시간에 따라 줄임 → 감쇠/클램프/K마스킹
  등 **현행 거동을 전제로 손튜닝된 값들의 전제가 깨짐** → 작성자가 잡아둔
  button-release jump 재발 가능. 그래서 코드만 보고 머지 불가, 데이터로만 판정.
- **관측성 위험(신규 발견):** 커서가 Madgwick 경로라 **E5가 애초에 inert일
  가능성**이 높음. Phase 0가 이걸 최우선으로 거른다 — inert면 수정은 순손해.
