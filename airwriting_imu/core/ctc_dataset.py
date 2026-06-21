"""
CTC Dataset — CTC 학습을 위한 가변 길이 시퀀스 데이터셋
=======================================================

기존 GestureDataset과의 핵심 차이:
  - GestureDataset: 200프레임 고정 리샘플링 → 단일 라벨 (분류)
  - CTCDataset: 원본 길이 보존 + 패딩 → 문자열 라벨 (시퀀스 인식)

CTC 학습에서 시간축이 중요한 이유:
  리샘플링하면 "쓰는 속도" 정보가 사라져서 blank/반복 패턴을 학습할 수 없음.
  원본 길이를 유지해야 모델이 "이 구간은 아직 글자를 안 쓰고 있다(blank)" vs
  "이 구간에서 글자를 쓰고 있다(character)"를 구별할 수 있음.

두 가지 모드:
  1. single_char: 기존 단일 글자 JSON → 각각 1글자 CTC 학습
  2. word: 여러 글자 JSON을 랜덤 연결하여 합성 단어 생성 (데이터 증강)
"""

import os
import json
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Optional, Dict
from pathlib import Path

# 프로젝트 경로
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DATASET_DIR = _PROJECT_ROOT / "dataset"


class CTCDataset(Dataset):
    """
    CTC 학습용 데이터셋.
    
    각 샘플:
      - features: [T_i, 11] 가변 길이 특징 벡터 (패딩 전)
      - target: [S_i] 문자 인덱스 시퀀스 (A=1, B=2, ..., Z=26, blank=0)
      
    DataLoader 사용 시 반드시 CTCDataset.collate_fn을 사용해야 함:
      loader = DataLoader(dataset, batch_size=16, collate_fn=CTCDataset.collate_fn)
    """
    
    # 문자 → 인덱스 매핑 (CTC blank=0이므로 문자는 1부터)
    BLANK_IDX = 0
    CHAR_TO_IDX = {c: i + 1 for i, c in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ")}
    IDX_TO_CHAR = {v: k for k, v in CHAR_TO_IDX.items()}
    
    def __init__(
        self,
        data_dir: Optional[str] = None,
        mode: str = "single_char",       # "single_char" 또는 "word"
        word_length_range: Tuple[int, int] = (2, 5),  # word 모드 합성 단어 길이
        synth_samples: int = 500,         # word 모드 합성 샘플 수
        min_seq_len: int = 5,             # 최소 시퀀스 길이 (너무 짧은 것 제거)
        normalize: bool = True,           # StandardScaler 적용 여부
    ):
        if data_dir is None:
            data_dir = str(_DATASET_DIR)
        
        self.data_dir = data_dir
        self.mode = mode
        self.word_length_range = word_length_range
        self.synth_samples = synth_samples
        self.min_seq_len = min_seq_len
        self.normalize = normalize
        
        # 데이터 저장소
        self.samples: List[np.ndarray] = []     # 각 원소: [T_i, 11]
        self.targets: List[List[int]] = []      # 각 원소: [S_i]
        self.target_strings: List[str] = []     # 디버그용 원본 문자열
        
        # 글자별 원본 데이터 (word 합성용)
        self._char_pool: Dict[str, List[np.ndarray]] = {}
        
        self.scaler = StandardScaler() if normalize else None
        
        self._load_and_build()
    
    def _extract_features(self, strokes_list: list) -> np.ndarray:
        """Thin wrapper over ai_model.extract_features: float32 for CTC training."""
        from airwriting_imu.core.ai_model import extract_features
        arr = extract_features(strokes_list)
        if arr.size == 0:
            return np.zeros((0, 11), dtype=np.float32)
        return arr.astype(np.float32, copy=False)
    
    def _load_and_build(self):
        """데이터 로드 + 모드별 데이터셋 구성"""
        files = sorted(glob.glob(os.path.join(self.data_dir, "*.json")))
        
        if not files:
            print(f"[CTCDataset] 데이터 없음: {self.data_dir}")
            return
        
        # Phase 1: 모든 JSON 로드 → 글자별 풀 구성
        for f in files:
            try:
                with open(f, 'r', encoding='utf-8') as fp:
                    data = json.load(fp)
                
                if isinstance(data, dict) and "strokes" in data:
                    strokes_list = data["strokes"]
                    label = data.get("label", "Unknown").upper()
                else:
                    strokes_list = [data]
                    label = os.path.basename(f).split('_')[0].upper()
                
                # 유효한 A-Z만 허용
                if label not in self.CHAR_TO_IDX:
                    continue
                
                features = self._extract_features(strokes_list)
                
                if len(features) < self.min_seq_len:
                    continue
                
                if label not in self._char_pool:
                    self._char_pool[label] = []
                self._char_pool[label].append(features)
                
            except Exception as e:
                print(f"[CTCDataset] 로드 실패 {f}: {e}")
        
        available_chars = sorted(self._char_pool.keys())
        total_samples = sum(len(v) for v in self._char_pool.values())
        print(f"[CTCDataset] 로드 완료: {len(available_chars)}개 문자, {total_samples}개 샘플")
        print(f"[CTCDataset] 문자 분포: {', '.join(f'{c}:{len(self._char_pool[c])}' for c in available_chars)}")
        
        if not self._char_pool:
            return
        
        # Phase 2: 모드별 데이터셋 구성
        if self.mode == "single_char":
            self._build_single_char()
        elif self.mode == "word":
            self._build_single_char()  # 기본 단일 글자도 포함
            self._build_synthetic_words()  # + 합성 단어 추가
        
        # Phase 3: 정규화 (scaler)
        if self.normalize and len(self.samples) > 0:
            all_points = np.vstack(self.samples)
            self.scaler.fit(all_points)
            self.samples = [self.scaler.transform(s) for s in self.samples]
        
        print(f"[CTCDataset] 최종 데이터셋: {len(self.samples)}개 샘플 (모드: {self.mode})")
    
    def _build_single_char(self):
        """단일 글자 모드: 각 JSON = 1글자 CTC 샘플"""
        for char, sequences in self._char_pool.items():
            char_idx = self.CHAR_TO_IDX[char]
            for seq in sequences:
                self.samples.append(seq)
                self.targets.append([char_idx])
                self.target_strings.append(char)
    
    def _build_synthetic_words(self):
        """
        합성 단어 모드: 개별 글자 시퀀스를 랜덤 연결하여 단어 생성.
        
        예: A 시퀀스 + 공백구간 + B 시퀀스 + 공백구간 + C 시퀀스 → "ABC"
        
        글자 사이에 "쓰지 않는 구간"(정지 상태 프레임)을 삽입하여
        CTC가 blank를 학습할 수 있게 함.
        """
        available_chars = list(self._char_pool.keys())
        if len(available_chars) < 2:
            print("[CTCDataset] 합성 단어 생성 불가: 2종류 이상의 글자 필요")
            return
        
        rng = np.random.RandomState(42)
        generated = 0
        
        for _ in range(self.synth_samples):
            # 랜덤 단어 길이
            word_len = rng.randint(self.word_length_range[0], self.word_length_range[1] + 1)
            
            # 랜덤 글자 선택
            word_chars = [available_chars[rng.randint(0, len(available_chars))] for _ in range(word_len)]
            
            # 각 글자의 시퀀스를 랜덤 선택 + 연결
            word_sequence_parts = []
            target_indices = []
            
            for i, char in enumerate(word_chars):
                pool = self._char_pool[char]
                char_seq = pool[rng.randint(0, len(pool))].copy()
                
                # 글자 간 공백 삽입 (첫 글자 앞엔 불필요)
                if i > 0:
                    gap_frames = self._generate_gap_frames(
                        num_frames=rng.randint(5, 20),  # 60~240ms 정도 공백
                        last_point=word_sequence_parts[-1][-1] if word_sequence_parts else None,
                    )
                    word_sequence_parts.append(gap_frames)
                
                word_sequence_parts.append(char_seq)
                target_indices.append(self.CHAR_TO_IDX[char])
            
            # 연결
            word_sequence = np.vstack(word_sequence_parts)
            
            self.samples.append(word_sequence)
            self.targets.append(target_indices)
            self.target_strings.append("".join(word_chars))
            generated += 1
        
        print(f"[CTCDataset] 합성 단어 {generated}개 생성 (길이 {self.word_length_range[0]}-{self.word_length_range[1]}글자)")
    
    def _generate_gap_frames(self, num_frames: int, last_point: Optional[np.ndarray] = None) -> np.ndarray:
        """
        글자 사이의 "비필기 구간" 프레임 생성.
        
        정지 상태를 시뮬레이션: 위치 유지, 가속도 ≈ 중력, 자이로 ≈ 0
        이 구간에서 CTC는 blank를 출력하도록 학습됨.
        """
        gap = np.zeros((num_frames, 11), dtype=np.float32)
        
        if last_point is not None:
            # 마지막 위치 유지 (x, y)
            gap[:, 0] = last_point[0]  # x
            gap[:, 1] = last_point[1]  # y
        
        # dx, dy = 0 (이동 없음)
        # is_new_stroke = 0
        
        # 정지 가속도 (중력): az ≈ 9.81 + 작은 노이즈
        gap[:, 7] = 9.81 + np.random.randn(num_frames) * 0.05  # az
        gap[:, 5] = np.random.randn(num_frames) * 0.1  # ax (노이즈)
        gap[:, 6] = np.random.randn(num_frames) * 0.1  # ay (노이즈)
        
        # 자이로 ≈ 0 (정지)
        gap[:, 8:11] = np.random.randn(num_frames, 3) * 0.01
        
        return gap
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        """
        Returns:
            features: [T_i, 11] — 가변 길이 시퀀스
            target: [S_i] — 타겟 문자 인덱스
            input_length: 시퀀스 길이
            target_length: 타겟 문자열 길이
        """
        features = torch.tensor(self.samples[idx], dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.long)
        
        return features, target, len(features), len(target)
    
    @staticmethod
    def collate_fn(batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        가변 길이 배치를 패딩하여 텐서로 변환.
        
        DataLoader에 반드시 이 함수를 전달해야 함:
          loader = DataLoader(dataset, collate_fn=CTCDataset.collate_fn)
        
        Returns:
            features_padded: [B, T_max, 11] — 제로 패딩된 배치
            targets: [sum(S_i)] — 1D concatenated 타겟 (CTC 표준 포맷)
            input_lengths: [B] — 각 샘플의 원본 시퀀스 길이
            target_lengths: [B] — 각 샘플의 타겟 문자열 길이
        """
        features_list, targets_list, input_lengths, target_lengths = zip(*batch)
        
        # 시퀀스 패딩 (길이 맞추기)
        features_padded = pad_sequence(features_list, batch_first=True, padding_value=0.0)
        
        # 타겟은 1D로 concat (CTC 표준)
        targets_concat = torch.cat(targets_list)
        
        input_lengths = torch.tensor(input_lengths, dtype=torch.long)
        target_lengths = torch.tensor(target_lengths, dtype=torch.long)
        
        return features_padded, targets_concat, input_lengths, target_lengths
    
    def get_vocab_size(self) -> int:
        """CTC 모델에 전달할 vocab 크기 (blank 포함)"""
        return len(self.CHAR_TO_IDX) + 1  # +1 for blank
    
    def get_stats(self) -> dict:
        """데이터셋 통계"""
        if not self.samples:
            return {"total": 0}
        
        lengths = [len(s) for s in self.samples]
        target_lens = [len(t) for t in self.targets]
        
        return {
            "total_samples": len(self.samples),
            "available_chars": sorted(self._char_pool.keys()),
            "num_chars": len(self._char_pool),
            "seq_length": {
                "min": min(lengths),
                "max": max(lengths),
                "mean": round(np.mean(lengths), 1),
                "median": round(np.median(lengths), 1),
            },
            "target_length": {
                "min": min(target_lens),
                "max": max(target_lens),
                "mean": round(np.mean(target_lens), 1),
            },
            "mode": self.mode,
        }


# ═══════════════════════════════════════════════════════════════════
# Self-Test
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    
    print("=" * 60)
    print("  CTCDataset — CTC 학습용 가변 길이 데이터셋")
    print("=" * 60)
    
    # 1. 단일 글자 모드 테스트
    print("\n--- 모드 1: single_char ---")
    ds_single = CTCDataset(mode="single_char")
    stats = ds_single.get_stats()
    print(f"   통계: {json.dumps(stats, indent=2, ensure_ascii=False)}")
    
    if len(ds_single) > 0:
        # DataLoader 테스트
        loader = DataLoader(ds_single, batch_size=8, shuffle=True, collate_fn=CTCDataset.collate_fn)
        batch = next(iter(loader))
        features, targets, input_lengths, target_lengths = batch
        print(f"\n   배치 테스트:")
        print(f"   features: {features.shape}")
        print(f"   targets: {targets.shape} (concat)")
        print(f"   input_lengths: {input_lengths.tolist()}")
        print(f"   target_lengths: {target_lengths.tolist()}")
    
    # 2. 합성 단어 모드 테스트
    print("\n--- 모드 2: word (합성 단어) ---")
    ds_word = CTCDataset(mode="word", synth_samples=100, word_length_range=(2, 4))
    stats_word = ds_word.get_stats()
    print(f"   통계: {json.dumps(stats_word, indent=2, ensure_ascii=False)}")
    
    if len(ds_word) > 0:
        # 합성 단어 샘플 확인
        print(f"\n   합성 단어 예시:")
        for i in range(min(10, len(ds_word))):
            if len(ds_word.target_strings[i]) > 1:
                print(f"   [{i}] '{ds_word.target_strings[i]}' — {len(ds_word.samples[i])} frames")
    
    print(f"\n✅ CTCDataset 검증 완료!")
