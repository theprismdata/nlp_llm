"""
커스텀 BPE 토크나이저 구현 (Qwen2.5와 유사)
작성자: theprismdata@gmail.com
"""

import json
import pickle
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Set, Optional


class CustomBPETokenizer:
    """
    BPE(Byte Pair Encoding) 기반 커스텀 토크나이저
    작성자: theprismdata@gmail.com
    """
    
    def __init__(self, vocab_size: int = 10000):
        """
        토크나이저 초기화
        작성자: theprismdata@gmail.com
        
        Args:
            vocab_size: 목표 어휘 크기
        """
        self.vocab_size = vocab_size
        self.vocab = {}  # token -> id
        self.inverse_vocab = {}  # id -> token
        self.merges = {}  # (pair) -> merged_token
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3,
        }
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3
        
        # 특수 토큰 추가
        for token, idx in self.special_tokens.items():
            self.vocab[token] = idx
            self.inverse_vocab[idx] = token
    
    def _get_stats(self, word_freqs: Dict[Tuple[str, ...], int]) -> Counter:
        """
        단어 쌍의 빈도를 계산합니다.
        작성자: theprismdata@gmail.com
        
        Args:
            word_freqs: 단어와 빈도 딕셔너리
            
        Returns:
            쌍의 빈도 Counter
        """
        pairs = Counter()
        for word, freq in word_freqs.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += freq
        return pairs
    
    def _merge_pair(self, pair: Tuple[str, str], word_freqs: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, ...], int]:
        """
        주어진 쌍을 병합합니다.
        작성자: theprismdata@gmail.com
        
        Args:
            pair: 병합할 쌍
            word_freqs: 단어와 빈도 딕셔너리
            
        Returns:
            병합된 단어와 빈도 딕셔너리
        """
        new_word_freqs = {}
        merged = ''.join(pair)
        
        for word, freq in word_freqs.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                    new_word.append(merged)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_freqs[tuple(new_word)] = freq
        
        return new_word_freqs
    
    def train(self, texts: List[str], verbose: bool = True):
        """
        텍스트 데이터로 토크나이저를 학습합니다.
        작성자: theprismdata@gmail.com
        
        Args:
            texts: 학습할 텍스트 리스트
            verbose: 진행 상황 출력 여부
        """
        if verbose:
            print(f"토크나이저 학습 시작...")
            print(f"  텍스트 수: {len(texts)}")
            print(f"  목표 어휘 크기: {self.vocab_size}")
        
        # 텍스트 전처리 및 단어 빈도 계산
        word_freqs = Counter()
        for text in texts:
            # 공백으로 단어 분리
            words = text.split()
            for word in words:
                # 각 문자를 개별 토큰으로 분리 (단어 끝 표시 추가)
                chars = tuple(list(word) + ['</w>'])
                word_freqs[chars] += 1
        
        if verbose:
            print(f"  고유 단어 수: {len(word_freqs)}")
        
        # 모든 개별 문자를 어휘에 추가
        base_vocab = set()
        for word in word_freqs.keys():
            base_vocab.update(word)
        
        current_vocab_size = len(self.special_tokens)
        for char in sorted(base_vocab):
            if char not in self.vocab:
                self.vocab[char] = current_vocab_size
                self.inverse_vocab[current_vocab_size] = char
                current_vocab_size += 1
        
        if verbose:
            print(f"  기본 어휘 크기 (특수 토큰 + 문자): {current_vocab_size}")
        
        # BPE 병합 수행
        num_merges = self.vocab_size - current_vocab_size
        merge_count = 0
        
        while merge_count < num_merges:
            # 가장 빈도가 높은 쌍 찾기
            pairs = self._get_stats(word_freqs)
            if not pairs:
                break
            
            best_pair = max(pairs, key=pairs.get)
            
            # 쌍 병합
            word_freqs = self._merge_pair(best_pair, word_freqs)
            
            # 병합된 토큰을 어휘에 추가
            merged_token = ''.join(best_pair)
            self.merges[best_pair] = merged_token
            self.vocab[merged_token] = current_vocab_size
            self.inverse_vocab[current_vocab_size] = merged_token
            current_vocab_size += 1
            merge_count += 1
            
            if verbose and (merge_count % 100 == 0 or merge_count == num_merges):
                print(f"  병합 진행: {merge_count}/{num_merges} - 최신 병합: {best_pair} -> {merged_token}")
        
        if verbose:
            print(f"\n토크나이저 학습 완료!")
            print(f"  최종 어휘 크기: {len(self.vocab)}")
            print(f"  병합 수: {len(self.merges)}")
    
    def _tokenize_word(self, word: str) -> List[str]:
        """
        단어를 토큰으로 분리합니다.
        작성자: theprismdata@gmail.com
        
        Args:
            word: 토큰화할 단어
            
        Returns:
            토큰 리스트
        """
        # 문자 단위로 분리
        tokens = list(word) + ['</w>']
        
        # BPE 병합 적용
        while len(tokens) > 1:
            # 현재 토큰에서 가능한 쌍 찾기
            pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
            
            # 학습된 병합 중에서 찾기
            valid_pairs = [pair for pair in pairs if pair in self.merges]
            if not valid_pairs:
                break
            
            # 가장 먼저 병합된 쌍 선택 (학습 순서 기준)
            best_pair = min(valid_pairs, key=lambda p: list(self.merges.keys()).index(p))
            
            # 쌍 병합
            merged_token = self.merges[best_pair]
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == best_pair[0] and tokens[i + 1] == best_pair[1]:
                    new_tokens.append(merged_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        
        return tokens
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        텍스트를 토큰 ID 리스트로 변환합니다.
        작성자: theprismdata@gmail.com
        
        Args:
            text: 인코딩할 텍스트
            add_special_tokens: 특수 토큰 추가 여부
            
        Returns:
            토큰 ID 리스트
        """
        # 단어 분리
        words = text.split()
        
        # 각 단어를 토큰화
        token_ids = []
        if add_special_tokens:
            token_ids.append(self.bos_token_id)
        
        for word in words:
            tokens = self._tokenize_word(word)
            for token in tokens:
                token_id = self.vocab.get(token, self.unk_token_id)
                token_ids.append(token_id)
        
        if add_special_tokens:
            token_ids.append(self.eos_token_id)
        
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        토큰 ID 리스트를 텍스트로 변환합니다.
        작성자: theprismdata@gmail.com
        
        Args:
            token_ids: 디코딩할 토큰 ID 리스트
            skip_special_tokens: 특수 토큰 건너뛰기 여부
            
        Returns:
            디코딩된 텍스트
        """
        tokens = []
        for token_id in token_ids:
            if skip_special_tokens and token_id in self.special_tokens.values():
                continue
            token = self.inverse_vocab.get(token_id, '<UNK>')
            tokens.append(token)
        
        # 토큰을 텍스트로 결합 (</w>를 공백으로 변환)
        text = ''.join(tokens).replace('</w>', ' ').strip()
        return text
    
    def convert_ids_to_tokens(self, token_ids: List[int]) -> List[str]:
        """
        토큰 ID를 토큰 문자열로 변환합니다.
        작성자: theprismdata@gmail.com
        
        Args:
            token_ids: 토큰 ID 리스트
            
        Returns:
            토큰 문자열 리스트
        """
        return [self.inverse_vocab.get(tid, '<UNK>') for tid in token_ids]
    
    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """
        토큰 문자열을 토큰 ID로 변환합니다.
        작성자: theprismdata@gmail.com
        
        Args:
            tokens: 토큰 문자열 리스트
            
        Returns:
            토큰 ID 리스트
        """
        return [self.vocab.get(token, self.unk_token_id) for token in tokens]
    
    def save(self, file_path: str):
        """
        토크나이저를 파일로 저장합니다.
        작성자: theprismdata@gmail.com
        
        Args:
            file_path: 저장할 파일 경로
        """
        data = {
            'vocab_size': self.vocab_size,
            'vocab': self.vocab,
            'inverse_vocab': self.inverse_vocab,
            'merges': self.merges,
            'special_tokens': self.special_tokens,
        }
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"토크나이저 저장 완료: {file_path}")
    
    @classmethod
    def load(cls, file_path: str):
        """
        파일에서 토크나이저를 로드합니다.
        작성자: theprismdata@gmail.com
        
        Args:
            file_path: 로드할 파일 경로
            
        Returns:
            로드된 토크나이저
        """
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        tokenizer = cls(vocab_size=data['vocab_size'])
        tokenizer.vocab = data['vocab']
        tokenizer.inverse_vocab = data['inverse_vocab']
        tokenizer.merges = data['merges']
        tokenizer.special_tokens = data['special_tokens']
        print(f"토크나이저 로드 완료: {file_path}")
        return tokenizer
    
    def __call__(self, text: str, max_length: int = None, truncation: bool = False, 
                 padding: bool = False, return_tensors: str = None, add_special_tokens: bool = True):
        """
        transformers 스타일의 호출 인터페이스
        작성자: theprismdata@gmail.com
        
        Args:
            text: 인코딩할 텍스트
            max_length: 최대 길이
            truncation: 트렁케이션 여부
            padding: 패딩 여부
            return_tensors: 반환 텐서 타입
            add_special_tokens: 특수 토큰 추가 여부
            
        Returns:
            인코딩 결과 딕셔너리
        """
        token_ids = self.encode(text, add_special_tokens=add_special_tokens)
        
        # 트렁케이션
        if truncation and max_length and len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        
        # 패딩
        if padding and max_length:
            if len(token_ids) < max_length:
                token_ids = token_ids + [self.pad_token_id] * (max_length - len(token_ids))
        
        return {'input_ids': token_ids}


def train_custom_tokenizer_from_json(json_file: str, vocab_size: int = 10000, 
                                     max_samples: int = None, save_path: str = None):
    """
    JSON 파일에서 텍스트를 로드하여 커스텀 토크나이저를 학습합니다.
    작성자: theprismdata@gmail.com
    
    Args:
        json_file: JSON 파일 경로
        vocab_size: 어휘 크기
        max_samples: 최대 샘플 수 (None이면 전체)
        save_path: 저장 경로 (None이면 저장 안 함)
        
    Returns:
        학습된 토크나이저
    """
    print(f"JSON 파일 로드 중: {json_file}")
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    data_info = data.get('data_info', [])
    if max_samples:
        data_info = data_info[:max_samples]
    
    print(f"데이터 항목 수: {len(data_info)}")
    
    # 텍스트 추출
    texts = []
    for item in data_info:
        contents = item.get('contents', '')
        if contents:
            texts.append(contents)
    
    print(f"텍스트 수: {len(texts)}")
    
    # 토크나이저 학습
    tokenizer = CustomBPETokenizer(vocab_size=vocab_size)
    tokenizer.train(texts, verbose=True)
    
    # 저장
    if save_path:
        tokenizer.save(save_path)
    
    return tokenizer


if __name__ == "__main__":
    # 예제: 커스텀 토크나이저 학습
    print("="*80)
    print("커스텀 BPE 토크나이저 학습 시작")
    print("="*80)
    
    # JSON 파일에서 학습
    tokenizer = train_custom_tokenizer_from_json(
        json_file='SL01-00-00.json',
        vocab_size=10000,
        max_samples=100,  # 빠른 테스트를 위해 100개만 사용
        save_path='custom_tokenizer.pkl'
    )
    
    # 테스트
    print("\n" + "="*80)
    print("토크나이저 테스트")
    print("="*80)
    
    test_text = "안녕하세요. 여러분 흑열전구입니다."
    print(f"\n원본 텍스트: {test_text}")
    
    # 인코딩
    token_ids = tokenizer.encode(test_text)
    print(f"토큰 ID: {token_ids}")
    
    # 토큰 확인
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    print(f"토큰: {tokens}")
    
    # 디코딩
    decoded_text = tokenizer.decode(token_ids)
    print(f"디코딩된 텍스트: {decoded_text}")
    
    print("\n" + "="*80)
    print("완료!")
    print("="*80)

