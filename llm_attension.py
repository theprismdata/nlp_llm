"""
Self Attention 구현
SL01-00-00.json 파일을 입력으로 받아 self attention을 수행합니다.
작성자: theprismdata@gmail.com
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import io
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer


def load_json_data(file_path: str) -> List[Dict]:
    """
    JSON 파일을 로드하여 데이터를 반환합니다.
    작성자: theprismdata@gmail.com
    
    Args:
        file_path: JSON 파일 경로
        
    Returns:
        data_info 리스트
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('data_info', [])


def load_tokenizer() -> AutoTokenizer:
    """
    Qwen2.5 토크나이저를 로드합니다.
    작성자: theprismdata@gmail.com
        
    Returns:
        Qwen2.5 토크나이저 객체
    """
    model_name = "Qwen/Qwen2.5-0.5B"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print(f"Qwen2.5 토크나이저 로드 완료: {model_name}")
        return tokenizer
    except Exception as e:
        raise Exception(f"Qwen2.5 토크나이저를 로드할 수 없습니다: {e}\ntransformers 라이브러리가 설치되어 있는지 확인하세요.")


def tokenize_text(text: str, tokenizer: AutoTokenizer, max_length: int = 512) -> List[int]:
    """
    텍스트를 토크나이저를 사용하여 토큰화합니다.
    작성자: theprismdata@gmail.com
    
    Args:
        text: 토큰화할 텍스트
        tokenizer: 토크나이저 객체
        max_length: 최대 시퀀스 길이
        
    Returns:
        토큰 ID 리스트
    """
    # 토크나이저로 인코딩
    encoded = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding=False,
        return_tensors=None
    )
    return encoded['input_ids']


def create_embeddings(vocab_size: int, d_model: int) -> nn.Embedding:
    """
    임베딩 레이어를 생성합니다.
    작성자: theprismdata@gmail.com
    
    Args:
        vocab_size: 어휘 크기
        d_model: 임베딩 차원
        
    Returns:
        임베딩 레이어
    """
    return nn.Embedding(vocab_size, d_model)


class TransformerModel(nn.Module):
    """
    Transformer 모델 (Embedding + Self Attention + 출력 레이어)
    작성자: theprismdata@gmail.com
    """
    
    def __init__(self, vocab_size: int, d_model: int = 128, num_heads: int = 8, num_layers: int = 1, dropout: float = 0.1):
        """
        Transformer 모델 초기화
        작성자: theprismdata@gmail.com
        
        Args:
            vocab_size: 어휘 크기
            d_model: 모델 차원
            num_heads: 어텐션 헤드 수
            num_layers: Self Attention 레이어 수
            dropout: 드롭아웃 비율
        """
        super(TransformerModel, self).__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # 임베딩 레이어
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Self Attention 레이어들
        self.attention_layers = nn.ModuleList([
            SelfAttention(d_model, num_heads, dropout) for _ in range(num_layers)
        ])
        
        # Layer Normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])
        
        # 출력 레이어 (언어 모델링용)
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        Forward pass
        작성자: theprismdata@gmail.com
        
        Args:
            x: 입력 토큰 텐서 (batch_size, seq_len)
            mask: 마스크 텐서 (batch_size, seq_len, seq_len) 또는 None
            
        Returns:
            출력 로짓 (batch_size, seq_len, vocab_size)과 attention weights 리스트
        """
        # 임베딩 (토큰 ID가 vocab_size를 초과하지 않도록 클램핑)
        x_clamped = torch.clamp(x, 0, self.vocab_size - 1)
        embedded = self.embedding(x_clamped) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32, device=x.device))  # 스케일링
        
        # Self Attention 레이어들 통과
        attention_weights_list = []
        output = embedded
        
        for attention_layer, layer_norm in zip(self.attention_layers, self.layer_norms):
            # Self Attention
            attn_output, attn_weights = attention_layer(output, mask)
            
            # Residual connection + Layer Normalization
            output = layer_norm(output + self.dropout(attn_output))
            attention_weights_list.append(attn_weights)
        
        # 출력 프로젝션
        logits = self.output_projection(output)
        
        return logits, attention_weights_list


class SelfAttention(nn.Module):
    """
    Self Attention 메커니즘 구현
    작성자: theprismdata@gmail.com
    """
    
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        """
        Self Attention 초기화
        작성자: theprismdata@gmail.com
        
        Args:
            d_model: 모델 차원
            num_heads: 어텐션 헤드 수
            dropout: 드롭아웃 비율
        """
        super(SelfAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model은 num_heads로 나누어떨어져야 합니다"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Query, Key, Value 선형 변환 레이어
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # 출력 선형 변환 레이어
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = np.sqrt(self.d_k)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Self Attention forward pass
        작성자: theprismdata@gmail.com
        
        Args:
            x: 입력 텐서 (batch_size, seq_len, d_model)
            mask: 마스크 텐서 (batch_size, seq_len, seq_len) 또는 None
            
        Returns:
            어텐션 출력 (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.size()
        
        # Query, Key, Value 계산
        Q = self.W_q(x)  # (batch_size, seq_len, d_model)
        K = self.W_k(x)  # (batch_size, seq_len, d_model)
        V = self.W_v(x)  # (batch_size, seq_len, d_model)
        
        # Multi-head를 위해 reshape
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)  # (batch_size, num_heads, seq_len, d_k)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)  # (batch_size, num_heads, seq_len, d_k)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)  # (batch_size, num_heads, seq_len, d_k)
        
        # Attention scores 계산
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (batch_size, num_heads, seq_len, seq_len)
        
        # 마스크 적용 (있는 경우)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax 적용
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Value와 가중치를 곱하여 어텐션 출력 계산
        attention_output = torch.matmul(attention_weights, V)  # (batch_size, num_heads, seq_len, d_k)
        
        # Multi-head를 다시 결합
        attention_output = attention_output.transpose(1, 2).contiguous()  # (batch_size, seq_len, num_heads, d_k)
        attention_output = attention_output.view(batch_size, seq_len, self.d_model)  # (batch_size, seq_len, d_model)
        
        # 최종 출력
        output = self.W_o(attention_output)
        
        return output, attention_weights


def process_text_data(data_info: List[Dict], tokenizer: AutoTokenizer, max_length: int = 512) -> Tuple[torch.Tensor, int, List[List[int]]]:
    """
    텍스트 데이터를 처리하여 토큰 ID 텐서로 변환합니다.
    작성자: theprismdata@gmail.com
    
    Args:
        data_info: JSON에서 로드한 데이터 정보 리스트
        tokenizer: 토크나이저 객체
        max_length: 최대 시퀀스 길이
        
    Returns:
        토큰 ID 텐서, 어휘 크기, 원본 토큰 ID 리스트
    """
    all_token_ids = []
    original_token_ids = []
    
    # 모든 텍스트를 토큰화
    for item in data_info:
        contents = item.get('contents', '')
        if contents:
            token_ids = tokenize_text(contents, tokenizer, max_length)
            original_token_ids.append(token_ids.copy())
            all_token_ids.append(token_ids)
    
    # 패딩 토큰 ID 가져오기
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_token_id is None:
        pad_token_id = 0
    
    # 어휘 크기 가져오기
    vocab_size = tokenizer.vocab_size
    
    # 시퀀스 길이를 max_length로 맞추기 (패딩 또는 트렁케이션)
    processed_token_ids = []
    for token_ids in all_token_ids:
        # 토큰 ID가 vocab_size를 초과하지 않도록 클램핑
        token_ids = [min(tid, vocab_size - 1) for tid in token_ids]
        
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        elif len(token_ids) < max_length:
            # 패딩 추가 (패딩 토큰도 vocab_size 내로)
            pad_id = min(pad_token_id, vocab_size - 1)
            token_ids = token_ids + [pad_id] * (max_length - len(token_ids))
        processed_token_ids.append(token_ids)
    
    # 텐서로 변환
    token_tensor = torch.tensor(processed_token_ids, dtype=torch.long)
    
    return token_tensor, vocab_size, original_token_ids


def tokens_to_text(token_ids: List[int], tokenizer: AutoTokenizer) -> List[str]:
    """
    토큰 ID 리스트를 텍스트로 변환합니다.
    작성자: theprismdata@gmail.com
    
    Args:
        token_ids: 토큰 ID 리스트
        tokenizer: 토크나이저 객체
        
    Returns:
        토큰 텍스트 리스트 (한글 포함)
    """
    # 각 토큰을 개별적으로 디코딩하여 한글이 제대로 표시되도록 함
    token_texts = []
    for tid in token_ids:
        try:
            # 토큰을 텍스트로 변환
            token_text = tokenizer.decode([tid], skip_special_tokens=False)
            # 특수 토큰 처리
            if token_text.strip() == "" or len(token_text) == 0:
                token_text = tokenizer.convert_ids_to_tokens([tid])[0]
                # 서브워드 토큰의 특수 문자 제거
                if token_text.startswith('##') or token_text.startswith('▁'):
                    token_text = token_text.lstrip('##').lstrip('▁')
            # 공백 문자 정리
            token_text = token_text.replace(' ', '').replace('\n', '').replace('\t', '')
            if not token_text:
                token_text = f"[{tid}]"
            token_texts.append(token_text)
        except Exception as e:
            # 디코딩 실패 시 토큰 ID 표시
            token_texts.append(f"[{tid}]")
    return token_texts


def get_original_text_tokens(data_info: List[Dict], tokenizer: AutoTokenizer, batch_idx: int = 0, max_length: int = 256) -> Tuple[List[str], List[int]]:
    """
    원본 텍스트를 단어 단위로 분리하여 반환합니다.
    작성자: theprismdata@gmail.com
    
    Args:
        data_info: JSON 데이터 정보
        tokenizer: 토크나이저 객체
        batch_idx: 배치 인덱스
        max_length: 최대 길이
        
    Returns:
        단어 리스트와 토큰 ID 리스트
    """
    if batch_idx >= len(data_info):
        return [], []
    
    contents = data_info[batch_idx].get('contents', '')
    if not contents:
        return [], []
    
    # 원본 텍스트를 단어 단위로 분리
    words = contents.split()
    words = words[:max_length]  # 최대 길이 제한
    
    # 각 단어를 토큰화하여 매핑
    token_ids = []
    word_to_tokens = []
    
    for word in words:
        word_tokens = tokenizer.encode(word, add_special_tokens=False)
        token_ids.extend(word_tokens)
        word_to_tokens.append((word, word_tokens))
    
    # 토큰 ID를 max_length로 제한
    token_ids = token_ids[:max_length]
    
    return words, token_ids


def analyze_word_relationships(
    attention_weights: torch.Tensor,
    token_ids: torch.Tensor,
    tokenizer: AutoTokenizer,
    data_info: List[Dict] = None,
    batch_idx: int = 0,
    head_idx: int = 0,
    top_k: int = 5,
    max_tokens: int = 20,
    output_file: str = None
):
    """
    각 단어별로 다른 단어들과의 관계를 분석하여 출력합니다.
    작성자: theprismdata@gmail.com
    
    Args:
        attention_weights: Attention weights 텐서 (batch_size, num_heads, seq_len, seq_len)
        token_ids: 토큰 ID 텐서 (batch_size, seq_len)
        tokenizer: 토크나이저 객체
        data_info: 원본 데이터 정보 (한글 표시용)
        batch_idx: 분석할 배치 인덱스
        head_idx: 분석할 헤드 인덱스
        top_k: 각 단어별로 보여줄 상위 관계 단어 수
        max_tokens: 분석할 최대 토큰 수
    """
    # 패딩 토큰 ID 가져오기
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    
    # 특정 배치와 헤드의 attention weights 추출
    attn = attention_weights[batch_idx, head_idx].detach().cpu()  # (seq_len, seq_len)
    
    # 토큰 ID 추출
    tokens = token_ids[batch_idx].detach().cpu().tolist()
    
    # 패딩 토큰 제외한 실제 토큰만 분석
    valid_indices = [i for i, tid in enumerate(tokens) if tid != pad_token_id]
    if len(valid_indices) == 0:
        print("분석할 유효한 토큰이 없습니다.")
        return
    
    # 최대 토큰 수로 제한
    valid_indices = valid_indices[:max_tokens]
    
    # 토큰을 텍스트로 변환 (한글 제대로 표시)
    token_texts = tokens_to_text(tokens, tokenizer)
    
    # 원본 텍스트에서 단어 매핑 시도
    original_words = None
    word_token_mapping = None
    if data_info and batch_idx < len(data_info):
        try:
            contents = data_info[batch_idx].get('contents', '')
            if contents:
                # 원본 텍스트를 단어 단위로 분리
                words = contents.split()
                original_words = words[:max_tokens]
                
                # 각 단어가 어떤 토큰 인덱스에 해당하는지 매핑
                word_token_mapping = {}
                current_token_idx = 0
                for word_idx, word in enumerate(words[:max_tokens]):
                    # 단어를 토큰화
                    word_tokens = tokenizer.encode(word, add_special_tokens=False)
                    if word_tokens:
                        # 이 단어의 토큰 인덱스 범위
                        token_indices = list(range(current_token_idx, current_token_idx + len(word_tokens)))
                        word_token_mapping[word_idx] = token_indices
                        current_token_idx += len(word_tokens)
        except Exception as e:
            pass
    
    print(f"\n{'='*80}")
    print(f"단어별 Attention 관계 분석 (배치 {batch_idx}, 헤드 {head_idx})")
    print(f"{'='*80}\n")
    
    # 단어 단위로 그룹화하여 분석 (원본 텍스트 기반)
    if original_words and word_token_mapping:
        # 단어별로 attention 집계
        word_attention = {}
        for word_idx, word in enumerate(original_words[:len(valid_indices)]):
            if word_idx not in word_token_mapping:
                continue
            
            token_indices_for_word = [idx for idx in word_token_mapping[word_idx] if idx in valid_indices]
            if not token_indices_for_word:
                continue
            
            # 이 단어의 모든 토큰에 대한 attention 평균
            word_attn_scores = {}
            for token_idx in token_indices_for_word:
                if token_idx >= len(attn):
                    continue
                scores = attn[token_idx, valid_indices].numpy()
                word_attn_scores[token_idx] = scores
            
            # 각 타겟 단어에 대한 attention 평균 계산
            target_word_scores = {}
            for target_word_idx, target_word in enumerate(original_words[:len(valid_indices)]):
                if target_word_idx not in word_token_mapping:
                    continue
                
                target_token_indices = [idx for idx in word_token_mapping[target_word_idx] if idx in valid_indices]
                if not target_token_indices:
                    continue
                
                # 소스 단어의 모든 토큰 -> 타겟 단어의 모든 토큰 attention 평균
                total_score = 0.0
                count = 0
                for src_token_idx in token_indices_for_word:
                    if src_token_idx >= len(attn):
                        continue
                    for tgt_token_idx in target_token_indices:
                        if tgt_token_idx < len(valid_indices):
                            total_score += attn[src_token_idx, valid_indices.index(tgt_token_idx)].item()
                            count += 1
                
                if count > 0:
                    target_word_scores[target_word_idx] = total_score / count
            
            word_attention[word_idx] = {
                'word': word,
                'scores': target_word_scores
            }
        
        # 단어별 관계 출력
        for word_idx, word_data in word_attention.items():
            word = word_data['word']
            scores = word_data['scores']
            
            # 상위 k개 관계 찾기
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k+1]
            
            print(f"[{word_idx+1:2d}] 단어: '{word}'")
            print(f"     가장 관련 있는 단어들:")
            
            for rank, (target_word_idx, score) in enumerate(sorted_scores):
                if target_word_idx < len(original_words):
                    target_word = original_words[target_word_idx]
                    if target_word_idx == word_idx:
                        print(f"       {rank+1}. 자기 자신: {score:.4f}")
                    else:
                        print(f"       {rank+1}. '{target_word}': {score:.4f}")
            
            print()
    else:
        # 토큰 단위로 분석 (원본 텍스트가 없는 경우)
        for i, token_idx in enumerate(valid_indices):
            token_text = token_texts[token_idx]
            
            # 현재 토큰이 다른 토큰들에게 주는 attention 가중치
            attention_scores = attn[token_idx, valid_indices].numpy()
            
            # 상위 k개 관계 찾기
            top_indices = np.argsort(attention_scores)[::-1][:top_k+1]
            
            print(f"[{i+1:2d}] 토큰: '{token_text}' (인덱스 {token_idx})")
            print(f"     가장 관련 있는 토큰들:")
            
            for rank, top_idx in enumerate(top_indices):
                if top_idx >= len(valid_indices):
                    continue
                target_idx = valid_indices[top_idx]
                target_text = token_texts[target_idx]
                score = attention_scores[top_idx]
                
                if target_idx == token_idx:
                    print(f"       {rank+1}. 자기 자신: {score:.4f}")
                else:
                    print(f"       {rank+1}. '{target_text}' (인덱스 {target_idx}): {score:.4f}")
            
            print()
    
    # 전체 attention matrix 요약
    print(f"\n{'='*80}")
    print(f"Attention Matrix 요약")
    print(f"{'='*80}")
    print(f"평균 attention score: {attn[valid_indices][:, valid_indices].mean().item():.4f}")
    print(f"최대 attention score: {attn[valid_indices][:, valid_indices].max().item():.4f}")
    print(f"최소 attention score: {attn[valid_indices][:, valid_indices].min().item():.4f}")
    
    # 자기 자신에게 주는 attention 평균
    self_attention = torch.diagonal(attn[valid_indices][:, valid_indices]).mean().item()
    print(f"자기 자신에게 주는 attention 평균: {self_attention:.4f}")
    
    # 다른 단어에게 주는 attention 평균
    mask = torch.eye(len(valid_indices), dtype=torch.bool)
    other_attention = attn[valid_indices][:, valid_indices][~mask].mean().item()
    print(f"다른 단어에게 주는 attention 평균: {other_attention:.4f}")
    print(f"{'='*80}\n")


def train_model(
    model: nn.Module,
    train_data: torch.Tensor,
    tokenizer: AutoTokenizer,
    num_epochs: int = 5,
    batch_size: int = 8,
    learning_rate: float = 0.001,
    device: str = 'cpu'
):
    """
    모델을 학습시킵니다.
    작성자: theprismdata@gmail.com
    
    Args:
        model: 학습할 모델
        train_data: 학습 데이터 텐서 (num_samples, seq_len)
        tokenizer: 토크나이저 객체
        num_epochs: 에폭 수
        batch_size: 배치 크기
        learning_rate: 학습률
        device: 사용할 디바이스 ('cpu' 또는 'cuda')
    """
    model = model.to(device)
    model.train()
    
    # 옵티마이저와 손실 함수 설정
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -100)
    
    num_samples = train_data.shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    print(f"\n학습 시작:")
    print(f"  총 샘플 수: {num_samples}")
    print(f"  배치 크기: {batch_size}")
    print(f"  배치 수: {num_batches}")
    print(f"  에폭 수: {num_epochs}")
    print(f"  학습률: {learning_rate}")
    print(f"  디바이스: {device}\n")
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        model.train()
        
        # 데이터 셔플
        indices = torch.randperm(num_samples)
        shuffled_data = train_data[indices]
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            
            # 배치 데이터
            batch_input = shuffled_data[start_idx:end_idx].to(device)
            
            # 토큰 ID가 vocab_size를 초과하지 않도록 클램핑
            vocab_size = model.vocab_size
            batch_input = torch.clamp(batch_input, 0, vocab_size - 1)
            
            # 입력과 타겟 생성 (다음 토큰 예측)
            input_tokens = batch_input[:, :-1]  # 마지막 토큰 제외
            target_tokens = batch_input[:, 1:]   # 첫 번째 토큰 제외
            
            # Forward pass
            optimizer.zero_grad()
            logits, _ = model(input_tokens)
            
            # 손실 계산
            # logits: (batch_size, seq_len-1, vocab_size)
            # target: (batch_size, seq_len-1)
            loss = criterion(logits.reshape(-1, logits.size(-1)), target_tokens.reshape(-1))
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # 진행 상황 출력
            if (batch_idx + 1) % 10 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{num_batches}, Loss: {avg_loss:.4f}")
        
        avg_epoch_loss = total_loss / num_batches
        print(f"\nEpoch {epoch+1}/{num_epochs} 완료 - 평균 Loss: {avg_epoch_loss:.4f}\n")
    
    print("학습 완료!")
    return model


def main():
    """
    메인 실행 함수
    작성자: theprismdata@gmail.com
    """
    # 디바이스 설정
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"사용 디바이스: {device}")
    
    # Qwen2.5 토크나이저 로드
    print("\nQwen2.5 토크나이저 로드 중...")
    tokenizer = load_tokenizer()
    
    # JSON 파일 로드
    json_file_path = 'SL01-00-00.json'
    print(f"\nJSON 파일 로드 중: {json_file_path}")
    data_info = load_json_data(json_file_path)
    print(f"로드된 데이터 항목 수: {len(data_info)}")
    
    # 텍스트 데이터 처리
    print("\n텍스트 데이터 처리 중...")
    token_tensor, vocab_size, original_token_ids = process_text_data(data_info, tokenizer, max_length=256)
    print(f"어휘 크기: {vocab_size}")
    print(f"토큰 텐서 shape: {token_tensor.shape}")
    
    # 모델 파라미터 설정
    d_model = 128
    num_heads = 8
    num_layers = 2  # Self Attention 레이어 수
    
    # 모델 생성
    print(f"\n모델 생성 중...")
    model = TransformerModel(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=0.1
    )
    
    # 모델 파라미터 수 출력
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"총 파라미터 수: {total_params:,}")
    print(f"학습 가능한 파라미터 수: {trainable_params:,}")
    
    # 학습 자동 진행
    print("\n학습을 시작합니다...")
    
    # 학습 데이터 준비 (전체 데이터 사용)
    train_data = token_tensor
    
    # 학습 실행
    model = train_model(
        model=model,
        train_data=train_data,
        tokenizer=tokenizer,
        num_epochs=3,
        batch_size=1,
        learning_rate=0.001,
        device=device
    )
    
    # 학습 후 모델을 평가 모드로 전환
    model.eval()
    
    # 추론 및 분석
    print("\n" + "="*80)
    print("Self Attention 분석 시작")
    print("="*80)
    
    batch_size = min(4, token_tensor.shape[0])
    input_tokens = token_tensor[:batch_size].to(device)
    
    model.eval()
    with torch.no_grad():
        logits, attention_weights_list = model(input_tokens)
        
        # 마지막 레이어의 attention weights 사용
        if attention_weights_list:
            attention_weights = attention_weights_list[-1]  # 마지막 레이어
            print(f"\nAttention weights shape: {attention_weights.shape}")
            
            # Attention weights 통계 출력
            print(f"\nAttention weights 통계:")
            print(f"  평균: {attention_weights.mean().item():.4f}")
            print(f"  표준편차: {attention_weights.std().item():.4f}")
            print(f"  최대값: {attention_weights.max().item():.4f}")
            print(f"  최소값: {attention_weights.min().item():.4f}")
            
            # 각 단어별 관계 분석
            print("\n" + "="*80)
            print("각 단어별 Attention 관계 분석 시작")
            print("="*80)
            
            # 첫 번째 배치의 첫 번째 헤드로 분석
            analyze_word_relationships(
                attention_weights=attention_weights,
                token_ids=input_tokens.cpu(),
                tokenizer=tokenizer,
                data_info=data_info,
                batch_idx=0,
                head_idx=0,
                top_k=5,
                max_tokens=20
            )
            
            # 여러 헤드의 평균으로도 분석
            print("\n" + "="*80)
            print("모든 헤드의 평균 Attention 관계 분석")
            print("="*80)
            
            # 모든 헤드의 평균 계산
            avg_attention = attention_weights.mean(dim=1)  # (batch_size, seq_len, seq_len)
            
            analyze_word_relationships(
                attention_weights=avg_attention.unsqueeze(1),  # (batch_size, 1, seq_len, seq_len)로 변환
                token_ids=input_tokens.cpu(),
                tokenizer=tokenizer,
                data_info=data_info,
                batch_idx=0,
                head_idx=0,
                top_k=5,
                max_tokens=20
            )


if __name__ == "__main__":
    main()

