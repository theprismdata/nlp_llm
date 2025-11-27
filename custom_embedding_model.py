"""
Custom 토크나이저 기반 임베딩 모델 정의
작성자: theprismdata@gmail.com
"""

import torch
import torch.nn as nn
from typing import Optional


class CustomTokenizerEmbeddingModel(nn.Module):
    """
    CustomBPETokenizer로부터 임베딩을 생성하는 단순 모델
    작성자: theprismdata@gmail.com
    """

    def __init__(self, vocab_size: int, d_model: int = 3072):
        """
        Custom 토크나이저 임베딩 모델 초기화
        작성자: theprismdata@gmail.com

        Args:
            vocab_size: 어휘 크기
            d_model: 임베딩 차원
        """
        super(CustomTokenizerEmbeddingModel, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        입력 토큰 ID로부터 문장 임베딩을 계산합니다.
        작성자: theprismdata@gmail.com

        Args:
            input_ids: 토큰 ID 텐서 (batch_size, seq_len)
            attention_mask: 어텐션 마스크 (batch_size, seq_len) – 1은 유효 토큰, 0은 패딩

        Returns:
            문장 임베딩 텐서 (batch_size, d_model)
        """
        # 토큰 ID가 어휘 범위를 벗어나지 않도록 클램핑
        input_ids = torch.clamp(input_ids, 0, self.vocab_size - 1)

        # 토큰 임베딩 계산
        token_embeds = self.embedding(input_ids)  # (batch_size, seq_len, d_model)

        # 마스크 기반 mean pooling
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()  # (batch_size, seq_len, 1)
            token_embeds = token_embeds * mask
            summed = token_embeds.sum(dim=1)  # (batch_size, d_model)
            denom = mask.sum(dim=1).clamp(min=1e-6)  # (batch_size, 1)
            sentence_embeds = summed / denom
        else:
            # 마스크가 없으면 단순 평균
            sentence_embeds = token_embeds.mean(dim=1)

        return sentence_embeds


