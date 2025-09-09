"""
Self-Attention 機制實作
展示 Transformer 架構中的核心注意力計算
"""

import torch
import math


def attention(Q, K, V):
    """
    Self-Attention 計算流程
    
    Args:
        Q (Query): 查詢向量
        K (Key): 鍵向量  
        V (Value): 值向量
    
    Returns:
        output: 注意力計算結果
    """
    # 獲取鍵向量的維度
    d_k = K.shape[-1]
    
    # 計算注意力分數
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    # 應用 softmax 獲得注意力權重
    attention_weights = torch.softmax(scores, dim=-1)
    
    # 計算加權的值向量
    output = torch.matmul(attention_weights, V)
    
    return output


if __name__ == "__main__":
    # 測試範例
    batch_size = 2
    seq_length = 4
    d_model = 8
    
    # 隨機初始化 Q, K, V
    Q = torch.randn(batch_size, seq_length, d_model)
    K = torch.randn(batch_size, seq_length, d_model)
    V = torch.randn(batch_size, seq_length, d_model)
    
    # 計算注意力
    output = attention(Q, K, V)
    
    print(f"輸入形狀:")
    print(f"  Q: {Q.shape}")
    print(f"  K: {K.shape}")
    print(f"  V: {V.shape}")
    print(f"輸出形狀: {output.shape}")