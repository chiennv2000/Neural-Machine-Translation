import torch
import torch.nn as nn
import torch.nn.functional as F
"""
(tgt_len, b) = (128, 32)
h = 100
vocab = 3000

P = torch.randn(tgt_len-1, b, vocab)
target_padded = torch.randint(0, 1, (tgt_len, b)).unsqueeze(-1)

target_gold_words_log_prob = torch.gather(P, index=target_padded[1: ], dim=-1).squeeze(-1)
print(target_gold_words_log_prob.sum(dim=0))
"""
P = torch.tensor([[[0.1, 0.9]], [[0.8, 0.2]], [[0.6, 0.4]]])    #(3, 1, 2)
target_padded = torch.tensor([[0], [1], [1]], dtype=torch.long)                #(3, 1)

print(torch.gather(P, index=target_padded.unsqueeze(-1), dim=-1))