import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleAttention(nn.Module):
    """
    Implements Eq. 2-4 of the paper:
        alpha = softmax(u_t^T W_alpha [g_1, ..., g_{t-1}])
        c_t   = alpha [g_1, ..., g_{t-1}]^T

    i.e. score each past global state against the current utterance,
    softmax those scores, then take a weighted sum of past global states.
    """

    def __init__(self, mem_dim, cand_dim):
        super().__init__()
        self.transform = nn.Linear(cand_dim, mem_dim, bias=False)

    def forward(self, M, x):
        """
        M: (seq_len, batch, mem_dim)   -- all past global states g_1..g_{t-1}
        x: (batch, cand_dim)            -- current utterance u_t
        returns: context c_t (batch, mem_dim), attention weights (batch, seq_len)
        """
        M_ = M.permute(1, 2, 0)                     # (batch, mem_dim, seq_len)
        x_ = self.transform(x).unsqueeze(1)         # (batch, 1, mem_dim)
        scores = torch.bmm(x_, M_)                  # (batch, 1, seq_len)
        alpha = F.softmax(scores, dim=2)            # (batch, 1, seq_len)
        attn_pool = torch.bmm(alpha, M.transpose(0, 1))  # (batch, 1, mem_dim)
        return attn_pool.squeeze(1), alpha.squeeze(1)