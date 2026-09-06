import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import SimpleAttention

class DialogueRNNCell(nn.Module):
    """
    One timestep of DialogueRNN: updates global state, party state (via
    attention over past global states), and emotion state — Eqs. 1, 2-4, 5, 8.
    """

    def __init__(self, D_m, D_g, D_p, D_e):
        super().__init__()
        self.D_m, self.D_g, self.D_p, self.D_e = D_m, D_g, D_p, D_e

        # Global GRU: input = utterance (D_m) + prev speaker state (D_p)
        self.global_gru = nn.GRUCell(D_m + D_p, D_g)

        # Party GRU: input = utterance (D_m) + attended context (D_g)
        self.party_gru = nn.GRUCell(D_m + D_g, D_p)

        # Emotion GRU: input = current speaker state (D_p)
        self.emotion_gru = nn.GRUCell(D_p, D_e)

        self.attention = SimpleAttention(D_g, D_m)

    def forward(self, u_t, speaker_onehot, g_hist, q_prev, e_prev):
        """
        u_t:           (batch, D_m)          current utterance features
        speaker_onehot:(batch, num_speakers) who's speaking this turn
        g_hist:        (t-1, batch, D_g)     all past global states (or None if t==1)
        q_prev:        (batch, num_speakers, D_p)  party states for every speaker
        e_prev:        (batch, D_e)          previous emotion state
        """
        batch_size = u_t.size(0)
        num_speakers = speaker_onehot.size(1)

        # --- get the current speaker's previous party state ---
        speaker_idx = speaker_onehot.argmax(dim=1)                      # (batch,)
        q_prev_speaker = q_prev[torch.arange(batch_size), speaker_idx]  # (batch, D_p)

        # --- Eq. 1: update global state ---
        g_input = torch.cat([u_t, q_prev_speaker], dim=1)
        g_t = self.global_gru(g_input, g_hist[-1] if g_hist is not None
                               else torch.zeros(batch_size, self.D_g, device=u_t.device))

        # --- Eq. 2-4: attention over past global states -> context c_t ---
        if g_hist is None or g_hist.size(0) == 0:
            c_t = torch.zeros(batch_size, self.D_g, device=u_t.device)
        else:
            c_t, _ = self.attention(g_hist, u_t)

        # --- Eq. 5: update speaker's party state ---
        p_input = torch.cat([u_t, c_t], dim=1)
        q_new_speaker = self.party_gru(p_input, q_prev_speaker)

        # write the updated state back for just this speaker, leave others untouched
        q_t = q_prev.clone()
        q_t[torch.arange(batch_size), speaker_idx] = q_new_speaker

        # --- Eq. 8: update emotion state ---
        e_t = self.emotion_gru(q_new_speaker, e_prev)

        return g_t, q_t, e_t


class DialogueRNN(nn.Module):
    def __init__(self, D_m=100, D_g=150, D_p=150, D_e=100, num_speakers=2, num_classes=6):
        super().__init__()
        self.D_g, self.D_p, self.D_e = D_g, D_p, D_e
        self.num_speakers = num_speakers

        self.cell = DialogueRNNCell(D_m, D_g, D_p, D_e)

        # Eqs. 9-11: classifier head
        self.linear = nn.Linear(D_e, D_e)
        self.smax = nn.Linear(D_e, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, U, speakers):
        """
        U:        (seq_len, batch, D_m)
        speakers: (seq_len, batch, num_speakers)
        returns:  (seq_len, batch, num_classes) log-probabilities
        """
        seq_len, batch_size, _ = U.size()
        device = U.device

        g_hist = None
        q_t = torch.zeros(batch_size, self.num_speakers, self.D_p, device=device)
        e_t = torch.zeros(batch_size, self.D_e, device=device)

        outputs = []
        for t in range(seq_len):
            g_t, q_t, e_t = self.cell(U[t], speakers[t], g_hist, q_t, e_t)

            g_hist = g_t.unsqueeze(0) if g_hist is None else torch.cat([g_hist, g_t.unsqueeze(0)], dim=0)

            l_t = F.relu(self.linear(e_t))
            l_t = self.dropout(l_t)
            log_probs = F.log_softmax(self.smax(l_t), dim=1)
            outputs.append(log_probs.unsqueeze(0))

        return torch.cat(outputs, dim=0)   # (seq_len, batch, num_classes)