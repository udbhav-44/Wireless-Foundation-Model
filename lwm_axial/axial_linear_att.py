import torch
import torch.nn as nn
import torch.nn.functional as F

class AxialSoftmaxAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, antennas=32, freq_groups=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k if d_k is not None else d_model // n_heads
        self.inner_dim = self.d_k * n_heads
        
        self.antennas = antennas
        self.freq_groups = freq_groups
        self.dropout_p = dropout

        # --- Antigen (Spatial) Attention Projections ---
        self.w_q_ant = nn.Linear(d_model, self.inner_dim)
        self.w_k_ant = nn.Linear(d_model, self.inner_dim)
        self.w_v_ant = nn.Linear(d_model, self.inner_dim)
        
        # --- Frequency (Spectral) Attention Projections ---
        # Input to freq attention is the output of antenna attention, which has dim = inner_dim
        self.w_q_freq = nn.Linear(self.inner_dim, self.inner_dim)
        self.w_k_freq = nn.Linear(self.inner_dim, self.inner_dim)
        self.w_v_freq = nn.Linear(self.inner_dim, self.inner_dim)

        self.out_proj = nn.Linear(self.inner_dim, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize
        nn.init.xavier_uniform_(self.w_q_ant.weight)
        nn.init.xavier_uniform_(self.w_k_ant.weight)
        nn.init.xavier_uniform_(self.w_v_ant.weight)
        nn.init.xavier_uniform_(self.w_q_freq.weight)
        nn.init.xavier_uniform_(self.w_k_freq.weight)
        nn.init.xavier_uniform_(self.w_v_freq.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)


    def forward(self, x, mask=None):
        """
        x: [Batch, SeqLen, D_Model]
        SeqLen should be (antennas * freq_groups) + CLS
        Expects Pre-Norm input (normalized).
        Returns: Attention Output (without residual added).
        """
        B, P, D = x.shape
        
        # 1. Separate CLS token if present (Index 0)
        cls_token = None
        grid = x
        
        if P > (self.antennas * self.freq_groups):
            cls_token = x[:, 0:1, :] # [B, 1, D]
            grid = x[:, 1:, :]       # [B, A*F, D]
            
        # 2. Reshape to [Batch, Ant, Freq, D]
        # Since torch_pipeline now guarantees [Ant0_F0, Ant0_F1...],
        # we can simply reshape without permutation hacks.
        grid = grid.reshape(B, self.antennas, self.freq_groups, D)

        # ---------------------------------------------------------
        # AXIS 1: ANTENNA (SPATIAL) ATTENTION
        # Input: [B, A, F, D]
        # Treat Freq as Batch dimension -> [B*F, A, D]
        # ---------------------------------------------------------
        
        # Transpose to put Freq in batch dim: [B, F, A, D]
        x_ant = grid.permute(0, 2, 1, 3).contiguous().view(B * self.freq_groups, self.antennas, D)
        
        q_ant = self.w_q_ant(x_ant).view(B * self.freq_groups, self.antennas, self.n_heads, self.d_k).transpose(1, 2)
        k_ant = self.w_k_ant(x_ant).view(B * self.freq_groups, self.antennas, self.n_heads, self.d_k).transpose(1, 2)
        v_ant = self.w_v_ant(x_ant).view(B * self.freq_groups, self.antennas, self.n_heads, self.d_k).transpose(1, 2)
        
        out_ant = F.scaled_dot_product_attention(q_ant, k_ant, v_ant, dropout_p=self.dropout_p if self.training else 0.0)
        
        # Reshape back: [B*F, A, inner_dim] -> [B, F, A, D] -> [B, A, F, D]
        out_ant = out_ant.transpose(1, 2).contiguous().view(B * self.freq_groups, self.antennas, self.inner_dim)
        x_freq_in = out_ant.view(B, self.freq_groups, self.antennas, self.inner_dim).permute(0, 2, 1, 3)

        # ---------------------------------------------------------
        # AXIS 2: FREQUENCY (SPECTRAL) ATTENTION
        # Input: [B, A, F, D]
        # Treat Ant as Batch dimension -> [B*A, F, D]
        # ---------------------------------------------------------
        x_freq = x_freq_in.contiguous().reshape(B * self.antennas, self.freq_groups, self.inner_dim)
        
        q_freq = self.w_q_freq(x_freq).view(B * self.antennas, self.freq_groups, self.n_heads, self.d_k).transpose(1, 2)
        k_freq = self.w_k_freq(x_freq).view(B * self.antennas, self.freq_groups, self.n_heads, self.d_k).transpose(1, 2)
        v_freq = self.w_v_freq(x_freq).view(B * self.antennas, self.freq_groups, self.n_heads, self.d_k).transpose(1, 2)
        
        out_freq = F.scaled_dot_product_attention(q_freq, k_freq, v_freq, dropout_p=self.dropout_p if self.training else 0.0)
        
        # Reshape back: [B*A, F, inner_dim] -> [B, A, F, D]
        out_freq = out_freq.transpose(1, 2).contiguous().view(B * self.antennas, self.freq_groups, self.inner_dim)
        grid_out = out_freq.view(B, self.antennas, self.freq_groups, self.inner_dim)
        
        # ---------------------------------------------------------
        # Final Projection & Flatten
        # [B, A*F, D]
        # ---------------------------------------------------------
        grid_flat = grid_out.reshape(B, self.antennas * self.freq_groups, self.inner_dim)
        grid_final = self.out_proj(grid_flat)
        grid_final = self.dropout(grid_final)
        
        # ---------------------------------------------------------
        # CLS Mixing (Lightweight Global Context)
        # ---------------------------------------------------------
        if cls_token is not None:
            # 1. Update CLS with Grid Mean
            grid_mean = grid_final.mean(dim=1, keepdim=True) # [B, 1, D]
            cls_out = cls_token + grid_mean
            
            # 2. Update Grid with UPDATED CLS (Broadcast)
            # This ensures grid sees the global context
            grid_final = grid_final + cls_out
            
            # Reattach
            out = torch.cat([cls_out, grid_final], dim=1)
        else:
            out = grid_final
            
        return out, None # match signature
