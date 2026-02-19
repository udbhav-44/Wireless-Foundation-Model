import torch
import torch.nn as nn
import torch.nn.functional as F

class AxialLinearAttention(nn.Module):
    """
    Axial Linear Attention module.
    
    Splits the sequence into a 2D grid (Frequency x Antennas) and applies
    Linear Attention (with ELU+1 kernel) along each axis sequentially.
    """
    def __init__(self, d_model, n_heads, d_k=None, antennas=32, freq_groups=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.antennas = antennas
        self.freq_groups = freq_groups
        self.d_k = d_k if d_k is not None else d_model // n_heads
        self.inner_dim = self.d_k * n_heads
        self.dropout_val = dropout
        
        # Attention weights for Antenna Axis
        self.ant_q = nn.Linear(d_model, self.inner_dim)
        self.ant_k = nn.Linear(d_model, self.inner_dim)
        self.ant_v = nn.Linear(d_model, self.inner_dim)
        self.ant_out = nn.Linear(self.inner_dim, d_model)
        
        # Attention weights for Frequency Axis
        self.freq_q = nn.Linear(d_model, self.inner_dim)
        self.freq_k = nn.Linear(d_model, self.inner_dim)
        self.freq_v = nn.Linear(d_model, self.inner_dim)
        self.freq_out = nn.Linear(self.inner_dim, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def _elu_plus_one(self, x):
        return F.elu(x) + 1.0

    def _linear_attention(self, q_proj, k_proj, v_proj, x_in, axis_len):
        """
        Applies linear attention along the axis of length `axis_len`.
        Input x_in shape: (Batch * Other_Dim, Axis_Len, D_Model)
        """
        batch_size_aug = x_in.size(0)
        
        # Linear projections
        Q = q_proj(x_in).view(batch_size_aug, axis_len, self.n_heads, self.d_k)
        K = k_proj(x_in).view(batch_size_aug, axis_len, self.n_heads, self.d_k)
        V = v_proj(x_in).view(batch_size_aug, axis_len, self.n_heads, self.d_k)
        
        # Apply kernel feature map phi(x) = ELU(x) + 1
        Q = self._elu_plus_one(Q)
        K = self._elu_plus_one(K)
        
        # Linear Attention: Q (K^T V)
        # K shape: (B, L, H, Dk)
        # V shape: (B, L, H, Dk) (Assume Dv = Dk)
        
        # Compute KV state: sum_{L} (K_{b,l,h,dk} * V_{b,l,h,dv}) -> (B, H, Dk, Dv)
        # Einsum: 'blhd,blhe->bhde' (d=dk, e=dv)
        KV = torch.einsum('blhd,blhe->bhde', K, V)
        
        # Compute Output: Q * KV
        # Q shape: (B, L, H, Dk)
        # KV shape: (B, H, Dk, Dv)
        # Einsum: 'blhd,bhde->blhe'
        out = torch.einsum('blhd,bhde->blhe', Q, KV)
        
        # Reshape back: (B, L, n_heads * Dk) -> (B, L, inner_dim)
        out = out.reshape(batch_size_aug, axis_len, self.inner_dim)
        
        # Normalization over L is not standard here as per attention paper, but scaling helps.
        # "Transformers are RNNs" suggests normalizer in denominator if needed, 
        # but pure linear attention Q(K^T V) is often sufficient with normalization layers outside.
        # We'll stick to the raw projection.
        
        return out

    def forward(self, x, mask=None):
        """
        x: (Batch, Seq_Len, D_Model)
        mask: Ignored for now (Axial Linear Attention typically global or causal, assuming global here)
        """
        B, L, D = x.shape
        
        # 1. Handle CLS Token
        # Assuming CLS token is at index 0. We separate it.
        # If explicit CLS handling is not desired, we can treat it as part of the grid if L fits perfectly.
        # However, L=129 and Grid=128 (32x4). So we separate CLS.
        
        use_cls = False
        if L == (self.antennas * self.freq_groups) + 1:
            cls_token = x[:, 0:1, :]
            grid = x[:, 1:, :]
            use_cls = True
        elif L == (self.antennas * self.freq_groups):
            grid = x
        else:
            raise ValueError(f"Sequence length {L} does not match grid dimensions {self.antennas}x{self.freq_groups} (+1 CLS optional).")
            
        # Grid shape: (B, P, D) -> (B, F, A, D)
        grid = grid.reshape(B, self.freq_groups, self.antennas, D)
        
        # --- AXIS 1: ANTENNAS ---
        grid_ant = grid.reshape(B * self.freq_groups, self.antennas, D)
        
        ant_out = self._linear_attention(self.ant_q, self.ant_k, self.ant_v, grid_ant, self.antennas)
        ant_out = self.ant_out(ant_out)
        ant_out = self.dropout(ant_out)
        
        # Residual connection (if desired inside the block, or outside). 
        # Standard Transformer has Add&Norm outside, but Axial usually does it per axis.
        # We will Add & Norm here to be safe and consistent with "Axial Attention" papers.
        grid_ant = grid_ant + ant_out
        # However, we don't have a specific Norm layer per axis in the init. 
        # Let's check LWM structure. LWM EncoderLayer does `norm(attn(x))`.
        # If we return just the attention output, the outer layer will norm it.
        # BUT, since we do TWO attentions, we should probably compose them.
        # "Axial Attention in Multidimensional Transformers" applies layers sequentially.
        # Layer 1: Axial-Width, Layer 2: Axial-Height.
        # Here we do both in one "Replacement" block.
        # So we should update `grid` after Antenna attention.
        
        grid = grid_ant.reshape(B, self.freq_groups, self.antennas, D)
        
        # --- AXIS 2: FREQUENCY ---
        # We want to attention over 'F' dimension.
        # Permute to put F in the middle: (B, A, F, D)
        grid = grid.permute(0, 2, 1, 3).contiguous()
        # Shape: (B * A, F, D)
        grid_freq = grid.reshape(B * self.antennas, self.freq_groups, D)
        
        freq_out = self._linear_attention(self.freq_q, self.freq_k, self.freq_v, grid_freq, self.freq_groups)
        freq_out = self.freq_out(freq_out)
        freq_out = self.dropout(freq_out)
        
        grid_freq = grid_freq + freq_out
        grid = grid_freq.reshape(B, self.antennas, self.freq_groups, D)
        
        # Permute back to (B, F, A, D)
        grid = grid.permute(0, 2, 1, 3).contiguous()
        
        # 3. Flatted back
        grid_flat = grid.reshape(B, -1, D)
        
        if use_cls:
            # Reattach CLS
            # Note: CLS was NOT updated by this attention. 
            # In standard transformers, CLS attends to everyone. 
            # With Axial, if we ignore CLS, it loses context.
            # SIMPLE FIX: Let CLS pass through identity, or (better)
            # Maybe we should assume the user handles Global Avg Pooling later?
            # LWM model ends with `h_masked = self.norm(F.relu(self.linear(h_masked)))` 
            # `h_masked` comes from `torch.gather(output, 1, masked_pos)`.
            # So CLS might not even be used for the final prediction if masked_pos points to grid!
            # Let's check LWM forward: `output = self.embedding(input_ids)`.
            # `masked_pos` gathers from output.
            # So as long as the grid is updated, we are good.
            out = torch.cat([cls_token, grid_flat], dim=1)
        else:
            out = grid_flat
            
        # The outer EncoderLayer expects (output, attn_scores).
        # Linear attention doesn't produce N^2 scores easily.
        # We return None for scores.
        return out, None
