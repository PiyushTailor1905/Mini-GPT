# src/model/multi_head_attention.py

import torch
import torch.nn as nn
from src.model.attention import SingleHeadAttention

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        
        # Ensure our math divides perfectly (e.g., 128 / 4 = 32)
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.head_dim = embed_dim // num_heads
        
        # 1. The Team of Specialists
        # We use nn.ModuleList to hold our heads so PyTorch knows they contain trainable weights.
        self.heads = nn.ModuleList([
            SingleHeadAttention(embed_dim, self.head_dim, block_size, dropout)
            for _ in range(num_heads)
        ])
        
        # 2. The Final Mixer (Output Projection)
        # After we glue the 4 heads back together, we need them to talk to each other.
        # This linear layer mixes the grammatical findings of Head 1 with the noun findings of Head 2.
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: Run all heads in parallel
        # Each head returns a tensor of shape (B, T, head_dim) -> e.g., (1, 4, 32)
        head_outputs = [head(x) for head in self.heads]
        
        # Step 2: Glue them back together along the last dimension
        # Four 32-dim tensors become one 128-dim tensor: (B, T, embed_dim)
        concatenated = torch.cat(head_outputs, dim=-1)
        
        # Step 3: Mix the results
        output = self.output_proj(concatenated)
        
        return self.dropout(output)

# --- QUICK TEST BLOCK ---
if __name__ == '__main__':
    # 1. Setup mock dimensions
    batch_size = 2
    block_size = 8  
    embed_dim = 128
    num_heads = 4
    
    # 2. Initialize Multi-Head Attention
    mha = MultiHeadAttention(embed_dim, num_heads, block_size, dropout=0.0)
    
    # 3. Create dummy embedding vectors
    dummy_x = torch.randn(batch_size, block_size, embed_dim)
    
    # 4. Run the forward pass
    output = mha(dummy_x)
    
    print(f"Input Shape:  {dummy_x.shape}")
    print(f"Output Shape: {output.shape}")
    
    # 5. Math check
    head_dim = embed_dim // num_heads
    print(f"\nMath Check:")
    print(f"Total embed_dim: {embed_dim}")
    print(f"Number of heads: {num_heads}")
    print(f"Dimensions per head: {head_dim} (Since {num_heads} * {head_dim} = {embed_dim})")
    
    assert output.shape == dummy_x.shape, "ERROR: Output shape must exactly match input shape!"
    print("\nSUCCESS: Multi-Head Attention is fully assembled.")