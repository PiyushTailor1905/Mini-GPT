# src/model/transformer_block.py

import torch
import torch.nn as nn
from src.model.multi_head_attention import MultiHeadAttention

class FeedForwardNetwork(nn.Module):
    def __init__(self, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        # The 4x Expansion:
        # We temporarily expand the 128-dimension vector to 512 dimensions.
        # This gives the network a massive "workspace" to compute complex features
        # before projecting it back down to 128 dimensions.
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),                            # A smooth, modern alternative to ReLU
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Applied to every single token independently
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        # 1. The Shock Absorbers
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        
        # 2. The Board Meeting (Communication)
        self.attention = MultiHeadAttention(embed_dim, num_heads, block_size, dropout)
        
        # 3. The Private Office (Computation)
        self.ffn = FeedForwardNetwork(embed_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: Communication
        # Apply LayerNorm, run Attention, and ADD the result back to the original 'x'
        x = x + self.attention(self.ln1(x))
        
        # Step 2: Computation
        # Apply LayerNorm, run FFN, and ADD the result back to 'x' again
        x = x + self.ffn(self.ln2(x))
        
        return x

# --- QUICK TEST BLOCK ---
if __name__ == '__main__':
    # 1. Setup mock dimensions
    batch_size = 2
    block_size = 8  
    embed_dim = 128
    num_heads = 4
    
    # 2. Initialize the full Transformer Block
    block = TransformerBlock(embed_dim, num_heads, block_size, dropout=0.0)
    
    # 3. Create dummy embedding vectors
    dummy_x = torch.randn(batch_size, block_size, embed_dim)
    
    # 4. Run the forward pass
    output = block(dummy_x)
    
    print(f"Input Shape:  {dummy_x.shape}")
    print(f"Output Shape: {output.shape}")
    
    assert output.shape == dummy_x.shape, "ERROR: Transformer Block changed the dimensions!"
    print("\nSUCCESS: The Transformer Block processed the data perfectly.")