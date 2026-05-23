# src/model/attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SingleHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, head_dim: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        self.head_dim = head_dim
        
        # 1. The Q, K, V Projections
        # These linear layers learn HOW to ask questions, HOW to advertise keys, 
        # and WHAT values to share.
        self.query_proj = nn.Linear(embed_dim, head_dim, bias=False)
        self.key_proj   = nn.Linear(embed_dim, head_dim, bias=False)
        self.value_proj = nn.Linear(embed_dim, head_dim, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        # 2. The Causal Mask (Lower Triangular Matrix)
        # register_buffer saves this grid in the model's memory without making it a trainable weight.
        # It looks like this (for block_size=4):
        # [[1, 0, 0, 0],  <- Token 0 can only see Token 0
        #  [1, 1, 0, 0],  <- Token 1 can see 0 and 1
        #  [1, 1, 1, 0],  <- Token 2 can see 0, 1, 2
        #  [1, 1, 1, 1]]  <- Token 3 can see everyone
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        
        # Step 1: Project the input into Q, K, V
        q = self.query_proj(x)  # (B, T, head_dim)
        k = self.key_proj(x)    # (B, T, head_dim)
        v = self.value_proj(x)  # (B, T, head_dim)
        
        # Step 2: Calculate the raw Attention Scores (Q dot K)
        # We transpose the last two dimensions of K to make the matrix multiplication work
        scores = q @ k.transpose(-2, -1) # (B, T, T)
        
        # Step 3: Scale the scores
        # If we don't divide by the square root of the head_dim, the dot products 
        # get too massive and break the Softmax function later.
        scores = scores * (1.0 / math.sqrt(self.head_dim))
        
        # Step 4: Apply the Causal Mask
        # Find all the '0's in our triangle mask and replace the score with -infinity.
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        
        # Step 5: Softmax converts the scores into percentages (weights) that sum to 1.
        # e-to-the-power-of(-inf) is 0. So future tokens get 0% attention.
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        
        # Step 6: Multiply the weights by the Values
        output = weights @ v # (B, T, head_dim)
        
        return output

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Helper function to visualize who is looking at who."""
        B, T, C = x.shape
        q, k = self.query_proj(x), self.key_proj(x)
        scores = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(self.head_dim))
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        return F.softmax(scores, dim=-1)


# --- QUICK TEST BLOCK ---
if __name__ == '__main__':
    # 1. Setup mock dimensions
    batch_size = 1
    block_size = 4  # e.g., the 4 letters of K-i-n-g
    embed_dim = 128
    head_dim = 128  # In single-head, this equals embed_dim
    
    # 2. Initialize the Attention Head
    attention_head = SingleHeadAttention(embed_dim, head_dim, block_size, dropout=0.0)
    
    # 3. Create dummy embedding vectors
    dummy_x = torch.randn(batch_size, block_size, embed_dim)
    
    # 4. Run the forward pass
    output = attention_head(dummy_x)
    print(f"Input Shape: {dummy_x.shape}")
    print(f"Output Shape: {output.shape}")
    
    # 5. Visualize the "No Cheating" Mask
    print("\nAttention Weights (Who is looking at who):")
    weights = attention_head.get_attention_weights(dummy_x)
    
    # Print the 4x4 grid of percentages for the first batch
    for row in weights[0]:
        formatted_row = [f"{val:.2f}" for val in row.tolist()]
        print(formatted_row)
        
    print("\nNotice the upper right triangle is all 0.00! The future is hidden.")