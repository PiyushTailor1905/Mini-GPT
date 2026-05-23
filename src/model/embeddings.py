# src/model/embeddings.py

import torch
import torch.nn as nn

class GPTEmbedding(nn.Module):
    def __init__(self, vocab_size: int, block_size: int, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        
        # 1. Token Embedding Table
        # A lookup matrix of shape (vocab_size, embed_dim)
        # e.g., 65 rows (one for each character), 128 columns (the vector coordinates)
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # 2. Positional Embedding Table
        # A lookup matrix of shape (block_size, embed_dim)
        # One unique vector for every position in our context window
        self.position_embedding = nn.Embedding(block_size, embed_dim)
        
        # Dropout randomly zeroes out some data during training to prevent overfitting
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Takes integer IDs and returns their rich vector representations.
        Input shape:  (Batch, Time)
        Output shape: (Batch, Time, Channels) -> where Channels = embed_dim
        """
        B, T = token_ids.shape
        
        # Look up the vector for the tokens
        # Output shape: (B, T, embed_dim)
        tok_emb = self.token_embedding(token_ids)
        
        # Generate position indices: [0, 1, 2, ..., T-1]
        # We must create this on the exact same hardware device (CPU/GPU) as the token_ids
        positions = torch.arange(T, device=token_ids.device)
        
        # Look up the vectors for the positions
        # Output shape: (T, embed_dim)
        pos_emb = self.position_embedding(positions)
        
        # Combine them via addition. 
        # PyTorch automatically broadcasts (T, embed_dim) across the Batch dimension
        # Output shape: (B, T, embed_dim)
        x = tok_emb + pos_emb
        
        return self.dropout(x)

# --- QUICK TEST BLOCK ---
if __name__ == '__main__':
    # 1. Setup mock dimensions
    vocab_size = 65    # Characters in Tiny Shakespeare
    block_size = 8     # Context window size
    embed_dim = 128    # Number of coordinates per vector
    batch_size = 4     # Sequences processed at once
    
    # 2. Initialize the Embedding Layer
    embedding_layer = GPTEmbedding(vocab_size, block_size, embed_dim, dropout=0.0)
    
    # 3. Create dummy integer input (simulating output from our Data Loader)
    # Shape: (4 batches, 8 tokens)
    dummy_input = torch.randint(0, vocab_size, (batch_size, block_size))
    
    # 4. Run the forward pass
    output = embedding_layer(dummy_input)
    
    print(f"Input Shape (B, T): {dummy_input.shape} - dtype: {dummy_input.dtype}")
    print(f"Output Shape (B, T, C): {output.shape} - dtype: {output.dtype}")
    
    # Verify the shapes match our math expectations
    assert output.shape == (batch_size, block_size, embed_dim), "ERROR: Output dimensions are wrong!"
    print("\nSUCCESS: The integers have been converted to high-dimensional vectors.")