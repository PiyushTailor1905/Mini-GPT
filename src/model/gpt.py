# src/model/gpt.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.embeddings import GPTEmbedding
from src.model.transformer_block import TransformerBlock

class MiniGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        block_size: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.block_size = block_size
        
        # 1. The Fuel Intake (Embeddings)
        self.embedding = GPTEmbedding(vocab_size, block_size, embed_dim, dropout)
        
        # 2. The Engine Cylinders (Transformer Blocks)
        # nn.Sequential automatically chains the blocks together so the output of Block 1 
        # feeds directly into Block 2, and so on.
        self.transformer = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, block_size, dropout)
            for _ in range(num_layers)
        ])
        
        # 3. Final Polish
        # One last layer normalization before we make our final prediction
        self.final_ln = nn.LayerNorm(embed_dim)
        
        # 4. The Drivetrain (Language Modeling Head)
        # Converts the 128-dimensional vectors back into 65 "scores" (logits)
        # One score for every possible character in our vocabulary.
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        
        # --- THE WEIGHT TYING TRICK ---
        self.lm_head.weight = self.embedding.token_embedding.weight
        
        # Apply the GPT-2 Initialization trick
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Sets the starting weights to the Goldilocks zone (std=0.02)"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, token_ids: torch.Tensor, targets: torch.Tensor = None):
        """
        The main pipeline. If 'targets' are provided, it automatically calculates
        how wrong the model is (the loss).
        """
        B, T = token_ids.shape
        
        # Push data through the pipeline
        x = self.embedding(token_ids)     # (B, T, C)
        x = self.transformer(x)           # (B, T, C)
        x = self.final_ln(x)              # (B, T, C)
        logits = self.lm_head(x)          # (B, T, vocab_size)
        
        loss = None
        if targets is not None:
            # We have to flatten the grids to use PyTorch's cross_entropy loss
            B, T, V = logits.shape
            logits_flat = logits.view(B * T, V)
            targets_flat = targets.view(B * T)
            
            # Cross Entropy measures the difference between our predictions and the actual answer
            loss = F.cross_entropy(logits_flat, targets_flat)
            
        return logits, loss

    def count_parameters(self) -> int:
        """Helper tool to see how massive our brain is."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0):
        """
        The actual Text Generator! It predicts one token, glues it to the end of the 
        input, and repeats the process.
        """
        self.eval() # Disable dropout during generation
        
        for _ in range(max_new_tokens):
            # Only look at the last 'block_size' tokens so we don't overflow the memory
            idx_cond = idx[:, -self.block_size:]
            
            # Get predictions
            logits, _ = self(idx_cond)
            
            # We only care about the very last token's prediction (the newest one)
            logits = logits[:, -1, :] 
            
            # Temperature controls creativity. 
            # High temp = more chaotic/random. Low temp = very predictable/safe.
            logits = logits / temperature
            
            # Convert raw scores into percentages
            probs = F.softmax(logits, dim=-1)
            
            # Roll a weighted die to pick the next character based on those percentages
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Glue the new character to our running sequence
            idx = torch.cat([idx, next_token], dim=1)
            
        return idx

# --- QUICK TEST BLOCK ---
if __name__ == '__main__':
    # 1. Setup our model specs
    vocab_size = 65
    embed_dim = 128
    num_heads = 4
    num_layers = 4  # We are stacking 4 transformer blocks!
    block_size = 64
    batch_size = 2
    
    # 2. Initialize the full GPT
    model = MiniGPT(vocab_size, embed_dim, num_heads, num_layers, block_size)
    print(f"Mini-GPT Parameter Count: {model.count_parameters():,}")
    
    # 3. Create dummy input and targets
    dummy_x = torch.randint(0, vocab_size, (batch_size, block_size))
    dummy_y = torch.randint(0, vocab_size, (batch_size, block_size))
    
    # 4. Run a forward pass to calculate loss
    logits, loss = model(dummy_x, dummy_y)
    
    print(f"\nInput Shape: {dummy_x.shape}")
    print(f"Logits (Predictions) Shape: {logits.shape}")
    print(f"Initial Loss (Untrained): {loss.item():.4f}")
    
    assert logits.shape == (batch_size, block_size, vocab_size), "ERROR: Logits shape is wrong!"
    print("\nSUCCESS: The full Mini-GPT architecture is assembled and ready to learn.")