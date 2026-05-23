# src/utils/data_loader.py

import torch
from src.tokenizer.char_tokenizer import CharTokenizer

def load_data(filepath: str, train_split: float = 0.9):
    """
    Reads the file, builds the tokenizer, and splits the data into train/val sets.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # 1. Build the translation dictionary using the full text
    tokenizer = CharTokenizer(text)
    
    # 2. Encode the entire text into a 1D PyTorch tensor of 64-bit integers
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    
    # 3. Split the data. 
    # NEVER shuffle sequential data (like text or time-series).
    # Shuffling destroys the temporal relationships the model needs to learn.
    n = int(train_split * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    return train_data, val_data, tokenizer

def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: str = 'cpu'):
    """
    Samples a randomized batch of inputs (x) and targets (y).
    """
    # 1. Generate 'batch_size' number of random starting indices
    # We subtract block_size so we don't accidentally try to read past the end of the text
    ix = torch.randint(len(data) - block_size, (batch_size,))
    
    # 2. Slice out the input (x) and target (y) sequences
    # torch.stack takes a list of 1D tensors and stacks them into a 2D grid
    x = torch.stack([data[i : i + block_size] for i in ix])
    
    # y is shifted exactly ONE position to the right of x
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    
    # 3. Move the data to the target hardware (CPU or GPU)
    return x.to(device), y.to(device)

# --- QUICK TEST BLOCK ---
if __name__ == '__main__':
    # 1. Create a tiny dummy dataset to verify the logic
    dummy_text = "The quick brown fox jumps over the lazy dog."
    
    # 2. Manually execute what load_data does so we don't need a file path for this test
    tokenizer = CharTokenizer(dummy_text)
    data = torch.tensor(tokenizer.encode(dummy_text), dtype=torch.long)
    
    block_size = 8  # How many characters the model looks at
    batch_size = 4  # How many sequences we process at once
    
    # 3. Run the batcher
    x, y = get_batch(data, block_size, batch_size)
    
    print(f"Batch Shape: {x.shape} (batch_size, block_size)")
    
    # 4. Verify the "Teacher Forcing" shift
    print("\nVerifying Teacher Forcing (Shift by 1):")
    for b in range(1): # Just look at the first sequence in the batch
        input_text = tokenizer.decode(x[b].tolist())
        target_text = tokenizer.decode(y[b].tolist())
        
        print(f"Input X: '{input_text}'")
        print(f"Target Y: '{target_text}'")
        
        # The first character of Y should be the second character of X
        assert y[b, 0] == x[b, 1], "ERROR: Y is not shifted correctly!"
    
    print("\nSUCCESS: The data loader and batcher logic works perfectly.")