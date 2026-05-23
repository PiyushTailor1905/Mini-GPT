# train.py

import torch
from src.utils.data_loader import load_data
from src.model.gpt import MiniGPT
from src.training.trainer import Trainer

# ─────────────────────────────────────────────
# 1. THE CONTROL PANEL (Configuration)
# ─────────────────────────────────────────────
CONFIG = {
    # Data logistics
    'data_path':     'data/input.txt',
    'block_size':    128,    # How many characters of context the model gets
    
    # Model architecture (The Engine)
    'embed_dim':     256,   
    'num_heads':     8,     
    'num_layers':    6,     # 4 Transformer blocks stacked
    'dropout':       0.1,
    
    # Training parameters (The Gym)
    'batch_size':    32,    # Sequences processed simultaneously
    'max_iters':     10000,  # Total gradient updates
    'learning_rate': 3e-4,  # Standard starting point for AdamW
    'eval_interval': 500,   # How often to check the validation loss
    'eval_iters':    50,    # How many batches to average for validation
    
    # Hardware routing
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # Generation settings
    'gen_tokens':    200,   # How many characters to write at the end
    'temperature':   0.8,   # 1.0 is default, 0.8 is slightly more conservative/safe
}

def main():
    print("=" * 50)
    print("INITIATING MINI-GPT PIPELINE")
    print("=" * 50)
    print(f"Target Hardware: {CONFIG['device'].upper()}")
    
    # ── Step 1: Load the Raw Materials ────────────────
    train_data, val_data, tokenizer = load_data(CONFIG['data_path'])
    
    # ── Step 2: Assemble the Engine ───────────────────
    model = MiniGPT(
        vocab_size  = tokenizer.vocab_size,
        embed_dim   = CONFIG['embed_dim'],
        num_heads   = CONFIG['num_heads'],
        num_layers  = CONFIG['num_layers'],
        block_size  = CONFIG['block_size'],
        dropout     = CONFIG['dropout'],
    )
    print(f"Model Parameters: {model.count_parameters():,}")
    
    # ── Step 3: Run the Training Loop ─────────────────
    trainer = Trainer(
        model          = model,
        train_data     = train_data,
        val_data       = val_data,
        batch_size     = CONFIG['batch_size'],
        block_size     = CONFIG['block_size'],
        learning_rate  = CONFIG['learning_rate'],
        device         = CONFIG['device'],
        eval_interval  = CONFIG['eval_interval'],
        eval_iters     = CONFIG['eval_iters'],
    )
    
    # This will take a few minutes on a CPU, or seconds on a GPU
    trainer.train(max_iters=CONFIG['max_iters'])
    
    # ── Step 4: The Showroom (Generation) ─────────────
    print("\n" + "=" * 50)
    print("MODEL FULLY TRAINED. GENERATING TEXT:")
    print("=" * 50)
    
    # We kickstart the generation with a single newline character
    start_ids = torch.tensor([[tokenizer.stoi['\n']]], dtype=torch.long, device=CONFIG['device'])
    
    # Let the model write
    generated_ids = model.generate(
        idx            = start_ids,
        max_new_tokens = CONFIG['gen_tokens'],
        temperature    = CONFIG['temperature'],
    )
    
    # Decode the integers back to Shakespearean text
    generated_text = tokenizer.decode(generated_ids[0].tolist())
    print(generated_text)

if __name__ == '__main__':
    main()