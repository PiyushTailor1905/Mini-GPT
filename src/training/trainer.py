# src/training/trainer.py

import torch
from tqdm import tqdm # This gives us a nice progress bar in the terminal
from src.utils.data_loader import get_batch

class Trainer:
    def __init__(
        self,
        model,
        train_data: torch.Tensor,
        val_data: torch.Tensor,
        batch_size: int = 32,
        block_size: int = 64,
        learning_rate: float = 3e-4, # 0.0003 is the industry standard starting rate for transformers
        device: str = 'cpu',
        eval_interval: int = 100,
        eval_iters: int = 50,
    ):
        self.model = model.to(device)
        self.train_data = train_data
        self.val_data = val_data
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device
        self.eval_interval = eval_interval
        self.eval_iters = eval_iters
        
        # The Optimizer (The Personal Trainer)
        # AdamW is an industry standard. It adjusts the learning rate automatically 
        # and applies "weight decay" (a penalty that stops numbers from getting too large).
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-1)

    @torch.no_grad()
    def estimate_loss(self):
        """
        Takes a break from training to test the model on both the training set 
        and the unseen validation set. 
        @torch.no_grad() tells PyTorch: "Do not track gradients here, we are just testing!"
        This saves a massive amount of memory and time.
        """
        out = {}
        self.model.eval() # Tell layers like Dropout to turn off during testing
        
        for split in ['train', 'val']:
            data = self.train_data if split == 'train' else self.val_data
            losses = torch.zeros(self.eval_iters)
            
            for k in range(self.eval_iters):
                x, y = get_batch(data, self.block_size, self.batch_size, self.device)
                _, loss = self.model(x, y)
                losses[k] = loss.item()
                
            out[split] = losses.mean().item()
            
        self.model.train() # Turn Dropout back on for training
        return out

    def train(self, max_iters: int):
        """The Main Engine Loop"""
        self.model.train()
        print(f"Starting training on {self.device}...")
        
        # tqdm creates a progress bar for our loop
        for iteration in tqdm(range(max_iters), desc="Training"):
            
            # 1. Periodically check how well we are doing
            if iteration % self.eval_interval == 0 or iteration == max_iters - 1:
                losses = self.estimate_loss()
                tqdm.write(f"Step {iteration:4d} | Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f}")
            
            # 2. Grab a batch of training data
            x, y = get_batch(self.train_data, self.block_size, self.batch_size, self.device)
            
            # 3. Forward Pass
            logits, loss = self.model(x, y)
            
            # 4. Wipe the slate clean from the last step
            self.optimizer.zero_grad(set_to_none=True)
            
            # 5. Backward Pass (Calculate the errors)
            loss.backward()
            
            # 6. Gradient Clipping (Safety Mechanism)
            # If a sudden error spike creates a massive gradient, it can destroy the weights.
            # This caps the maximum adjustment at 1.0.
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 7. Update the weights
            self.optimizer.step()

# --- QUICK TEST BLOCK ---
if __name__ == '__main__':
    from src.model.gpt import MiniGPT
    
    # 1. Setup minimal dummy environment
    vocab_size = 65
    block_size = 8
    batch_size = 4
    embed_dim = 32
    device = 'cpu'
    
    # Create dummy datasets (100 random characters each)
    dummy_train = torch.randint(0, vocab_size, (100,))
    dummy_val = torch.randint(0, vocab_size, (100,))
    
    # 2. Initialize Model and Trainer
    model = MiniGPT(vocab_size, embed_dim, num_heads=2, num_layers=2, block_size=block_size)
    trainer = Trainer(
        model=model, 
        train_data=dummy_train, 
        val_data=dummy_val, 
        batch_size=batch_size, 
        block_size=block_size,
        device=device,
        eval_interval=5, # Evaluate very often for this quick test
        eval_iters=2     # Only test 2 batches to save time
    )
    
    # 3. Run a tiny training loop (10 iterations)
    trainer.train(max_iters=10)
    print("\nSUCCESS: The Training Loop executes without crashing!")