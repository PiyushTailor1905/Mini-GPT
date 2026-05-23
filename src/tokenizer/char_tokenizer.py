# src/tokenizer/char_tokenizer.py

class CharTokenizer:
    def __init__(self, text: str):
        # 1. Identify the unique components
        # set(text) removes all duplicate characters.
        # sorted() ensures 'a' always gets the same ID every time you run the script.
        self.chars = sorted(set(text))
        self.vocab_size = len(self.chars)
        
        # 2. Build the translation dictionaries
        # stoi (string-to-integer): The encoder ring (e.g., {'a': 0, 'b': 1})
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        
        # itos (integer-to-string): The decoder ring (e.g., {0: 'a', 1: 'b'})
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
    
    def encode(self, text: str) -> list[int]:
        """Translates a string into a list of integers."""
        return [self.stoi[ch] for ch in text]
    
    def decode(self, indices: list[int]) -> str:
        """Translates a list of integers back into a string."""
        return ''.join([self.itos[i] for i in indices])

# --- QUICK TEST BLOCK ---
if __name__ == '__main__':
    # We use a dummy string to represent our "dataset" for this test
    sample_dataset = "hello world! this is a test."
    
    # Initialize the tokenizer
    tokenizer = CharTokenizer(sample_dataset)
    
    print(f"Vocabulary Size: {tokenizer.vocab_size}")
    print(f"Characters: {tokenizer.chars}")
    
    # The Roundtrip Test
    original_text = "hello test!"
    encoded = tokenizer.encode(original_text)
    decoded = tokenizer.decode(encoded)
    
    print(f"\nOriginal: {original_text}")
    print(f"Encoded:  {encoded}")
    print(f"Decoded:  {decoded}")
    
    assert original_text == decoded, "ERROR: The decoder did not return the original text!"
    print("\nSUCCESS: The roundtrip test passed.")