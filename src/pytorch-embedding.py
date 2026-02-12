import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_seq_len):
        super().__init__()
        # Learned embedding matrices
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        
    def forward(self, token_ids):
        # token_ids shape: [batch_size, seq_len]
        batch_size, seq_len = token_ids.shape
        
        # Get token embeddings via lookup
        tok_emb = self.token_embedding(token_ids)  # [batch, seq_len, embed_dim]
        
        # Create position indices [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, device=token_ids.device)
        pos_emb = self.position_embedding(positions)  # [seq_len, embed_dim]
        
        # Add them together (broadcasting handles batch dimension)
        return tok_emb + pos_emb

# Example usage
vocab_size = 10000
embed_dim = 768
max_seq_len = 512

embedding_layer = TokenEmbedding(vocab_size, embed_dim, max_seq_len)

# Simulated token IDs for "I love cats"
token_ids = torch.tensor([[101, 1045, 2293, 8870, 102]])  # [CLS] I love cats [SEP]

output = embedding_layer(token_ids)
print(output.shape)  # torch.Size([1, 5, 768])