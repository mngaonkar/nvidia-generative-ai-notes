from transformers import BertTokenizer, BertModel
import torch

# Load pretrained model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Tokenize input
text = "I love cats"
inputs = tokenizer(text, return_tensors='pt')

print("Token IDs:", inputs['input_ids'])
# tensor([[ 101, 1045, 2293, 8870,  102]])
# [CLS]=101, I=1045, love=2293, cats=8870, [SEP]=102

print("Tokens:", tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]))
# ['[CLS]', 'i', 'love', 'cats', '[SEP]']

# Access embedding layers directly
token_embeddings = model.embeddings.word_embeddings(inputs['input_ids'])
position_ids = torch.arange(inputs['input_ids'].shape[1]).unsqueeze(0)
position_embeddings = model.embeddings.position_embeddings(position_ids)

print("Token embedding shape:", token_embeddings.shape)  # [1, 5, 768]
print("Position embedding shape:", position_embeddings.shape)  # [1, 5, 768]

# BERT also adds token_type_embeddings (segment embeddings)
token_type_ids = torch.zeros_like(inputs['input_ids'])
token_type_embeddings = model.embeddings.token_type_embeddings(token_type_ids)

# Final embedding (before LayerNorm and dropout)
combined = token_embeddings + position_embeddings + token_type_embeddings
print("Combined shape:", combined.shape)  # [1, 5, 768]