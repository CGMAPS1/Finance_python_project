import transformers
import torch
import nltk
import spacy
import math

# Example: Using transformers pipeline and AutoTokenizer

# Load a sentiment-analysis pipeline
classifier = transformers.pipeline("sentiment-analysis")

# Run the pipeline on a sample text
result = classifier("Transformers library makes NLP easy!")
print("Sentiment analysis result:", result)

# Using AutoTokenizer to tokenize text
from transformers import AutoTokenizer

# It's recommended to specify the model name so that the tokenizer matches the model used in the pipeline.
# For example, if you use a pipeline with a specific model, use the same model name for the tokenizer:
# It is not strictly necessary to specify the model name, but doing so ensures the tokenizer is compatible with the model.
# If you do not specify a model, the tokenizer may not match the model's vocabulary or tokenization rules.
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# This line tokenizes the input text "Transformers are powerful!" and returns the result as PyTorch tensors.
# return_tensors="pt" means the output will be PyTorch tensors (if you use TensorFlow, use "tf" instead).
tokens = tokenizer("Transformers are powerful!", return_tensors="pt")
print("Tokenized input:", tokens)

def positional_encoding(seq_len, d_model):
    """
    Create positional encoding for input sequences.

    Args:
        seq_len (int): Length of the input sequence.
        d_model (int): Embedding dimension.

    Returns:
        torch.Tensor: Positional encoding of shape (seq_len, d_model)
    """
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

# Example usage:
seq_len = 10
d_model = 16
pe = positional_encoding(seq_len, d_model)
print("Positional Encoding:\n", pe)

