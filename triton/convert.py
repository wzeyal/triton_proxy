import torch
from transformers import pipeline

# Download sentiment analysis model from Hugging Face
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
sentiment_analyzer = pipeline("sentiment-analysis", model=model_name, framework="pt")

# Example text for inference
text = "I love using Hugging Face for natural language processing tasks!"

# Perform inference with the PyTorch model
output = sentiment_analyzer(text)

# Save the PyTorch model
torch_model_path = "sentiment_model.pth"
torch.save(sentiment_analyzer.model, torch_model_path)

# Convert the PyTorch model to ONNX
dummy_input = torch.tensor([sentiment_analyzer.tokenizer.encode(text)])  # Convert to LongTensor
onnx_model_path = "sentiment_model.onnx"
torch.onnx.export(sentiment_analyzer.model, dummy_input, onnx_model_path)

print(f"Model saved to {torch_model_path}")
print(f"ONNX model saved to {onnx_model_path}")
