import torch
from transformers import pipeline

# Download sentiment analysis model from Hugging Face
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
sentiment_analyzer = pipeline("sentiment-analysis", model=model_name, framework="pt")

# Example text for inference
text = "I love using Hugging Face for natural language processing tasks!"

# Perform inference with the PyTorch model
output = sentiment_analyzer(text)

print(output)

tokens = sentiment_analyzer.tokenizer(text, return_tensors="pt")
# Perform forward pass to get logits
with torch.no_grad():
    logits = sentiment_analyzer.model(**tokens).logits
print(logits)

probabilities = torch.softmax(logits, dim=-1).tolist()
print(probabilities)

# Save the PyTorch model
torch_model_path = "sentiment_model.pth"
torch.save(sentiment_analyzer.model, torch_model_path)
sentiment_analyzer.model.save_pretrained("./model_pre_trained/")
sentiment_analyzer.tokenizer.save_pretrained("./tokenizer_pre_trained/")

model_input = sentiment_analyzer.tokenizer.encode(text, padding="max_length")

batch_size = 1

# Convert the PyTorch model to ONNX
dummy_input = torch.tensor([model_input])  # Convert to LongTensor

dummy_input = torch.randint(low=0, high=100, size=(1, 512), dtype=torch.int64)

onnx_model_path = "/home/eyalshw/git/server/docs/examples/model_repository/sentiment_model/1/sentiment_model.onnx"
# torch.onnx.export(sentiment_analyzer.model, dummy_input, onnx_model_path)

torch.onnx.export(
    sentiment_analyzer.model,
    dummy_input,
    onnx_model_path,
    input_names = ['input_0'],
    output_names = ['output_0'],
    dynamic_axes={
        'input_0': {0: 'batch_size'},
        'output_0': {0: 'batch_size'}}
)

print(f"Model saved to {torch_model_path}")
print(f"ONNX model saved to {onnx_model_path}")
