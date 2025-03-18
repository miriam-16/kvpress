import torch
from transformers import pipeline

from kvpress import SnapKVPress, KeyRerotationPress, FinchPress

device = "cuda:5"
model = "HuggingFaceTB/SmolLM-135M-Instruct"
# model = "mistralai/Mistral-7B-Instruct-v0.3"
model_kwargs = {"attn_implementation": "sdpa", "torch_dtype": torch.float16}
pipe = pipeline("kv-press-text-generation", model=model, device=device, model_kwargs=model_kwargs)

context = "My name is Elia and I am 60 years old."
question = "\nHow old is Elia?"  # optional

tokenizer = pipe.tokenizer
tok_context = tokenizer(context, return_tensors="pt")["input_ids"]
tok_question = tokenizer(question, return_tensors="pt")["input_ids"]
print("len context:", len(tok_context[0]))
print("len question:", len(tok_question[0]))
# press = KeyRerotationPress(FinchPress(compression_ratio=0.2))
press = FinchPress(compression_ratio=0.2)
answer = pipe(context, question=question, press=press)["answer"]

print(answer)