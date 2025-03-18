import torch
from transformers import pipeline

from kvpress import FinchPress

device = "cuda:5"
model = "mistralai/Mistral-7B-Instruct-v0.3"
model_kwargs = {"attn_implementation": "eager", "torch_dtype": torch.float16}
pipe = pipeline("kv-press-text-generation", model=model, device=device, model_kwargs=model_kwargs)

context = "A very long text you want to compress once and for all"
question = "\nA question about the compressed context"  # optional

tokenizer = pipe.tokenizer
tok_context = tokenizer(context, return_tensors="pt")["input_ids"]
tok_question = tokenizer(question, return_tensors="pt")["input_ids"]
print("len context:", len(tok_context[0]))
print("len question:", len(tok_question[0]))
press = FinchPress(compression_ratio=0.5)
answer = pipe(context, question=question, press=press)["answer"]

print(answer)