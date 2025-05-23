import requests
from bs4 import BeautifulSoup

import torch
from transformers import pipeline
from kvpress import FinchPress, KeyRerotationPress, SnapKVPress, FinchPressTSNaive



context = "[[1, 2, 3]<tuple_end>[4, 5, 6]<tuple_end>[7, 8, 9]<tuple_end>]"
question= "come stai?"

model_name= "HuggingFaceTB/SmolLM-135M-Instruct"
device="cpu"


model_kwargs = {"attn_implementation": "sdpa", "torch_dtype": torch.float16}
pipe = pipeline("kv-press-text-generation", model=model_name, device=device, model_kwargs=model_kwargs)


print("MODEL LOADED")



press= FinchPressTSNaive(compression_ratio=0.6, split_size=1)

#press=SnapKVPress(compression_ratio=0.1,window_size=3)
answer = pipe(context, question=question, press=press)["answer"]


print("FINAL ANSWER IS")
print(answer)