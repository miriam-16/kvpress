import requests
from bs4 import BeautifulSoup

import torch
from transformers import pipeline
from kvpress import FinchPress, KeyRerotationPress, SnapKVPress, FinchPressTSNaive,FinchPressWTS, FinchPressWCS,FinchPressTCSNaive, FinchPressTSHavg, FinchPressTWSHavg,FinchPressTSHavgPrecise,FinchPressTCSNaiveHavg, FinchPressCWSHavg, FinchPressKeepStructuralTokensForTSHAvg



context =  '[<header>["Year","Film","Role","Language","Notes"]<header>["2008","Moggina Manasu","Chanchala","Kannada","Filmfare Award for Best Actress - Kannada\nKarnataka State Film Award for Best Actress"]<tuple_end>["2009","Olave Jeevana Lekkachaara","Rukmini","Kannada","Innovative Film Award for Best Actress"]<tuple_end>["2009","Love Guru","Kushi","Kannada","Filmfare Award for Best Actress - Kannada"]<tuple_end>["2010","Krishnan Love Story","Geetha","Kannada","Filmfare Award for Best Actress - Kannada\nUdaya Award for Best Actress"]<tuple_end>["2010","Gaana Bajaana","Radhey","Kannada",""]<tuple_end>["2011","Hudugaru","Gayithri","Kannada","Nominated, Filmfare Award for Best Actress – Kannada"]<tuple_end>["2012","Alemari","Neeli","Kannada",""]<tuple_end>["2012","Breaking News","Shraddha","Kannada",""]<tuple_end>["2012","Addhuri","Poorna","Kannada","Udaya Award for Best Actress\nNominated — SIIMA Award for Best Actress\nNominated — Filmfare Award for Best Actress – Kannada"]<tuple_end>["2012","18th Cross","Punya","Kannada",""]<tuple_end>["2012","Sagar","Kajal","Kannada",""]<tuple_end>["2012","Drama","Nandini","Kannada",""]<tuple_end>["2013","Kaddipudi","Uma","Kannada",""]<tuple_end>["2013","Dilwala","Preethi","Kannada",""]<tuple_end>["2013","Bahaddoor","Anjali","Kannada","Filming"]<tuple_end>["2014","Mr. & Mrs. Ramachari","","","Announced"]<tuple_end>["2014","Endendigu","","","Filming"]<tuple_end>]'
question= '''
Question:what is the total number of films with the language of kannada listed?'''

model_name= "HuggingFaceTB/SmolLM-135M-Instruct"
device="cpu"


model_kwargs = {"attn_implementation": "sdpa", "torch_dtype": torch.float16}
pipe = pipeline("kv-press-text-generation", model=model_name, device=device, model_kwargs=model_kwargs)


print("MODEL LOADED")



press= FinchPressTCSNaive(compression_ratio=0.8, split_size=1)

#press=SnapKVPress(compression_ratio=0.1,window_size=3)
answer = pipe(context, question=question, press=press)["answer"]


print("FINAL ANSWER IS")
print(answer)