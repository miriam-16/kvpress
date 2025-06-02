import requests
from bs4 import BeautifulSoup

import torch
from transformers import pipeline
from kvpress import FinchPress, KeyRerotationPress, SnapKVPress, FinchPressTSNaive,FinchPressWTS, FinchPressWCS,FinchPressTCSNaive, FinchPressTSHavg, FinchPressTWSHavg,FinchPressTSHavgPrecise,FinchPressTCSNaiveHavg, FinchPressCWSHavg, FinchPressKeepStructuralTokensForTSHAvg



context =  '[<header>["Rank","Cyclist","Team","Time","UCI ProTour\nPoints"]<header>["1","Alejandro Valverde (ESP)","Caisse d\'Epargne","5h 29\' 10"","40"]<tuple_end>["2","Alexandr Kolobnev (RUS)","Team CSC Saxo Bank","s.t.","30"]<tuple_end>["3","Davide Rebellin (ITA)","Gerolsteiner","s.t.","25"]<tuple_end>["4","Paolo Bettini (ITA)","Quick Step","s.t.","20"]<tuple_end>["5","Franco Pellizotti (ITA)","Liquigas","s.t.","15"]<tuple_end>["6","Denis Menchov (RUS)","Rabobank","s.t.","11"]<tuple_end>["7","Samuel Sánchez (ESP)","Euskaltel-Euskadi","s.t.","7"]<tuple_end>["8","Stéphane Goubert (FRA)","Ag2r-La Mondiale","+ 2"","5"]<tuple_end>["9","Haimar Zubeldia (ESP)","Euskaltel-Euskadi","+ 2"","3"]<tuple_end>["10","David Moncoutié (FRA)","Cofidis","+ 2"","1"]<tuple_end>]'
question= "Question:which country had the most cyclists finish within the top 10?"

model_name= "HuggingFaceTB/SmolLM-135M-Instruct"
device="cpu"


model_kwargs = {"attn_implementation": "sdpa", "torch_dtype": torch.float16}
pipe = pipeline("kv-press-text-generation", model=model_name, device=device, model_kwargs=model_kwargs)


print("MODEL LOADED")



press= FinchPressTSNaive(compression_ratio=0.8, split_size=1)

#press=SnapKVPress(compression_ratio=0.1,window_size=3)
answer = pipe(context, question=question, press=press)["answer"]


print("FINAL ANSWER IS")
print(answer)