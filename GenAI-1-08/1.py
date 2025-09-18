# pip install transformers torch
import torch
from transformers import pipeline

def show_topk(results, k=3):
    for i, r in enumerate(results[:k], 1):
        print(f"{i}. {r['token_str']}  (score={r['score']:.4f})")

text = "Москва - столица [MASK]."

device = 0 if torch.cuda.is_available() else -1

print("En BERT")
unmasker_en = pipeline("fill-mask", model="bert-base-uncased", device=device)
res_en = unmasker_en(text, top_k=3)
show_topk(res_en, k=3)

print("Ру BERT")
unmasker_ru = pipeline("fill-mask", model="DeepPavlov/rubert-base-cased", device=device)
res_ru = unmasker_ru(text, top_k=3)
show_topk(res_ru, k=3)

print("Многоязычная BERT")
unmasker_mbert = pipeline("fill-mask", model="bert-base-multilingual-cased", device=device)
res_mbert = unmasker_mbert(text, top_k=3)
show_topk(res_mbert, k=3)

gold_variants = {"России", "Российской Федерации"}
pred_tokens = {r["token_str"].strip() for r in res_ru[:3]}
hit = len(gold_variants & pred_tokens) > 0.
print("\nМетрика (правильный вариант в топ-3 у русской модели):", "OK" if hit else "MISS")
