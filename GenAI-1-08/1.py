from transformers import pipeline
from transformers import logging

logging.set_verbosity_error()


def show_topk(results, k=3):
    for i, r in enumerate(results[:k], 1):
        print(f"{i}. {r['token_str']}  (score={r['score']:.4f})")

def BERT(model_version, device='cpu'):
    model = pipeline("fill-mask", model=model_version, device=device)
    res = model(text, top_k=3)
    return res

def test(predict):
    gold_variants = {"России", "Российской Федерации", "РФ"}
    pred_tokens = {r["token_str"].strip() for r in predict[:3]}
    hit = len(gold_variants & pred_tokens) > 0.
    return hit

text = "Москва - столица [MASK]."

print("En BERT")
res_en = BERT('bert-base-uncased')
show_topk(res_en, k=3)
print("Метрика:", "OK" if test(res_en) else "MISS", '\n')


print("Ру BERT")
res_ru = BERT('DeepPavlov/rubert-base-cased')
show_topk(res_ru, k=3)
print("Метрика:", "OK" if test(res_ru) else "MISS", '\n')


print("Многоязычная BERT")
res_mbert = BERT('bert-base-multilingual-cased')
show_topk(res_mbert, k=3)
print("Метрика:", "OK" if test(res_mbert) else "MISS", '\n')



