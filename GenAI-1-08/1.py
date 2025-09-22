from transformers import pipeline
from transformers import logging
import warnings

logging.set_verbosity_error()


def show_topk(results, k=3):
    """
    Выводит топ-k результатов предсказания модели.
    
    Args:
        results (list): Список результатов предсказания
        k (int): Количество результатов для отображения
    """
    for i, r in enumerate(results[:k], 1):
        print(f"{i}. {r['token_str']}  (score={r['score']:.4f})")


def bert_model(model_version, text, device='cpu', top_k=3):
    """
    Загружает и использует BERT модель для заполнения маски.
    
    Args:
        model_version (str): Версия/название модели
        text (str): Текст с маской [MASK] для заполнения
        device (str): Устройство для выполнения (cpu/cuda)
        top_k (int): Количество лучших вариантов для возврата
        
    Returns:
        list: Результаты предсказания или пустой список при ошибке
    """
    try:
        model = pipeline("fill-mask", model=model_version, device=device)
        results = model(text, top_k=top_k)
        return results
    except Exception as e:
        warnings.warn(f"Ошибка при загрузке модели {model_version}: {str(e)}")
        return []


def check_prediction_accuracy(predictions):
    """
    Проверяет, содержит ли хотя бы одно из предсказаний правильный ответ.
    
    Args:
        predictions (list): Список предсказаний модели
        
    Returns:
        bool: True если есть правильное предсказание, иначе False
    """
    gold_variants = {"России", "Российской Федерации", "РФ"}


    # Бозопасное извлечение ответов 
    pred_tokens = {pred["token_str"].strip() for pred in predictions[:3]}
    return len(gold_variants & pred_tokens) > 0


def main():
    """Основная функция для тестирования различных BERT моделей."""
    text = "Москва - столица [MASK]."
    
    print("En BERT")
    res_en = bert_model('bert-base-uncased', text)
    if res_en:
        show_topk(res_en, k=3)
        accuracy = check_prediction_accuracy(res_en)
        print("Метрика:", "OK" if accuracy else "MISS")
    else:
        print("Не удалось загрузить модель или получить предсказания")
    print()
    
    print("Ру BERT")
    res_ru = bert_model('DeepPavlov/rubert-base-cased', text)
    if res_ru:
        show_topk(res_ru, k=3)
        accuracy = check_prediction_accuracy(res_ru)
        print("Метрика:", "OK" if accuracy else "MISS")
    else:
        print("Не удалось загрузить модель или получить предсказания")
    print()
    
    print("Многоязычная BERT")
    res_mbert = bert_model('bert-base-multilingual-cased', text)
    if res_mbert:
        show_topk(res_mbert, k=3)
        accuracy = check_prediction_accuracy(res_mbert)
        print("Метрика:", "OK" if accuracy else "MISS")
    else:
        print("Не удалось загрузить модель или получить предсказания")
    print()


if __name__ == "__main__":
    main()