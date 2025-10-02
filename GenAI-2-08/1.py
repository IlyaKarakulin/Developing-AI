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
    # Обрабатываем случай, когда results - список списков (несколько масок)
    if results and isinstance(results[0], list):
        for mask_idx, mask_results in enumerate(results):
            print(f"Маска {mask_idx + 1}:")
            for i, r in enumerate(mask_results[:k], 1):
                print(f"  {i}. {r['token_str']}  (score={r['score']:.4f})")
            print()
    else:
        # Обычный случай с одной маской
        for i, r in enumerate(results[:k], 1):
            print(f"{i}. {r['token_str']}  (score={r['score']:.4f})")


def bert_model(model_version, text, device='cpu', top_k=3, targets=None):
    """
    Загружает и использует BERT модель для заполнения маски.
    
    Args:
        model_version (str): Версия/название модели
        text (str): Текст с маской [MASK] для заполнения
        device (str): Устройство для выполнения (cpu/cuda)
        top_k (int): Количество лучших вариантов для возврата
        targets (list): Список целевых слов для ограничения предсказаний
        
    Returns:
        list: Результаты предсказания или пустой список при ошибке
    """
    try:
        model = pipeline("fill-mask", model=model_version, device=device)
        if targets:
            results = model(text, top_k=top_k, targets=targets)
        else:
            results = model(text, top_k=top_k)
        return results
    except Exception as e:
        warnings.warn(f"Ошибка при загрузке модели {model_version}: {str(e)}")
        return []


def get_best_prediction(results, mask_index=0):
    """
    Извлекает лучший вариант предсказания для указанной маски.
    
    Args:
        results (list): Результаты модели
        mask_index (int): Индекс маски (если несколько)
        
    Returns:
        str: Лучший вариант предсказания
    """
    if not results:
        return "[MASK]"
    
    # Если несколько масок, results - список списков
    if isinstance(results[0], list):
        if mask_index < len(results) and results[mask_index]:
            return results[mask_index][0]['token_str'].strip()
        else:
            return "[MASK]"
    else:
        # Одна маска
        return results[0]['token_str'].strip()


def fill_multiple_masks(model_version, text, targets=None, device='cpu'):
    """
    Заполняет несколько масок с использованием targets.
    
    Args:
        model_version (str): Версия модели
        text (str): Текст с несколькими [MASK]
        targets (list): Список целевых слов для ограничения предсказаний
        device (str): Устройство для выполнения
        
    Returns:
        list: Результаты предсказания
    """
    results = bert_model(model_version, text, device=device, top_k=3, targets=targets)
    
    if results:
        show_topk(results, k=3)
        
        filled_text = text
        mask_count = text.count('[MASK]')
        
        for i in range(mask_count):
            best_pred = get_best_prediction(results, i)
            filled_text = filled_text.replace('[MASK]', best_pred, 1)
        
        print(f"Заполненный текст: {filled_text}")
    else:
        print("Не удалось получить предсказания")
    
    print()
    return results


def main():
    text = "Москва - [MASK] [MASK]."
        
    # 1. Последовательное заполнение масок
    print("1. ПОСЛЕДОВАТЕЛЬНОЕ ЗАПОЛНЕНИЕ МАСОК")
    print("-" * 40)

    print("Ру BERT (с targets):")
    fill_multiple_masks('DeepPavlov/rubert-base-cased', text)
    
    print("Многоязычная BERT (последовательно):")
    fill_multiple_masks('bert-base-multilingual-cased', text)
    print()
    
    # 2. Заполнение с использованием targets
    print("2. ЗАПОЛНЕНИЕ С ИСПОЛЬЗОВАНИЕМ TARGETS")
    print("-" * 40)
    
    # Целевые слова для ограничения предсказаний
    targets = ["столица", "России", "город", "большой"]
    
    print("Ру BERT (с targets):")
    fill_multiple_masks('DeepPavlov/rubert-base-cased', text, targets)
    
    print("Многоязычная BERT (с targets):")
    fill_multiple_masks('bert-base-multilingual-cased', text, targets)

if __name__ == "__main__":
    main()