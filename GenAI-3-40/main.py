import json
import re
from typing import Dict, List, Any, Tuple
from datetime import datetime
import random
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    pipeline,
    set_seed
)
import nltk

nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocalContentGenerator:
    def __init__(self, model_name: str = "IlyaGusev/rut5_base"):
        """Инициализация локальной модели для генерации контента"""
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Используется устройство: {self.device}")
        
        # Инициализация компонентов
        self.topics = self._initialize_topics()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Загрузка модели для генерации текста
        try:
            logger.info("Загрузка модели для генерации текста...")
            self.text_generator = pipeline(
                "text-generation",
                model=model_name,
                tokenizer=model_name,
                device=0 if self.device == "cuda" else -1
            )
            logger.info("Модель успешно загружена")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            self.text_generator = None
    
    def _initialize_topics(self) -> List[str]:
        """Инициализация списка синтетических тем"""
        return [
            "Искусственный интеллект в современном образовании: возможности и вызовы",
            "Возобновляемая энергетика и устойчивое развитие городов",
            "Квантовые вычисления: революция в компьютерных технологиях",
            "Биотехнологии и генная инженерия в медицине будущего",
            "Космический туризм и освоение космоса частными компаниями",
            "Цифровая трансформация бизнеса в постпандемийную эпоху",
            "Умные города и интернет вещей: технологии для комфортной жизни",
            "Нейротехнологии и интерфейсы мозг-компьютер",
            "Блокчейн и децентрализованные финансы (DeFi)",
            "Робототехника и автоматизация в промышленности 4.0"
        ]
    
    def select_topic(self) -> str:
        """1. Выбор темы"""
        topic = random.choice(self.topics)
        logger.info(f"Выбрана тема: {topic}")
        return topic
    
    def analyze_topic(self, topic: str) -> Dict[str, Any]:
        """2. Анализ текстов по теме (ключевые слова, стиль)"""
        logger.info("Проведение анализа темы...")
        
        # Извлечение ключевых слов из темы
        keywords = self._extract_keywords(topic)
        
        # Определение стиля на основе темы
        style = self._determine_style(topic)
        
        # Определение тональности
        tone = self._determine_tone(topic)
        
        analysis = {
            "keywords": keywords,
            "style": style,
            "tone": tone,
            "target_audience": self._determine_audience(topic),
            "complexity": self._assess_complexity(topic)
        }
        
        logger.info(f"Анализ завершен: {len(keywords)} ключевых слов, стиль: {style}")
        return analysis
    
    def _extract_keywords(self, topic: str) -> List[str]:
        """Извлечение ключевых слов из темы"""
        # Удаляем стоп-слова и выделяем значимые слова
        stop_words = {"в", "и", "на", "с", "по", "для", "из", "от", "до", "за", "не", "что", "как"}
        words = re.findall(r'\b[а-яё]+\b', topic.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Добавляем связанные термины на основе темы
        related_terms = self._get_related_terms(topic)
        keywords.extend(related_terms)
        
        return list(set(keywords))  # Убираем дубликаты
    
    def _get_related_terms(self, topic: str) -> List[str]:
        """Генерация связанных терминов на основе темы"""
        topic_lower = topic.lower()
        
        related_terms_map = {
            "искусственный интеллект": ["машинное обучение", "нейросети", "ai", "алгоритмы", "данные"],
            "образование": ["обучение", "студенты", "преподаватели", "университет", "курсы"],
            "энергетика": ["энергия", "солнечная", "ветровая", "электростанция", "экология"],
            "квантовые": ["квант", "вычисления", "кубиты", "суперпозиция", "запутывание"],
            "биотехнологии": ["гены", "днк", "клетки", "терапия", "генетика"],
            "космический": ["космос", "ракеты", "орбита", "спутники", "наса"],
            "цифровая": ["digital", "онлайн", "автоматизация", "технологии", "инновации"],
            "умные города": ["iot", "сенсоры", "инфраструктура", "транспорт", "безопасность"],
            "нейротехнологии": ["мозг", "нейроны", "сигналы", "интерфейс", "когнитивный"],
            "блокчейн": ["криптовалюта", "смарт-контракты", "децентрализация", "биткоин", "эфириум"],
            "робототехника": ["роботы", "автоматизация", "производство", "ии", "манипуляторы"]
        }
        
        related_terms = []
        for key, terms in related_terms_map.items():
            if key in topic_lower:
                related_terms.extend(terms)
        
        return related_terms[:5]  # Ограничиваем количество
    
    def _determine_style(self, topic: str) -> str:
        """Определение стиля написания на основе темы"""
        scientific_keywords = ["квантовые", "биотехнологии", "нейротехнологии", "генная инженерия"]
        popular_keywords = ["туризм", "умные города", "образование", "блокчейн"]
        
        topic_lower = topic.lower()
        
        if any(keyword in topic_lower for keyword in scientific_keywords):
            return "научно-популярный"
        elif any(keyword in topic_lower for keyword in popular_keywords):
            return "публицистический"
        else:
            return "информационно-аналитический"
    
    def _determine_tone(self, topic: str) -> str:
        """Определение тональности"""
        positive_keywords = ["возможности", "развитие", "будущее", "инновации", "перспективы"]
        neutral_keywords = ["анализ", "исследование", "технологии", "современное"]
        
        topic_lower = topic.lower()
        
        if any(keyword in topic_lower for keyword in positive_keywords):
            return "нейтрально-позитивный"
        elif any(keyword in topic_lower for keyword in neutral_keywords):
            return "нейтральный"
        else:
            return "объективный"
    
    def _determine_audience(self, topic: str) -> str:
        """Определение целевой аудитории"""
        if "образование" in topic.lower():
            return "педагоги, студенты, родители"
        elif "медицина" in topic.lower():
            return "врачи, пациенты, исследователи"
        elif "технологии" in topic.lower():
            return "IT-специалисты, бизнес-лидеры, технологические энтузиасты"
        elif "бизнес" in topic.lower():
            return "предприниматели, менеджеры, инвесторы"
        else:
            return "широкая аудитория, интересующаяся технологиями"
    
    def _assess_complexity(self, topic: str) -> str:
        """Оценка сложности темы"""
        complex_keywords = ["квантовые", "нейротехнологии", "генная инженерия", "блокчейн"]
        if any(keyword in topic.lower() for keyword in complex_keywords):
            return "высокая"
        else:
            return "средняя"
    
    def generate_content(self, topic: str, analysis: Dict[str, Any]) -> Dict[str, str]:
        """3. Генерация контента с помощью локальной модели"""
        logger.info("Генерация контента...")
        
        if self.text_generator is None:
            return self._generate_fallback_content(topic, analysis)
        
        try:
            set_seed(42)  # Для воспроизводимости
            
            # Генерация заголовка
            title = self._generate_title(topic, analysis)
            
            # Генерация статьи
            article = self._generate_article(topic, analysis, title)
            
            # Генерация meta description
            meta_description = self._generate_meta_description(topic, article)
            
            # Генерация тегов
            tags = self._generate_tags(topic, analysis, article)
            
            content = {
                "title": title,
                "article": article,
                "meta_description": meta_description,
                "tags": tags
            }
            
            logger.info("Генерация контента завершена")
            return content
            
        except Exception as e:
            logger.error(f"Ошибка генерации контента: {e}")
            return self._generate_fallback_content(topic, analysis)
    
    def _generate_title(self, topic: str, analysis: Dict[str, Any]) -> str:
        """Генерация заголовка"""
        prompt = f"Создай краткий и привлекательный заголовок на тему: {topic}. Стиль: {analysis['style']}. Тональность: {analysis['tone']}."
        
        try:
            result = self.text_generator(
                prompt,
                max_length=100,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=50256
            )
            title = result[0]['generated_text'].replace(prompt, '').strip()
            # Очистка и форматирование заголовка
            title = re.split(r'[.!?]', title)[0].strip()
            return title if len(title) > 10 else f"{topic}: новые перспективы"
        except:
            return f"{topic}: анализ и перспективы"
    
    def _generate_article(self, topic: str, analysis: Dict[str, Any], title: str) -> str:
        """Генерация основной статьи"""
        prompt = f"""Напиши развернутую статью на тему "{title}". 
Стиль: {analysis['style']}. 
Тональность: {analysis['tone']}.
Целевая аудитория: {analysis['target_audience']}.
Используй ключевые слова: {', '.join(analysis['keywords'][:5])}.
Статья должна содержать введение, основную часть и заключение. Объем: 5-7 абзацев."""

        try:
            result = self.text_generator(
                prompt,
                max_length=1500,
                num_return_sequences=1,
                temperature=0.8,
                do_sample=True,
                pad_token_id=50256
            )
            article = result[0]['generated_text'].replace(prompt, '').strip()
            
            # Постобработка статьи
            article = self._postprocess_article(article)
            return article
        except:
            return self._generate_fallback_article(topic, analysis)
    
    def _generate_meta_description(self, topic: str, article: str) -> str:
        """Генерация meta description"""
        # Берем первые два предложения из статьи или создаем описание
        sentences = re.split(r'[.!?]', article)
        if len(sentences) >= 2:
            description = ' '.join(sentences[:2]).strip() + '.'
        else:
            description = f"Узнайте больше о {topic.lower()} в нашей подробной статье."
        
        # Обрезаем до 160 символов (стандарт для meta description)
        if len(description) > 160:
            description = description[:157] + '...'
        
        return description
    
    def _generate_tags(self, topic: str, analysis: Dict[str, Any], article: str) -> List[str]:
        """Генерация тегов"""
        base_tags = analysis['keywords'][:3]  # Берем первые 3 ключевых слова
        
        # Добавляем темы из статьи
        words = re.findall(r'\b[а-яё]{4,}\b', article.lower())
        word_freq = {}
        for word in words:
            if word not in analysis['keywords'] and len(word) > 3:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Берем 2 самых частых слова из статьи
        top_article_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:2]
        article_tags = [word for word, freq in top_article_words]
        
        all_tags = base_tags + article_tags
        return list(set(all_tags))[:5]  # Убираем дубли и ограничиваем количество
    
    def _postprocess_article(self, article: str) -> str:
        """Постобработка сгенерированной статьи"""
        # Удаляем повторяющиеся предложения
        sentences = re.split(r'[.!?]', article)
        unique_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and sentence not in unique_sentences:
                unique_sentences.append(sentence)
        
        # Форматируем в абзацы (каждые 3-4 предложения - абзац)
        formatted_paragraphs = []
        for i in range(0, len(unique_sentences), 3):
            paragraph = '. '.join(unique_sentences[i:i+3]) + '.'
            formatted_paragraphs.append(paragraph)
        
        return '\n\n'.join(formatted_paragraphs)
    
    def _generate_fallback_content(self, topic: str, analysis: Dict[str, Any]) -> Dict[str, str]:
        """Резервная генерация контента если модель недоступна"""
        logger.warning("Использование резервного генератора контента")
        
        title = f"{topic}: анализ современных тенденций"
        
        article_template = """
Введение: Тема {topic} становится все более актуальной в современном мире. 
Это направление объединяет различные аспекты технологического и социального развития.

Основная часть: Современные исследования показывают, что развитие в области {keywords} открывает новые возможности. 
Многие эксперты сходятся во мнении, что это направление будет определять будущее развитие. 
Технологический прогресс позволяет решать задачи, которые ранее казались невозможными.

Заключение: Таким образом, {topic} представляет собой перспективное направление для дальнейшего изучения и развития. 
Необходимо продолжать исследования в этой области для достижения максимального эффекта.
""".format(
    topic=topic,
    keywords=', '.join(analysis['keywords'][:3])
)

        meta_description = f"Анализ современных тенденций и перспектив развития в области {topic.lower()}."
        
        tags = analysis['keywords'][:4] + [topic.split(':')[0].strip()]
        
        return {
            "title": title,
            "article": article_template.strip(),
            "meta_description": meta_description,
            "tags": tags
        }
    
    def _generate_fallback_article(self, topic: str, analysis: Dict[str, Any]) -> str:
        """Резервная генерация статьи"""
        return f"""
Статья посвящена анализу темы "{topic}". В современном мире эта область приобретает все большее значение.

Ключевые аспекты, рассматриваемые в статье:
- Основные тенденции развития
- Технологические инновации
- Перспективы на будущее

Исследования показывают, что использование технологий в данной сфере позволяет достичь значительных результатов. 
Эксперты отмечают рост интереса к этому направлению со стороны как научного сообщества, так и бизнеса.

В заключение можно отметить, что дальнейшее развитие темы "{topic}" будет способствовать прогрессу в смежных областях 
и открывать новые возможности для исследований и практического применения.
""".strip()
    
    def validate_content(self, content: Dict[str, str], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """4. Валидация сгенерированного контента"""
        logger.info("Валидация контента...")
        
        article = content["article"]
        title = content["title"]
        meta_description = content["meta_description"]
        
        validation_results = {
            "length_validation": self._validate_length(content),
            "keywords_validation": self._validate_keywords(article, analysis["keywords"]),
            "tone_validation": self._validate_tone(article, analysis["tone"]),
            "readability_validation": self._validate_readability(article),
            "seo_validation": self._validate_seo(content)
        }
        
        # Общий статус
        overall_status = "valid"
        for check_name, check_result in validation_results.items():
            if not check_result["status"]:
                overall_status = "needs_improvement"
                break
        
        validation_results["overall_status"] = overall_status
        
        logger.info(f"Валидация завершена. Общий статус: {overall_status}")
        return validation_results
    
    def _validate_length(self, content: Dict[str, str]) -> Dict[str, Any]:
        """Проверка длины контента"""
        article_len = len(content["article"])
        title_len = len(content["title"])
        meta_len = len(content["meta_description"])
        
        requirements = {
            "article": {"min": 500, "max": 3000},
            "title": {"min": 20, "max": 80},
            "meta_description": {"min": 50, "max": 160}
        }
        
        article_ok = requirements["article"]["min"] <= article_len <= requirements["article"]["max"]
        title_ok = requirements["title"]["min"] <= title_len <= requirements["title"]["max"]
        meta_ok = requirements["meta_description"]["min"] <= meta_len <= requirements["meta_description"]["max"]
        
        return {
            "status": article_ok and title_ok and meta_ok,
            "details": {
                "article_length": article_len,
                "title_length": title_len,
                "meta_description_length": meta_len,
                "article_requirements": f"{requirements['article']['min']}-{requirements['article']['max']}",
                "title_requirements": f"{requirements['title']['min']}-{requirements['title']['max']}",
                "meta_requirements": f"{requirements['meta_description']['min']}-{requirements['meta_description']['max']}"
            }
        }
    
    def _validate_keywords(self, article: str, keywords: List[str]) -> Dict[str, Any]:
        """Проверка использования ключевых слов"""
        article_lower = article.lower()
        found_keywords = []
        missing_keywords = []
        
        for keyword in keywords[:8]:  # Проверяем первые 8 ключевых слов
            if keyword.lower() in article_lower:
                found_keywords.append(keyword)
            else:
                missing_keywords.append(keyword)
        
        coverage = len(found_keywords) / len(keywords[:8]) if keywords else 0
        status = coverage >= 0.5  # Требуем хотя бы 50% покрытия
        
        return {
            "status": status,
            "details": {
                "total_keywords_checked": len(keywords[:8]),
                "found_keywords": found_keywords,
                "missing_keywords": missing_keywords,
                "coverage_percentage": round(coverage * 100, 1)
            }
        }
    
    def _validate_tone(self, article: str, expected_tone: str) -> Dict[str, Any]:
        """Проверка тональности"""
        # Упрощенная проверка тональности
        positive_words = ["успех", "перспектива", "развитие", "инновация", "прогресс"]
        negative_words = ["проблема", "риск", "опасность", "угроза", "сложность"]
        
        article_lower = article.lower()
        positive_count = sum(1 for word in positive_words if word in article_lower)
        negative_count = sum(1 for word in negative_words if word in article_lower)
        
        # Определяем фактическую тональность
        if positive_count > negative_count + 2:
            actual_tone = "позитивный"
        elif negative_count > positive_count + 2:
            actual_tone = "негативный"
        else:
            actual_tone = "нейтральный"
        
        # Сравниваем с ожидаемой
        tone_mapping = {
            "нейтрально-позитивный": ["нейтральный", "позитивный"],
            "нейтральный": ["нейтральный"],
            "объективный": ["нейтральный", "позитивный"]
        }
        
        expected_tones = tone_mapping.get(expected_tone, ["нейтральный"])
        status = actual_tone in expected_tones
        
        return {
            "status": status,
            "details": {
                "expected_tone": expected_tone,
                "detected_tone": actual_tone,
                "positive_words_count": positive_count,
                "negative_words_count": negative_count
            }
        }
    
    def _validate_readability(self, article: str) -> Dict[str, Any]:
        """Проверка читаемости"""
        sentences = re.split(r'[.!?]', article)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        words = re.findall(r'\b[а-яё]+\b', article.lower())
        
        if sentences and words:
            avg_sentence_length = len(words) / len(sentences)
            avg_word_length = sum(len(word) for word in words) / len(words)
        else:
            avg_sentence_length = 0
            avg_word_length = 0
        
        # Критерии читаемости
        sentence_length_ok = 10 <= avg_sentence_length <= 25
        word_length_ok = 4 <= avg_word_length <= 8
        
        return {
            "status": sentence_length_ok and word_length_ok,
            "details": {
                "avg_sentence_length": round(avg_sentence_length, 1),
                "avg_word_length": round(avg_word_length, 1),
                "total_sentences": len(sentences),
                "total_words": len(words)
            }
        }
    
    def _validate_seo(self, content: Dict[str, str]) -> Dict[str, Any]:
        """SEO-валидация"""
        title = content["title"]
        article = content["article"]
        meta_description = content["meta_description"]
        
        # Проверяем наличие ключевых слов в заголовке и начале статьи
        first_paragraph = article[:200].lower()
        
        issues = []
        
        if len(title) < 20:
            issues.append("Заголовок слишком короткий")
        if len(title) > 80:
            issues.append("Заголовок слишком длинный")
        
        if len(meta_description) < 50:
            issues.append("Meta description слишком короткий")
        if len(meta_description) > 160:
            issues.append("Meta description слишком длинный")
        
        if len(article) < 1000:
            issues.append("Статья слишком короткая для хорошего SEO")
        
        return {
            "status": len(issues) == 0,
            "details": {
                "issues": issues,
                "seo_score": max(0, 10 - len(issues))
            }
        }
    
    def generate_complete_report(self) -> Dict[str, Any]:
        """Генерация полного отчета"""
        logger.info("Запуск комплексной системы генерации контента...")
        
        # Шаг 1: Выбор темы
        topic = self.select_topic()
        
        # Шаг 2: Анализ темы
        analysis = self.analyze_topic(topic)
        
        # Шаг 3: Генерация контента
        content = self.generate_content(topic, analysis)
        
        # Шаг 4: Валидация
        validation = self.validate_content(content, analysis)
        
        # Шаг 5: Формирование отчета
        report = {
            "metadata": {
                "system_name": "LocalContentGenerator",
                "model_used": self.model_name,
                "generation_date": datetime.now().isoformat(),
                "topic_type": "Синтетическая тема"
            },
            "topic_selection": {
                "selected_topic": topic,
                "available_topics_count": len(self.topics)
            },
            "content_analysis": analysis,
            "generated_content": content,
            "content_validation": validation,
            "export_info": {
                "format": "JSON",
                "encoding": "UTF-8",
                "version": "1.0"
            }
        }
        
        logger.info("Комплексная система завершила работу")
        return report
    
    def save_report(self, report: Dict[str, Any], filename: str = None) -> str:
        """Сохранение отчета в JSON файл"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"content_report_{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Отчет сохранен в файл: {filename}")
            return filename
        except Exception as e:
            logger.error(f"Ошибка сохранения отчета: {e}")
            return None

def main():
    """Основная функция для демонстрации работы системы"""
    print("=== Комплексная система генерации контента ===")
    print("Локальная генерация с использованием трансформерных моделей")
    print("=" * 60)
    
    # Инициализация генератора
    generator = LocalContentGenerator()
    
    # Генерация полного отчета
    print("🔄 Запуск процесса генерации контента...")
    report = generator.generate_complete_report()
    
    # Сохранение отчета
    filename = generator.save_report(report)
    
    # Вывод результатов
    print("\n✅ Процесс завершен успешно!")
    print(f"📄 Отчет сохранен: {filename}")
    print(f"📝 Тема: {report['topic_selection']['selected_topic']}")
    print(f"📊 Статус валидации: {report['content_validation']['overall_status']}")
    
    # Детали валидации
    validation = report['content_validation']
    print(f"📏 Длина статьи: {validation['length_validation']['details']['article_length']} символов")
    print(f"🔑 Покрытие ключевых слов: {validation['keywords_validation']['details']['coverage_percentage']}%")
    print(f"😊 Тональность: {validation['tone_validation']['details']['detected_tone']}")
    
    # Вывод сгенерированного контента
    content = report['generated_content']
    print(f"\n🎯 Заголовок: {content['title']}")
    print(f"📋 Meta Description: {content['meta_description']}")
    print(f"🏷️ Теги: {', '.join(content['tags'])}")
    
    print(f"\n📖 Статья: {content['article']}...")
    
    return report

if __name__ == "__main__":
    # Запуск системы
    report = main()