"""Шаблон для реализации RAG по справке Яндекс Метрики."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi

load_dotenv()

# Цена в курсе: 0,1 ₽ за 1000 токенов (вход + выход).
PRICE_PER_1K_RUB = 0.1

# Путь до выгрузки документации Яндекс Метрики
DEFAULT_DOCS_DIR = Path('metrica-docs-snapshot/docs')

# Размер фрагмента при индексации (символы).
CHUNK_SIZE = 4000

# Токены для BM25 / метаданных.
TOKEN_RE = re.compile(r'[A-Za-zА-Яа-я0-9_]+')

# Бонус к BM25 за каждое пересечение токена запроса с path/header.
METADATA_BONUS = 2.0

# Системная инструкция для этапа Augmentation / Generation.
SYSTEM_INSTRUCTION = (
    'Ты помощник по документации Яндекс Метрики. '
    'Отвечай только на основе блока CONTEXT. '
    'Если в CONTEXT нет данных для ответа, честно скажи об этом, ничего не придумывай. '
    'В ответе указывай источники в виде path (например, path=ru/webvisor/info.md).'
)

# Служебные слова для очистки запроса перед BM25.
QUERY_STOPWORDS = frozenset(
    {
        'и',
        'в',
        'на',
        'по',
        'для',
        'а',
        'с',
        'к',
        'из',
        'не',
        'у',
        'или',
        'это',
        'что',
        'как',
        'где',
        'такое',
    }
)

# Тестовые вопросы
GOLDEN_QA = [
    {
        'id': 'q1',
        'question': 'Куда поставить код счётчика Яндекс Метрики?',
        'relevant_paths': [
            'ru/troubleshooting/install-tag.md',
            'ru/general/creating-counter.md',
            'ru/quick-start.md',
            'ru/code/install',
        ],
    },
    {
        'id': 'q2',
        'question': 'Как проверить, что счётчик установлен корректно?',
        'relevant_paths': [
            'ru/general/check-counter.md',
            'ru/troubleshooting/install-tag.md',
            'ru/quick-start.md',
        ],
    },
    {
        'id': 'q3',
        'question': 'Что такое Вебвизор?',
        'relevant_paths': ['ru/webvisor', 'ru/general/counter-webvisor.md'],
    },
    {
        'id': 'q4',
        'question': 'Как создать цель в Метрике?',
        'relevant_paths': [
            'ru/general/goal',
            'ru/general/choose-goal.md',
            'ru/simple-goal',
            'ru/integrated-goal',
            'ru/general/goals.md',
        ],
    },
    {
        'id': 'q5',
        'question': 'Почему в отчётах нет данных после установки счётчика?',
        'relevant_paths': [
            'ru/troubleshooting',
            'ru/general/check-counter.md',
            'ru/general/sources-qanda.md',
        ],
    },
]


def ask(question: str) -> dict:
    """
    Отвечает на вопрос по документации Яндекс Метрики.

    Возвращает словарь с ключами:
    - answer: str — текст ответа модели
    - sources: list[str] — список путей файлов источников (до 3)
    - cost_rub: float — стоимость запроса в рублях
    """
    fragments = _retrieve(question)
    sources = [frag['path'] for frag in fragments]
    system, user = _build_prompt(fragments, question)

    api_key = os.getenv('LLM_API_KEY', '')
    base_url = os.getenv('LLM_BASE_URL', 'https://llm.api.cloud.yandex.net/v1')
    model = os.getenv('LLM_MODEL', '')

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': user},
        ],
        temperature=0,
        max_tokens=1500,
        # Qwen/thinking-модели иначе тратят лимит на reasoning, а content пустой.
        extra_body={'chat_template_kwargs': {'enable_thinking': False}},
    )

    answer = (response.choices[0].message.content or '').strip()
    usage = response.usage
    prompt_tokens = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0
    cost_rub = round(
        (prompt_tokens + completion_tokens) / 1000 * PRICE_PER_1K_RUB,
        4,
    )

    return {
        'answer': answer,
        'sources': sources,
        'cost_rub': cost_rub,
    }


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    """Режет текст на фрагменты ~chunk_size: сначала по абзацам, длинные — по длине."""
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    chunks: list[str] = []
    current = ''

    for paragraph in paragraphs:
        if len(paragraph) > chunk_size:
            if current:
                chunks.append(current)
                current = ''
            for i in range(0, len(paragraph), chunk_size):
                part = paragraph[i : i + chunk_size]
                if i + chunk_size < len(paragraph):
                    chunks.append(part)
                else:
                    current = part  # хвост может склеиться со следующими абзацами
            continue

        if not current:
            current = paragraph
        elif len(current) + 2 + len(paragraph) <= chunk_size:
            current = f'{current}\n\n{paragraph}'
        else:
            chunks.append(current)
            current = paragraph

    if current:
        chunks.append(current)
    return chunks


def _indexing(docs_dir: Path) -> list[dict]:
    """Читает *.md и возвращает список фрагментов с path, header и text."""
    index: list[dict] = []
    for file in docs_dir.glob('**/*.md'):
        relative_path = file.relative_to(docs_dir).as_posix()
        text = file.read_text(encoding='utf-8')
        header = next(
            (
                line.removeprefix('#').strip()
                for line in text.splitlines()
                if line.strip().startswith('#')
            ),
            '',
        )
        for fragment in _chunk_text(text):
            index.append(
                {
                    'path': relative_path,
                    'header': header,
                    'text': fragment,
                }
            )
    return index


def _normalize(text: str) -> str:
    """Нормализует текст: ё -> е и нижний регистр."""
    return text.lower().replace('ё', 'е')


def _tokenize(text: str) -> list[str]:
    """Разбивает текст на токены по TOKEN_RE."""
    return TOKEN_RE.findall(_normalize(text))


def _prepare_query(question: str) -> list[str]:
    """Фильтрует токены запроса от стоп-слов; если пусто — исходные токены."""
    tokens = _tokenize(question)
    filtered = [t for t in tokens if t not in QUERY_STOPWORDS]
    return filtered or tokens


@dataclass
class BM25Index:
    """Хранит корпус фрагментов и индекс BM25."""
    corpus: list[dict]
    bm25: BM25Okapi

    @classmethod
    def build(cls, docs_dir: Path = DEFAULT_DOCS_DIR) -> BM25Index:
        corpus = _indexing(docs_dir)
        bm25 = BM25Okapi([_tokenize(doc['text']) for doc in corpus])
        return cls(corpus=corpus, bm25=bm25)


@cache
def _get_or_create_bm25_index() -> BM25Index:
    """Возвращает или создает индекс BM25."""
    return BM25Index.build()


def _metadata_bonus(query_tokens: set[str], path: str, header: str) -> float:
    """Начисляет бонус за каждый общий токен запроса с path/header."""
    meta_tokens = set(_tokenize(f'{path} {header}'))
    return METADATA_BONUS * len(query_tokens & meta_tokens)


def _retrieve(question: str, top_k: int = 3) -> list[dict]:
    """Ищет top-k фрагментов по BM25 + бонус path/header, с дедупом по путям."""
    index = _get_or_create_bm25_index()
    query_tokens = _prepare_query(question)
    query_set = set(query_tokens)
    scores = index.bm25.get_scores(query_tokens)

    ranked: list[tuple[float, int]] = []
    for i, doc in enumerate(index.corpus):
        bonus = _metadata_bonus(query_set, doc['path'], doc['header'])
        ranked.append((float(scores[i]) + bonus, i))

    ranked.sort(key=lambda item: item[0], reverse=True)

    results: list[dict] = []
    seen_paths: set[str] = set()
    for score, i in ranked:
        path = index.corpus[i]['path']
        if path in seen_paths:
            continue
        seen_paths.add(path)
        results.append(
            {
                'path': path,
                'header': index.corpus[i]['header'],
                'text': index.corpus[i]['text'],
                'score': score,
            }
        )
        if len(results) >= top_k:
            break
    return results


def _build_prompt(fragments: list[dict], question: str) -> tuple[str, str]:
    """Собирает system-инструкцию и user-промпт (CONTEXT + QUESTION)."""
    if fragments:
        context_blocks = [
            f'[{i}] path={frag["path"]}\n{frag["text"]}'
            for i, frag in enumerate(fragments, start=1)
        ]
        context = '\n\n'.join(context_blocks)
    else:
        context = '(контекст пуст)'

    user_prompt = f'CONTEXT:\n{context}\n\nQUESTION:\n{question}'
    return SYSTEM_INSTRUCTION, user_prompt
