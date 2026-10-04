"""
Haystack Agent — персональный помощник с инструментами и контекстом из Pinecone.
"""

from __future__ import annotations

import os
from typing import Any

from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage, Document
from haystack.tools import create_tool_from_function
from openai import OpenAI

from hay.dog_tools import fetch_and_describe_random_dog_image, fetch_random_dog_fact
from hay.haystack_memory import HaystackUserMemory

AGENT_SYSTEM_PROMPT = """\
Ты дружелюбный персональный помощник в Telegram.
Отвечай по-русски, естественно продолжай диалог и опирайся на историю сообщений в этом чате.
В блоке «Память о пользователе» — факты из прошлых разговоров (векторный поиск, косинусное сходство).
Не выдумывай факты о пользователе, которых нет в памяти. Если данных мало — уточни вопрос.

Инструменты (используй по запросу):
- random_dog_fact — случайный факт о собаках из внешнего API;
- random_dog_breed_from_image — случайное фото собаки, распознавание породы и краткая история породы.

Если пользователь просит факт или картинку про собак — вызывай соответствующий инструмент.
"""


def format_memory_block(documents: list[Document]) -> str:
    if not documents:
        return "Память о пользователе: (пока пусто)"

    lines = ["Память о пользователе:"]
    seen: set[str] = set()
    for doc in documents:
        meta = doc.meta or {}
        kind = str(meta.get("kind") or "fact")
        text = (doc.content or meta.get("text") or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        score = doc.score
        suffix = f" (релевантность {score:.2f})" if score is not None else ""
        lines.append(f"- [{kind}]{suffix} {text}")

    if len(lines) == 1:
        return "Память о пользователе: (пока пусто)"
    return "\n".join(lines)


def build_dog_tools(openai_client: OpenAI, vision_model: str) -> list:
    def random_dog_fact() -> str:
        """Случайный факт о собаках с бесплатного API dog-api.kinduff.com."""
        return fetch_random_dog_fact()

    def random_dog_breed_from_image() -> str:
        """Случайное фото с dog.ceo, порода и история породы через OpenAI Vision."""
        return fetch_and_describe_random_dog_image(
            openai_client,
            vision_model=vision_model,
        )

    return [
        create_tool_from_function(random_dog_fact),
        create_tool_from_function(random_dog_breed_from_image),
    ]


class PersonalAssistant:
    def __init__(
        self,
        memory: HaystackUserMemory,
        *,
        chat_model: str | None = None,
        openai_api_key: str | None = None,
        openai_base_url: str | None = None,
    ) -> None:
        self.memory = memory
        self.chat_model = chat_model or os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Нужен OPENAI_API_KEY для агента и vision-инструмента.")

        openai_kwargs: dict[str, Any] = {"api_key": api_key}
        base_url = openai_base_url or os.getenv("OPENAI_BASE_URL")
        if base_url:
            openai_kwargs["base_url"] = base_url
        self._openai = OpenAI(**openai_kwargs)

        tools = build_dog_tools(self._openai, vision_model=self.chat_model)
        self._agent = Agent(
            chat_generator=OpenAIChatGenerator(model=self.chat_model),
            system_prompt=AGENT_SYSTEM_PROMPT,
            tools=tools,
            exit_conditions=["text"],
        )

    def generate_reply(
        self,
        *,
        user_id: int,
        user_text: str,
        session_messages: list[ChatMessage],
        memory_kinds: frozenset[str],
        memory_top_k: int,
    ) -> str:
        memory_docs = self.memory.retrieve_relevant(
            user_id,
            user_text,
            top_k=memory_top_k,
            kinds=memory_kinds,
        )
        memory_block = format_memory_block(memory_docs)

        messages = [
            ChatMessage.from_system(memory_block),
            *session_messages,
            ChatMessage.from_user(user_text),
        ]

        result = self._agent.run(messages=messages)
        last = result.get("last_message")
        text = (last.text if last else "") or ""
        return text.strip() or "Не удалось сформировать ответ. Попробуйте переформулировать."
