"""
Инструменты агента: внешние API про собак и описание породы по фото (OpenAI Vision).
"""

from __future__ import annotations

import logging

import requests
from openai import OpenAI

logger = logging.getLogger(__name__)

DOG_FACT_URL = "https://dog-api.kinduff.com/api/facts"
DOG_IMAGE_URL = "https://dog.ceo/api/breeds/image/random"


def fetch_random_dog_fact() -> str:
    """Случайный факт о собаках (dog-api.kinduff.com)."""
    response = requests.get(DOG_FACT_URL, timeout=20)
    response.raise_for_status()
    payload = response.json()
    facts = payload.get("facts") or []
    if not facts:
        return "Не удалось получить факт о собаках — попробуйте вызвать инструмент ещё раз."
    return str(facts[0]).strip()


def fetch_and_describe_random_dog_image(
    openai_client: OpenAI,
    *,
    vision_model: str,
) -> str:
    """Случайное фото с dog.ceo и описание породы через мультимодальную модель OpenAI."""
    img_response = requests.get(DOG_IMAGE_URL, timeout=20)
    img_response.raise_for_status()
    image_url = str(img_response.json().get("message") or "").strip()
    if not image_url:
        return "Не удалось получить URL изображения собаки."

    completion = openai_client.chat.completions.create(
        model=vision_model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "На фото собака. Определи породу или наиболее вероятный тип. "
                            "Ответь по-русски структурированно: "
                            "1) название породы; 2) краткие черты; "
                            "3) откуда порода и как сформировалась (2–4 предложения). "
                            "Если породу точно определить нельзя — так и скажи и опиши внешность."
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        max_tokens=600,
        temperature=0.4,
    )
    description = (completion.choices[0].message.content or "").strip()
    return f"Ссылка на фото: {image_url}\n\n{description}"
