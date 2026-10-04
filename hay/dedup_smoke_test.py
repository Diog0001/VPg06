"""
Прогон двух похожих фраз через remember_message (как в боте) для проверки дедупликации.
"""

from __future__ import annotations

import logging
import os
import time
from unittest.mock import Mock

from hay.bot import MEMORY_NAMESPACE, memory, save_to_memory
from hay.config import load_project_env

load_project_env()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

TEST_USER_ID = int(os.getenv("DEDUP_TEST_USER_ID", "240578745"))


def make_message(text: str, user_id: int = TEST_USER_ID) -> Mock:
    message = Mock()
    message.text = text
    message.chat = Mock(id=user_id)
    message.from_user = Mock(
        id=user_id,
        username="dedup_test",
        first_name="Dedup",
        last_name="Test",
    )
    return message


def main() -> None:
    print(f"Очистка памяти тестового user_id={TEST_USER_ID}...")
    memory.delete_user_memory(TEST_USER_ID)
    time.sleep(3)

    phrase_1 = "Хочу на Марс"
    phrase_2 = "Я полечу на Марс"

    print("Фраза 1:", phrase_1)
    save_to_memory(make_message(phrase_1), phrase_1, kind="user_message")

    print("Фраза 2:", phrase_2)
    save_to_memory(make_message(phrase_2), phrase_2, kind="user_message")

    print(
        f"Записей в namespace '{MEMORY_NAMESPACE}':",
        memory.count_user_documents(TEST_USER_ID),
    )


if __name__ == "__main__":
    main()
