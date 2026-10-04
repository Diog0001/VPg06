"""
Telegram-помощник: Haystack Agent + PineconeDocumentStore + PyTelegramBotAPI.
"""

from __future__ import annotations

import atexit
import logging
import msvcrt
import os
import re
import sys
import textwrap
from collections import defaultdict
from typing import Any, TextIO

import telebot
from haystack.dataclasses import ChatMessage
from telebot import types

from hay.assistant import PersonalAssistant, format_memory_block
from hay.config import PROJECT_ROOT, load_project_env
from hay.haystack_memory import MEMORY_COSINE_SIMILARITY_THRESHOLD, HaystackUserMemory

load_project_env()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
MEMORY_NAMESPACE = os.getenv("PINECONE_MEMORY_NAMESPACE", "telegram-users")
MEMORY_RETRIEVAL_TOP_K = int(os.getenv("MEMORY_RETRIEVAL_TOP_K", "6"))
MAX_SESSION_MESSAGES = int(os.getenv("CHAT_SESSION_MAX_MESSAGES", "20"))

MEMORY_KINDS_TO_STORE = frozenset({"user_message", "assistant_message", "profile"})
MEMORY_KINDS_FOR_CONTEXT = frozenset({"user_message", "assistant_message", "profile"})

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("Задайте TELEGRAM_BOT_TOKEN в .env")

memory = HaystackUserMemory(namespace=MEMORY_NAMESPACE)
assistant = PersonalAssistant(memory, chat_model=OPENAI_CHAT_MODEL)
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN, parse_mode="HTML")

_chat_sessions: dict[int, list[ChatMessage]] = defaultdict(list)
_BOT_LOCK_PATH = PROJECT_ROOT / ".bot.lock"
_bot_lock_file: TextIO | None = None

_DOG_CEO_URL_RE = re.compile(r"https://images\.dog\.ceo/[^\s<]+")


def acquire_single_instance_lock() -> None:
    """Не даёт запустить второй polling для того же токена (ошибка Telegram 409)."""
    global _bot_lock_file
    _bot_lock_file = open(_BOT_LOCK_PATH, "a+", encoding="utf-8")
    try:
        msvcrt.locking(_bot_lock_file.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        logger.error(
            "Уже запущен другой экземпляр бота. Остановите его и запустите снова."
        )
        sys.exit(1)

    _bot_lock_file.seek(0)
    _bot_lock_file.truncate()
    _bot_lock_file.write(str(os.getpid()))
    _bot_lock_file.flush()
    atexit.register(release_single_instance_lock)


def release_single_instance_lock() -> None:
    global _bot_lock_file
    if _bot_lock_file is None:
        return
    try:
        msvcrt.locking(_bot_lock_file.fileno(), msvcrt.LK_UNLCK, 1)
    except OSError:
        pass
    _bot_lock_file.close()
    _bot_lock_file = None


def display_user_name(user: types.User) -> str:
    parts = [part for part in (user.first_name, user.last_name) if part]
    if parts:
        return " ".join(parts)
    if user.username:
        return f"@{user.username}"
    return f"user_{user.id}"


def base_user_metadata(message: types.Message) -> dict[str, Any]:
    user = message.from_user
    if user is None:
        raise ValueError("Сообщение без from_user нельзя сохранить в память.")

    meta: dict[str, Any] = {
        "user_id": str(user.id),
        "chat_id": str(message.chat.id),
        "display_name": display_user_name(user),
    }
    if user.username:
        meta["username"] = user.username
    if user.first_name:
        meta["first_name"] = user.first_name
    if user.last_name:
        meta["last_name"] = user.last_name
    return meta


def format_memory_block_from_docs(user_id: int) -> str:
    docs = memory.retrieve_relevant(
        user_id,
        "имя город работа хобби предпочтения факты профиль пользователя",
        top_k=MEMORY_RETRIEVAL_TOP_K,
        kinds=MEMORY_KINDS_FOR_CONTEXT,
    )
    return format_memory_block(docs)


def save_to_memory(message: types.Message, text: str, kind: str) -> None:
    if kind not in MEMORY_KINDS_TO_STORE:
        return

    cleaned = text.strip()
    if not cleaned:
        return

    meta = base_user_metadata(message)
    meta["kind"] = kind
    result = memory.remember_message(
        cleaned,
        metadata=meta,
        user_id=message.from_user.id,
        on_duplicate="update",
    )
    logger.info(
        "memory store user_id=%s kind=%s action=%s similarity=%s memory_id=%s",
        message.from_user.id,
        kind,
        result.action,
        result.similarity,
        result.memory_id,
    )


def append_session(user_id: int, user_text: str, assistant_text: str) -> None:
    session = _chat_sessions[user_id]
    session.append(ChatMessage.from_user(user_text))
    session.append(ChatMessage.from_assistant(assistant_text))
    if len(session) > MAX_SESSION_MESSAGES:
        _chat_sessions[user_id] = session[-MAX_SESSION_MESSAGES:]


def send_long_message(chat_id: int, text: str) -> None:
    limit = 4000
    if len(text) <= limit:
        bot.send_message(chat_id, text)
        return
    for i in range(0, len(text), limit):
        bot.send_message(chat_id, text[i : i + limit])


def maybe_send_dog_photo(chat_id: int, reply_text: str) -> None:
    match = _DOG_CEO_URL_RE.search(reply_text)
    if not match:
        return
    url = match.group(0).rstrip(").,")
    try:
        bot.send_photo(chat_id, url, caption="Случайная собака с dog.ceo")
    except Exception:
        logger.exception("Не удалось отправить фото dog.ceo в chat_id=%s", chat_id)


@bot.message_handler(commands=["start", "help"])
def handle_start(message: types.Message) -> None:
    user = message.from_user
    greeting = textwrap.dedent(
        f"""
        Привет, {display_user_name(user)}!

        Я персональный помощник на Haystack: помню контекст в Pinecone (косинусное сходство)
        и веду диалог как живой собеседник.

        Могу по запросу:
        • рассказать случайный факт о собаках (внешний API);
        • прислать фото собаки и описать породу через OpenAI Vision.

        Команды:
        /memory — что я помню о тебе
        /forget — удалить твою память
        /reset — очистить текущую сессию диалога
        /help — справка

        Просто пиши — я отвечу с учётом памяти и истории чата.
        """
    ).strip()
    bot.send_message(message.chat.id, greeting)

    profile_bits = [f"Telegram id={user.id}", f"display_name={display_user_name(user)}"]
    if user.username:
        profile_bits.append(f"username=@{user.username}")
    save_to_memory(message, "Профиль: " + ", ".join(profile_bits), kind="profile")


@bot.message_handler(commands=["memory"])
def handle_memory(message: types.Message) -> None:
    bot.send_chat_action(message.chat.id, "typing")
    user_id = message.from_user.id
    block = format_memory_block_from_docs(user_id)
    count = memory.count_user_documents(user_id)
    footer = (
        f"\n\n<i>Порог дедупликации: {MEMORY_COSINE_SIMILARITY_THRESHOLD}</i>\n"
        f"<i>Namespace Pinecone: {MEMORY_NAMESPACE}</i>\n"
        f"<i>Записей пользователя: {count}</i>"
    )
    send_long_message(message.chat.id, block + footer)


@bot.message_handler(commands=["forget"])
def handle_forget(message: types.Message) -> None:
    user_id = message.from_user.id
    try:
        memory.delete_user_memory(user_id)
        _chat_sessions.pop(user_id, None)
        bot.send_message(
            message.chat.id,
            "Готово: вся сохранённая память о тебе удалена из Pinecone.",
        )
    except Exception:
        logger.exception("Ошибка удаления памяти user_id=%s", user_id)
        bot.send_message(
            message.chat.id,
            "Не удалось очистить память. Попробуй позже.",
        )


@bot.message_handler(commands=["reset"])
def handle_reset(message: types.Message) -> None:
    user_id = message.from_user.id
    _chat_sessions.pop(user_id, None)
    bot.send_message(
        message.chat.id,
        "Сессия диалога в памяти бота сброшена. Долговременная память в Pinecone сохранена.",
    )


@bot.message_handler(content_types=["text"])
def handle_text(message: types.Message) -> None:
    if not message.text or message.text.startswith("/"):
        return

    user_text = message.text.strip()
    if not user_text:
        return

    user_id = message.from_user.id
    bot.send_chat_action(message.chat.id, "typing")

    try:
        save_to_memory(message, user_text, kind="user_message")

        reply = assistant.generate_reply(
            user_id=user_id,
            user_text=user_text,
            session_messages=list(_chat_sessions[user_id]),
            memory_kinds=MEMORY_KINDS_FOR_CONTEXT,
            memory_top_k=MEMORY_RETRIEVAL_TOP_K,
        )

        save_to_memory(message, reply, kind="assistant_message")
        append_session(user_id, user_text, reply)

        send_long_message(message.chat.id, reply)
        maybe_send_dog_photo(message.chat.id, reply)
    except Exception:
        logger.exception("Ошибка обработки сообщения user_id=%s", user_id)
        bot.send_message(
            message.chat.id,
            "Произошла ошибка при обработке сообщения. Попробуй ещё раз через минуту.",
        )


def main() -> None:
    acquire_single_instance_lock()
    logger.info(
        "Бот запущен (Haystack Agent). Модель: %s, namespace: %s",
        OPENAI_CHAT_MODEL,
        MEMORY_NAMESPACE,
    )
    bot.infinity_polling(
        timeout=60,
        long_polling_timeout=60,
        skip_pending=True,
    )


if __name__ == "__main__":
    main()
