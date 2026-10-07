from __future__ import annotations

import asyncio
import logging
import time

from telegram import BotCommand, Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from .config import Config
from .services import GirlfriendService

log = logging.getLogger(__name__)

_TEXT_LIMIT = 4096


def split_text(text: str, limit: int = _TEXT_LIMIT) -> list[str]:
    """Split long text on newlines/spaces so we never cut mid-word or mid-emoji run."""
    text = text.strip() if text else ""
    if not text:
        return ["…"]
    chunks: list[str] = []
    rest = text
    while len(rest) > limit:
        window = rest[:limit]
        # Prefer a newline, then a space, within the window.
        cut = window.rfind("\n")
        if cut < limit // 2:
            cut = window.rfind(" ")
        if cut <= 0:
            cut = limit
        else:
            # Avoid splitting a ZWJ / variation-selector emoji sequence.
            while cut > 0 and rest[cut - 1] in ("\u200d", "\ufe0f"):
                cut -= 1
            if cut <= 0:
                cut = limit
        chunks.append(rest[:cut].rstrip())
        rest = rest[cut:].lstrip()
    if rest:
        chunks.append(rest)
    return chunks


class TelegramBot:
    def __init__(self, cfg: Config, service: GirlfriendService):
        self.cfg = cfg
        self.service = service
        self._chat_locks: dict[int, asyncio.Lock] = {}
        self._last_handled: dict[int, float] = {}
        self._pending: dict[int, list[str]] = {}

    def _lock_for(self, chat_id: int) -> asyncio.Lock:
        lock = self._chat_locks.get(chat_id)
        if lock is None:
            lock = asyncio.Lock()
            self._chat_locks[chat_id] = lock
        return lock

    def _is_allowed(self, chat_id: int) -> bool:
        if self.cfg.allows_everyone:
            return True
        return chat_id in self.cfg.allowed_chat_ids

    async def _keep_typing(self, chat_id: int, app: Application) -> None:
        try:
            while True:
                try:
                    await app.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
                except Exception:
                    pass
                await asyncio.sleep(4.5)
        except asyncio.CancelledError:
            raise

    @staticmethod
    async def _stop_typing(task: asyncio.Task) -> None:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

    async def _send_chunks(self, app: Application, chat_id: int, text: str) -> None:
        for chunk in split_text(text):
            if chunk:
                await app.bot.send_message(chat_id=chat_id, text=chunk)

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat = update.effective_chat
        message = update.effective_message
        if chat is None or message is None:
            return
        if not self._is_allowed(chat.id):
            # Silent-ignore strangers: don't confirm the bot exists.
            log.warning("ignoring message from chat id %s", chat.id)
            return
        text = (message.text or message.caption or "").strip()
        if not text:
            # Graceful fallback for non-text content instead of silent drop.
            if getattr(message, "voice", None) or getattr(message, "audio", None):
                await message.reply_text("ooh a voice note… i can't listen yet, type it for me? ♡")
            elif getattr(message, "photo", None):
                await message.reply_text("cute pic! tell me about it in words too? ♡")
            elif getattr(message, "sticker", None) or getattr(message, "video_note", None):
                await message.reply_text("hehe ♡ tell me in words?")
            return
        now = time.monotonic()
        last = self._last_handled.get(chat.id, 0.0)
        lock = self._lock_for(chat.id)
        # Coalesce rapid bursts instead of dropping: buffer while busy/recent.
        if lock.locked() or (now - last < 1.5 and self._pending.get(chat.id)):
            self._pending.setdefault(chat.id, []).append(text)
            return
        if now - last < 1.5:
            # First rapid message: buffer briefly so the tail of a burst joins.
            self._pending.setdefault(chat.id, []).append(text)
            await asyncio.sleep(1.5)
            buffered = self._pending.pop(chat.id, [])
            text = "\n".join(buffered) if buffered else text
        else:
            # Drain any stale buffer together with this message.
            buffered = self._pending.pop(chat.id, [])
            if buffered:
                text = "\n".join([*buffered, text])
        self._last_handled[chat.id] = time.monotonic()
        if lock.locked():
            await message.reply_text("one sec babe, typing… ♡")
        async with lock:
            # Opportunistically merge messages that arrived while waiting.
            await asyncio.sleep(0.6)
            extra = self._pending.pop(chat.id, [])
            if extra:
                text = "\n".join([text, *extra])
            typing_task = asyncio.create_task(self._keep_typing(chat.id, context.application))
            # Streaming: show a live-typing placeholder edited in place.
            placeholder = None
            full = ""
            last_edit = 0.0

            async def _on_chunk(chunk: str) -> None:
                nonlocal full, last_edit, placeholder
                full += chunk
                now_edit = time.monotonic()
                if placeholder is not None and now_edit - last_edit > 2.0 and len(full) % 40 < 8:
                    try:
                        await placeholder.edit_text(full[: _TEXT_LIMIT - 20] + "…")
                        last_edit = now_edit
                    except Exception:
                        pass

            try:
                try:
                    placeholder = await message.reply_text("typing… ♡")
                except Exception:
                    placeholder = None
                reply = await self.service.reply_stream(text, on_chunk=_on_chunk)
            except Exception as exc:
                log.warning("reply failed: %s", exc)
                await self._stop_typing(typing_task)
                await message.reply_text("ugh, my brain lagged… try again? ♡")
                return
            await self._stop_typing(typing_task)
            try:
                if placeholder is not None:
                    await placeholder.delete()
            except Exception:
                pass
        await self._send_chunks(context.application, chat.id, reply.text)
        if reply.media:
            try:
                if reply.media_type == "video":
                    await context.application.bot.send_video(
                        chat_id=chat.id, video=reply.media, caption=reply.caption
                    )
                else:
                    await context.application.bot.send_photo(
                        chat_id=chat.id, photo=reply.media, caption=reply.caption
                    )
            except Exception as exc:
                log.warning("failed to send media: %s", exc)

    async def cmd_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat = update.effective_chat
        message = update.effective_message
        if chat is None or message is None or not self._is_allowed(chat.id):
            return
        subject = " ".join(context.args) if context.args else None
        typing_task = asyncio.create_task(self._keep_typing(chat.id, context.application))
        try:
            try:
                image = await self.service.generate_image(subject=subject)
            except Exception as exc:
                await message.reply_text(f"couldn't generate: {exc}")
                return
            await message.reply_photo(photo=image, caption=self.service.photo_caption())
        finally:
            await self._stop_typing(typing_task)

    async def cmd_video(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat = update.effective_chat
        message = update.effective_message
        if chat is None or message is None or not self._is_allowed(chat.id):
            return
        if not self.cfg.video_workflow:
            await message.reply_text("no video workflow configured.")
            return
        subject = " ".join(context.args) if context.args else None
        typing_task = asyncio.create_task(self._keep_typing(chat.id, context.application))
        try:
            try:
                video = await self.service.generate_video(subject=subject)
            except Exception as exc:
                await message.reply_text(f"couldn't generate: {exc}")
                return
            await message.reply_video(video=video, caption=self.service.photo_caption())
        finally:
            await self._stop_typing(typing_task)

    async def cmd_remember(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.effective_message
        if message is None or not self._is_allowed(update.effective_chat.id if update.effective_chat else -1):
            return
        text = " ".join(context.args) if context.args else ""
        if not text:
            await message.reply_text("usage: /remember <something about you>")
            return
        text = text.strip()[:500]
        await self.service.memory.add_memory(text, kind="pref", importance=0.8)
        await message.reply_text(f"got it — wrote it down ♡ (\"{text[:60]}{'...' if len(text) > 60 else ''}\")")

    async def cmd_memories(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.effective_message
        if message is None or not self._is_allowed(update.effective_chat.id if update.effective_chat else -1):
            return
        args = context.args or []
        page = 1
        if args and args[0].isdigit():
            page = max(1, int(args[0]))
        per_page = 10
        offset = (page - 1) * per_page
        total = await self.service.memory.count_memories()
        memories = await self.service.memory.list_memories(limit=per_page, offset=offset)
        if not memories:
            await message.reply_text("i don't remember anything yet. tell me things ♡")
            return
        lines = [
            f"{m['id']}. [{m['kind']}] {m['text']}" for m in memories
        ]
        footer = f"\n— page {page} of {max(1, (total + per_page - 1) // per_page)} (/memories {page + 1} for more)"
        await self._send_chunks(context.application, update.effective_chat.id, "\n".join(lines) + footer)

    async def cmd_forget(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.effective_message
        if message is None or not self._is_allowed(update.effective_chat.id if update.effective_chat else -1):
            return
        args = context.args or []
        if args and args[0].isdigit():
            await self.service.memory.delete_memory(int(args[0]))
            await message.reply_text(f"forgot #{args[0]} ♡")
            return
        if args and args[0].lower() in ("all", "everything"):
            await self.service.memory.forget_all()
            await self.service.memory.clear_profile()
            await self.service.memory.clear_chat()
            await message.reply_text("okay… wiped it all. feels like we just met \U0001F622")
            return
        if args and args[0].lower() in ("profile", "prefs", "kinks"):
            await self.service.memory.clear_profile()
            await message.reply_text("cleared what i knew about your tastes ♡")
            return
        await message.reply_text("usage: /forget <id> — /forget profile — or /forget all to wipe everything")

    async def cmd_clear(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.effective_message
        if message is None or not self._is_allowed(update.effective_chat.id if update.effective_chat else -1):
            return
        await self.service.memory.clear_chat()
        await message.reply_text("fresh start, just us ♡")

    async def cmd_model(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.effective_message
        if message is None or not self._is_allowed(update.effective_chat.id if update.effective_chat else -1):
            return
        status = await self.service.status()
        models = ", ".join(status["ollama_models"][:10]) or "none"
        await message.reply_text(f"current: {self.cfg.llm_model}\nlocal: {models}")

    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.effective_message
        if message is None or not self._is_allowed(update.effective_chat.id if update.effective_chat else -1):
            return
        await message.reply_text(
            "just chat with me ♡\n"
            "/photo <idea> — pic of me\n"
            "/video <idea> — clip of me\n"
            "/remember <fact> — keep it forever\n"
            "/memories [page] — what i remember\n"
            "/forget <id>|profile|all\n"
            "/clear — clear chat history\n"
            "/model — llm info\n"
            "/status — service health"
        )

    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.effective_message
        if message is None or not self._is_allowed(update.effective_chat.id if update.effective_chat else -1):
            return
        status = await self.service.status()
        models = ", ".join(status["ollama_models"]) or "none"
        text = (
            f"model: {self.cfg.llm_model} ({models})\n"
            f"comfyui: {'reachable' if status['comfyui_reachable'] else 'NOT reachable'}\n"
            f"image workflow: {status['image_workflow'] or 'not set'}\n"
            f"video workflow: {status['video_workflow'] or 'not set'}\n"
            f"nsfw: {status['nsfw']}"
        )
        await message.reply_text(text)

    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.effective_message
        if message is None or not self._is_allowed(update.effective_chat.id if update.effective_chat else -1):
            return
        await message.reply_text(
            "hey babe ♡ I'm here. You can just chat with me, ask me for a photo, "
            "or whisper a /remember so I keep it forever. x"
        )

    async def _set_commands(self, app: Application) -> None:
        commands = [
            BotCommand("start", "say hi"),
            BotCommand("photo", "ask for a photo"),
            BotCommand("video", "ask for a video"),
            BotCommand("remember", "store a fact about you"),
            BotCommand("memories", "list what she remembers"),
            BotCommand("forget", "wipe her memory"),
            BotCommand("clear", "clear chat history"),
            BotCommand("model", "llm info"),
            BotCommand("status", "check ollama / comfyui health"),
            BotCommand("help", "what can she do"),
        ]
        await app.bot.set_my_commands(commands)

    async def _on_error(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        log.warning("telegram handler error: %s", context.error)

    def build(self) -> Application:
        if not self.cfg.telegram_token:
            raise RuntimeError("TELEGRAM_BOT_TOKEN not set — copy .env.example to .env and fill it in")
        app = ApplicationBuilder().token(self.cfg.telegram_token).post_init(self._set_commands).build()

        app.add_handler(CommandHandler("start", self.cmd_start))
        app.add_handler(CommandHandler("help", self.cmd_help))
        app.add_handler(CommandHandler("photo", self.cmd_photo))
        app.add_handler(CommandHandler("video", self.cmd_video))
        app.add_handler(CommandHandler("remember", self.cmd_remember))
        app.add_handler(CommandHandler("memories", self.cmd_memories))
        app.add_handler(CommandHandler("forget", self.cmd_forget))
        app.add_handler(CommandHandler("clear", self.cmd_clear))
        app.add_handler(CommandHandler("model", self.cmd_model))
        app.add_handler(CommandHandler("status", self.cmd_status))
        app.add_handler(MessageHandler((filters.TEXT | filters.CAPTION) & ~filters.COMMAND, self.handle_message))
        # Catch stickers/voice/photos so handle_message can reply gracefully.
        app.add_handler(MessageHandler(filters.ATTACHMENT & ~filters.COMMAND, self.handle_message))
        app.add_error_handler(self._on_error)
        return app
