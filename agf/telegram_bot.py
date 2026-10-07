from __future__ import annotations

import asyncio
import logging

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


class TelegramBot:
    def __init__(self, cfg: Config, service: GirlfriendService):
        self.cfg = cfg
        self.service = service
        self._chat_locks: dict[int, asyncio.Lock] = {}
        self._last_handled: dict[int, float] = {}

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
        text = text.strip() if text else ""
        if not text:
            text = "…"
        for index in range(0, len(text), _TEXT_LIMIT):
            chunk = text[index : index + _TEXT_LIMIT]
            if chunk:
                await app.bot.send_message(chat_id=chat_id, text=chunk)

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat = update.effective_chat
        message = update.effective_message
        if chat is None or message is None:
            return
        text = (message.text or message.caption or "").strip()
        if not text:
            return  # ignore stickers/voice/photos for now
        if not self._is_allowed(chat.id):
            log.warning("ignoring message from chat id %s", chat.id)
            try:
                await message.reply_text("sorry, I only talk to my partner ♡")
            except Exception:
                pass
            return
        # simple per-chat rate limit: drop messages arriving <1.5s after previous
        import time as _time

        now = _time.monotonic()
        last = self._last_handled.get(chat.id, 0.0)
        if now - last < 1.5:
            log.debug("dropping rapid message from %s", chat.id)
            return
        self._last_handled[chat.id] = now
        lock = self._lock_for(chat.id)
        # serialize slow LLM calls per chat; second message waits instead of overlapping
        if lock.locked():
            await message.reply_text("one sec babe, typing… ♡")
        async with lock:
            typing_task = asyncio.create_task(self._keep_typing(chat.id, context.application))
            try:
                reply = await self.service.reply(text)
            except Exception as exc:
                log.warning("reply failed: %s", exc)
                await self._stop_typing(typing_task)
                await message.reply_text("ugh, my brain lagged… try again? ♡")
                return
            await self._stop_typing(typing_task)
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
        await self.service.memory.add_memory(text, kind="pref", importance=0.8)
        await message.reply_text(f"got it — wrote it down \u2661 (\"{text[:60]}{'...' if len(text) > 60 else ''}\")")

    async def cmd_memories(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.effective_message
        if message is None or not self._is_allowed(update.effective_chat.id if update.effective_chat else -1):
            return
        memories = await self.service.memory.list_memories(limit=50)
        if not memories:
            await message.reply_text("i don't remember anything yet. tell me things \u2661")
            return
        lines = [
            f"{m['id']}. [{m['kind']}] {m['text']}" for m in memories
        ]
        await self._send_chunks(context.application, update.effective_chat.id, "\n".join(lines))

    async def cmd_forget(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.effective_message
        if message is None or not self._is_allowed(update.effective_chat.id if update.effective_chat else -1):
            return
        args = context.args or []
        if args and args[0].isdigit():
            await self.service.memory.delete_memory(int(args[0]))
            try:
                profile = await self.service.memory.recall_profile()
                # best-effort: profile keys are free-form, nothing to drop by id
                _ = profile
            except Exception:
                pass
            await message.reply_text(f"forgot #{args[0]} ♡")
            return
        if args and args[0].lower() in ("all", "everything"):
            await self.service.memory.forget_all()
            await self.service.memory.clear_chat()
            await message.reply_text("okay… wiped it all. feels like we just met \U0001F622")
            return
        await message.reply_text("usage: /forget <id> — or /forget all to wipe everything")

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
            "hey babe \u2661 I'm here. You can just chat with me, ask me for a photo, "
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
            BotCommand("status", "check ollama / comfyui health"),
        ]
        await app.bot.set_my_commands(commands)

    def build(self) -> Application:
        if not self.cfg.telegram_token:
            raise RuntimeError("TELEGRAM_BOT_TOKEN not set — copy .env.example to .env and fill it in")
        app = ApplicationBuilder().token(self.cfg.telegram_token).post_init(self._set_commands).build()

        app.add_handler(CommandHandler("start", self.cmd_start))
        app.add_handler(CommandHandler("photo", self.cmd_photo))
        app.add_handler(CommandHandler("video", self.cmd_video))
        app.add_handler(CommandHandler("remember", self.cmd_remember))
        app.add_handler(CommandHandler("memories", self.cmd_memories))
        app.add_handler(CommandHandler("forget", self.cmd_forget))
        app.add_handler(CommandHandler("status", self.cmd_status))
        app.add_handler(MessageHandler((filters.TEXT | filters.CAPTION) & ~filters.COMMAND, self.handle_message))
        return app