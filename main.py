from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from agf.comfy_client import ComfyClient
from agf.config import Config
from agf.memory import MemoryStore
from agf.ollama_client import OllamaClient
from agf.services import GirlfriendService
from agf.telegram_bot import TelegramBot
from agf import proactive

log = logging.getLogger("aigirlfriend")


def setup_logging(verbose: bool, log_file: str | None = None) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(RotatingFileHandler(log_file, maxBytes=1_000_000, backupCount=3))
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("telegram").setLevel(logging.INFO)


async def check(cfg: Config) -> int:
    ollama = OllamaClient(cfg.ollama_url, timeout=10, default_model=cfg.llm_model)
    comfy = ComfyClient(cfg.comfyui_url, timeout=10)
    ok = True
    try:
        models = await ollama.tags()
        print(f"ollama: OK ({len(models)} models: {', '.join(models[:8])}{'…' if len(models) > 8 else ''})")
        if cfg.llm_model not in models and not any(cfg.llm_model in m for m in models):
            print(f"  ! configured model '{cfg.llm_model}' not in the list above — run: ollama pull {cfg.llm_model}")
    except Exception as exc:
        ok = False
        print(f"ollama: FAIL — {exc}")
    if await comfy.health():
        print("comfyui: OK")
    else:
        ok = False
        print("comfyui: FAIL — not reachable; start ComfyUI with --listen")
    for label, path in (("image", cfg.image_workflow), ("video", cfg.video_workflow)):
        if path:
            problems = ComfyClient.validate_workflow(path)
            try:
                graph = ComfyClient.load_graph(path)
                try:
                    server_problems = await comfy.validate_against_server(graph)
                except Exception:
                    server_problems = []
                problems = problems + server_problems
                if not problems:
                    print(f"{label} workflow {path}: OK ({len(graph)} nodes)")
                else:
                    print(f"{label} workflow {path}: OK with warnings: {'; '.join(problems)}")
                    if any("asks for" in p for p in server_problems):
                        ok = False
            except Exception as exc:
                ok = False
                print(f"{label} workflow {path}: FAIL — {exc}")
        else:
            print(f"{label} workflow: not configured")
    for problem in cfg.validate():
        print(f"config: {problem}")
        if "not set" in problem and "TELEGRAM" in problem:
            ok = False
    if not Path(cfg.persona_path).exists():
        print(f"persona: WARN — {cfg.persona_path} missing, fallback persona in use")
    await ollama.close()
    await comfy.close()
    return 0 if ok else 1


async def run(cfg: Config) -> None:
    if not cfg.allowed_chat_ids and not cfg.allows_everyone:
        log.warning("ALLOWED_CHAT_ID is not set — the bot will refuse every chat. Set it in .env.")
    if not cfg.allowed_chat_ids and cfg.allows_everyone:
        log.warning("ALLOWED_CHAT_ID is '*' — the bot talks to ANY chat. For testing only.")
    for problem in cfg.validate():
        log.warning("config: %s", problem)
    for label, path in (("image", cfg.image_workflow), ("video", cfg.video_workflow)):
        if path:
            for problem in ComfyClient.validate_workflow(path):
                log.warning("%s workflow %s: %s", label, path, problem)

    ollama = OllamaClient(cfg.ollama_url, default_model=cfg.llm_model)
    comfy = ComfyClient(cfg.comfyui_url)
    try:
        models = await ollama.tags()
        if not any(cfg.llm_model in m for m in models):
            log.warning("model '%s' not found locally — run `ollama pull %s`", cfg.llm_model, cfg.llm_model)
    except Exception as exc:
        log.warning("could not reach Ollama at %s: %s", cfg.ollama_url, exc)

    async def _embed_fn(text: str):
        return await ollama.embed(text, model=cfg.embed_model)

    memory = MemoryStore(cfg.db_path, embed_fn=_embed_fn)
    # Best-effort online backup at startup (protects the single sqlite file).
    try:
        await memory.backup_to(Path(cfg.data_dir) / "memory.db.bak")
    except Exception as exc:
        log.debug("startup backup failed: %s", exc)

    service = GirlfriendService(cfg, ollama, comfy, memory)
    bot = TelegramBot(cfg, service)
    app = bot.build()

    targets = list(cfg.allowed_chat_ids)
    stop = asyncio.Event()

    def _request_stop(*_args) -> None:
        stop.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _request_stop)
        except (NotImplementedError, RuntimeError):
            pass

    async def send_text(text: str) -> None:
        if not targets:
            log.debug("proactive text skipped: no allowed chat ids (use ALLOWED_CHAT_ID)")
            return
        for cid in targets:
            try:
                await app.bot.send_message(chat_id=cid, text=text)
            except Exception as exc:
                log.warning("proactive send to %s failed: %s", cid, exc)

    async def send_photo(photo: bytes, caption: str) -> None:
        if not targets:
            log.debug("proactive photo skipped: no allowed chat ids")
            return
        for cid in targets:
            try:
                await app.bot.send_photo(chat_id=cid, photo=photo, caption=caption)
            except Exception as exc:
                log.warning("proactive photo to %s failed: %s", cid, exc)

    log.info("starting… (ctrl+c to stop)")
    async with app:
        await app.initialize()
        await app.updater.start_polling()
        await app.start()
        proactive_task = asyncio.create_task(proactive.proactive_loop(service, send_text, send_photo, cfg))
        try:
            await stop.wait()
        finally:
            proactive_task.cancel()
            try:
                await proactive_task
            except (asyncio.CancelledError, Exception):
                pass
            try:
                await service.shutdown()
            except Exception:
                pass
            try:
                await app.updater.stop()
                await app.stop()
                await app.shutdown()
            except Exception:
                pass
    memory.close()
    await ollama.close()
    await comfy.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Girlfriend — Ollama + ComfyUI + Telegram")
    parser.add_argument("--check", action="store_true", help="health-check services and exit")
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    parser.add_argument("--log-file", default=None, help="also log to a rotating file (e.g. logs/bot.log)")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    args = parser.parse_args()

    setup_logging(args.verbose, log_file=args.log_file)
    cfg = Config.load(config_path=args.config)

    if args.check:
        sys.exit(asyncio.run(check(cfg)))
    try:
        asyncio.run(run(cfg))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
