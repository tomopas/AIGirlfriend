from __future__ import annotations

import asyncio
import logging
import random
import time
from datetime import datetime
from typing import Awaitable, Callable

from .config import Config
from .services import GirlfriendService

log = logging.getLogger(__name__)

SendText = Callable[[str], Awaitable[None]]
SendPhoto = Callable[[bytes, str], Awaitable[None]]


def in_quiet_hours(now_hour: int, start: int | None, end: int | None) -> bool:
    if start is None or end is None:
        return False
    start, end = start % 24, end % 24
    if start == end:
        return False
    if start < end:
        return start <= now_hour < end
    return now_hour >= start or now_hour < end


async def proactive_loop(
    service: GirlfriendService,
    send_text: SendText,
    send_photo: SendPhoto,
    cfg: Config,
) -> None:
    log.info("proactive loop started (%s-%s h, pic prob %s)", cfg.proactive_min_hours, cfg.proactive_max_hours, cfg.proactive_image_prob)
    while True:
        delay = random.uniform(cfg.proactive_min_hours, cfg.proactive_max_hours) * 3600.0
        await asyncio.sleep(delay)
        try:
            # quiet hours: stay silent overnight by default if configured
            hour = datetime.now().hour
            if in_quiet_hours(hour, cfg.proactive_quiet_start, cfg.proactive_quiet_end):
                log.debug("proactive tick skipped (quiet hours)")
                continue
            # recency: don't ping if user was recently active
            try:
                last_ts = await service.memory.last_activity_ts()
            except Exception:
                last_ts = None
            if last_ts is not None:
                idle_min = (time.time() - last_ts) / 60.0
                if idle_min < cfg.proactive_min_idle_minutes:
                    log.debug("proactive tick skipped (active %.1f min ago)", idle_min)
                    continue
            roll = random.random()
            if roll < cfg.proactive_image_prob and cfg.image_workflow:
                image = await service.generate_image()
                caption = service.photo_caption()
                await send_photo(image, caption)
                try:
                    await service.memory.add_chat("assistant", f"[sent a photo] {caption}")
                except Exception:
                    pass
                log.info("sent proactive photo")
            else:
                message = await service.proactive_message()
                if message and message.strip():
                    await send_text(message)
                    log.info("sent proactive text")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("proactive tick failed: %s", exc)