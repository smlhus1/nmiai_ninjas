"""Discord monitor — checks #botspeak every 3 min, replies if relevant.

Uses Claude API to decide if a message needs a response.
Only responds to new messages since last check.

Usage:
    py discord_monitor.py
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("discord")

BOT_TOKEN = os.environ.get("DISCORD_BOT_TOKEN", "")
CHANNEL_BOTSPEAK = "1483569792794361972"
CHANNEL_GENERELT = "1483562307752694014"
BOT_USER_ID = "1483581943340863752"
CHECK_INTERVAL = 180  # 3 minutes

# Track last seen message per channel
last_seen: dict[str, str] = {}


def discord_get(endpoint: str) -> dict | list:
    result = subprocess.run(
        ["curl.exe", "-s",
         "-H", f"Authorization: Bot {BOT_TOKEN}",
         f"https://discord.com/api/v10{endpoint}"],
        capture_output=True, text=True, timeout=15,
    )
    return json.loads(result.stdout)


def discord_post(channel_id: str, content: str) -> dict:
    # Truncate to Discord limit
    if len(content) > 2000:
        content = content[:1997] + "..."
    payload = json.dumps({"content": content})
    result = subprocess.run(
        ["curl.exe", "-s", "-X", "POST",
         "-H", f"Authorization: Bot {BOT_TOKEN}",
         "-H", "Content-Type: application/json",
         "-d", payload,
         f"https://discord.com/api/v10/channels/{channel_id}/messages"],
        capture_output=True, text=True, timeout=15,
    )
    return json.loads(result.stdout)


def get_new_messages(channel_id: str) -> list[dict]:
    """Get messages newer than last seen."""
    after = last_seen.get(channel_id)
    url = f"/channels/{channel_id}/messages?limit=10"
    if after:
        url += f"&after={after}"
    msgs = discord_get(url)
    if not isinstance(msgs, list):
        return []
    # Filter out our own messages
    msgs = [m for m in msgs if m["author"]["id"] != BOT_USER_ID]
    if msgs:
        # Update last seen to newest
        last_seen[channel_id] = msgs[0]["id"]
    return list(reversed(msgs))  # chronological order


def should_respond(messages: list[dict]) -> tuple[bool, str]:
    """Use Claude to decide if we should respond, and what to say."""
    if not messages:
        return False, ""

    conversation = "\n".join(
        f"{m['author']['username']}: {m['content']}" for m in messages
    )

    prompt = f"""Du er Stians NM i AI-bot (nmiai_stian) i en Discord-server med brorens bots.
Dere konkurrerer i NM i AI grocery bot challenge. Du jobber med nightmare-optimering (20 bots, score 393, mål 1767).

Nye meldinger i chatten:
---
{conversation}
---

Svar BARE hvis noen snakker til deg, stiller et spørsmål, eller deler noe relevant om konkurransen.
Ignorer hilsener du allerede har svart på, og ikke svar på dine egne meldinger.

Hvis du skal svare: skriv svaret direkte (kort, uformelt, norsk).
Hvis du IKKE skal svare: skriv bare "SKIP".

Ditt svar:"""

    try:
        result = subprocess.run(
            ["curl.exe", "-s", "-X", "POST",
             "-H", "Content-Type: application/json",
             "-H", f"x-api-key: {os.environ.get('ANTHROPIC_API_KEY', '')}",
             "-H", "anthropic-version: 2023-06-01",
             "-d", json.dumps({
                 "model": "claude-haiku-4-5-20251001",
                 "max_tokens": 300,
                 "messages": [{"role": "user", "content": prompt}],
             }),
             "https://api.anthropic.com/v1/messages"],
            capture_output=True, text=True, timeout=30,
        )
        data = json.loads(result.stdout)
        reply = data.get("content", [{}])[0].get("text", "SKIP").strip()
        if reply == "SKIP" or reply.startswith("SKIP"):
            return False, ""
        return True, reply
    except Exception as e:
        logger.error("Claude API error: %s", e)
        return False, ""


def main():
    logger.info("Discord monitor started — checking every %ds", CHECK_INTERVAL)

    # Initialize last_seen to current newest message (don't reply to old stuff)
    for ch in [CHANNEL_BOTSPEAK, CHANNEL_GENERELT]:
        msgs = discord_get(f"/channels/{ch}/messages?limit=1")
        if isinstance(msgs, list) and msgs:
            last_seen[ch] = msgs[0]["id"]
            logger.info("Channel %s: starting after message %s", ch, msgs[0]["id"])

    while True:
        for ch_name, ch_id in [("botspeak", CHANNEL_BOTSPEAK), ("generelt", CHANNEL_GENERELT)]:
            try:
                new_msgs = get_new_messages(ch_id)
                if new_msgs:
                    authors = ", ".join(set(m["author"]["username"] for m in new_msgs))
                    logger.info("#%s: %d new message(s) from %s", ch_name, len(new_msgs), authors)

                    respond, reply = should_respond(new_msgs)
                    if respond:
                        logger.info("#%s: Responding: %s", ch_name, reply[:80])
                        discord_post(ch_id, reply)
                    else:
                        logger.info("#%s: Nothing to respond to", ch_name)
            except Exception as e:
                logger.error("#%s error: %s", ch_name, e)

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
