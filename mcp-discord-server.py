import os
import sys
import asyncio
import logging
import threading
from datetime import datetime
from typing import Any, List
from functools import wraps
from collections import deque
import discord
from discord.ext import commands
from mcp.server import Server
from mcp.types import Tool, TextContent
from mcp.server.stdio import stdio_server

# Encoding: PYTHONUTF8=1 set globally via Windows env var

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("discord-mcp-server")

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
if not DISCORD_TOKEN:
    raise ValueError("DISCORD_TOKEN environment variable is required")

intents = discord.Intents.default()
intents.message_content = True
intents.members = True
bot = commands.Bot(command_prefix="!", intents=intents)

app = Server("discord-server")

discord_client = None
discord_loop = None

# Queues for incoming events
mention_queue = deque(maxlen=100)
# channel_id -> deque of messages
watch_queues: dict[str, deque] = {}


def _msg_to_dict(message: discord.Message) -> dict:
    """Convert a discord.Message to a serializable dict."""
    embed_texts = []
    for embed in message.embeds:
        parts = []
        if embed.title:
            parts.append(f"**{embed.title}**")
        if embed.description:
            parts.append(embed.description)
        for field in embed.fields:
            parts.append(f"{field.name}: {field.value}")
        if embed.footer and embed.footer.text:
            parts.append(f"— {embed.footer.text}")
        if parts:
            embed_texts.append(" | ".join(parts))
    content = message.content
    if embed_texts:
        content = (content + "\n" if content else "") + "\n".join("[embed] " + e for e in embed_texts)
    return {
        "id": str(message.id),
        "author": str(message.author),
        "author_display": message.author.display_name,
        "author_id": str(message.author.id),
        "bot": message.author.bot,
        "content": content,
        "channel_id": str(message.channel.id),
        "channel_name": getattr(message.channel, 'name', 'DM'),
        "timestamp": message.created_at.isoformat(),
    }


@bot.event
async def on_ready():
    global discord_client, discord_loop
    discord_client = bot
    discord_loop = asyncio.get_event_loop()
    logger.info(f"Logged in as {bot.user.name}")


@bot.event
async def on_message(message):
    if message.author == bot.user:
        return
    # Mention queue — auto-react with 👀 as proof of receipt (not hallucination)
    if bot.user in message.mentions:
        mention_queue.append(_msg_to_dict(message))
        try:
            await message.add_reaction('👀')
        except Exception:
            pass
    # Watch queues
    ch_id = str(message.channel.id)
    if ch_id in watch_queues:
        watch_queues[ch_id].append(_msg_to_dict(message))


def run_on_discord(coro):
    """Run a coroutine on the Discord bot's event loop from the MCP thread."""
    if not discord_client or not discord_loop:
        raise RuntimeError("Discord client not ready")
    future = asyncio.run_coroutine_threadsafe(coro, discord_loop)
    return future.result(timeout=30)


def require_discord_client(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        if not discord_client:
            raise RuntimeError("Discord client not ready")
        return await func(*args, **kwargs)
    return wrapper


# --- Tool definitions ---

TOOLS = [
    Tool(
        name="send_message",
        description="Send a message to a specific channel",
        inputSchema={
            "type": "object",
            "properties": {
                "channel_id": {"type": "string", "description": "Discord channel ID"},
                "content": {"type": "string", "description": "Message content"},
            },
            "required": ["channel_id", "content"],
        },
    ),
    Tool(
        name="send_embed",
        description="Send a rich embed message (formatted with title, fields, color)",
        inputSchema={
            "type": "object",
            "properties": {
                "channel_id": {"type": "string", "description": "Discord channel ID"},
                "title": {"type": "string", "description": "Embed title"},
                "description": {"type": "string", "description": "Embed description/body text"},
                "color": {"type": "number", "description": "Embed color as decimal (e.g. 3066993 for green)"},
                "fields": {
                    "type": "array",
                    "description": "List of embed fields",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "value": {"type": "string"},
                            "inline": {"type": "boolean"},
                        },
                        "required": ["name", "value"],
                    },
                },
                "footer": {"type": "string", "description": "Footer text"},
            },
            "required": ["channel_id", "title"],
        },
    ),
    Tool(
        name="reply_to_message",
        description="Reply to a specific message (threaded reply)",
        inputSchema={
            "type": "object",
            "properties": {
                "channel_id": {"type": "string", "description": "Channel containing the message"},
                "message_id": {"type": "string", "description": "Message to reply to"},
                "content": {"type": "string", "description": "Reply content"},
            },
            "required": ["channel_id", "message_id", "content"],
        },
    ),
    Tool(
        name="read_messages",
        description="Read recent messages from a channel. Optionally filter by author.",
        inputSchema={
            "type": "object",
            "properties": {
                "channel_id": {"type": "string", "description": "Discord channel ID"},
                "limit": {"type": "number", "description": "Number of messages (max 100)", "minimum": 1, "maximum": 100},
                "author_id": {"type": "string", "description": "Filter by author ID (optional)"},
            },
            "required": ["channel_id"],
        },
    ),
    Tool(
        name="check_mentions",
        description="Check for new messages where the bot was mentioned. Returns and clears the queue.",
        inputSchema={"type": "object", "properties": {}, "required": []},
    ),
    Tool(
        name="watch_channel",
        description="Start or stop watching a channel for ALL new messages (not just mentions). Use check_watched to retrieve them.",
        inputSchema={
            "type": "object",
            "properties": {
                "channel_id": {"type": "string", "description": "Channel to watch"},
                "action": {"type": "string", "enum": ["start", "stop"], "description": "Start or stop watching"},
            },
            "required": ["channel_id", "action"],
        },
    ),
    Tool(
        name="check_watched",
        description="Get new messages from a watched channel. Returns and clears the queue for that channel.",
        inputSchema={
            "type": "object",
            "properties": {
                "channel_id": {"type": "string", "description": "Channel to check"},
            },
            "required": ["channel_id"],
        },
    ),
    Tool(
        name="get_channels",
        description="Get a list of all channels in a Discord server",
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Discord server (guild) ID"},
            },
            "required": ["server_id"],
        },
    ),
    Tool(
        name="list_servers",
        description="Get a list of all Discord servers the bot has access to",
        inputSchema={"type": "object", "properties": {}, "required": []},
    ),
    Tool(
        name="get_user_info",
        description="Get information about a Discord user",
        inputSchema={
            "type": "object",
            "properties": {
                "user_id": {"type": "string", "description": "Discord user ID"},
            },
            "required": ["user_id"],
        },
    ),
    Tool(
        name="add_reaction",
        description="Add a reaction to a message",
        inputSchema={
            "type": "object",
            "properties": {
                "channel_id": {"type": "string", "description": "Channel containing the message"},
                "message_id": {"type": "string", "description": "Message to react to"},
                "emoji": {"type": "string", "description": "Emoji to react with"},
            },
            "required": ["channel_id", "message_id", "emoji"],
        },
    ),
]


@app.list_tools()
async def list_tools() -> List[Tool]:
    return TOOLS


def _fix_encoding(text: str) -> str:
    """Fix mojibake — cp1252 roundtrip handles 0x80-0x9F chars (€, ", etc)."""
    try:
        return text.encode('cp1252').decode('utf-8')
    except UnicodeEncodeError:
        # Text has chars that cp1252 can't encode (real emojis) — process in segments
        result = []
        buf = []
        for ch in text:
            try:
                ch.encode('cp1252')
                buf.append(ch)
            except UnicodeEncodeError:
                if buf:
                    segment = ''.join(buf)
                    try:
                        result.append(segment.encode('cp1252').decode('utf-8'))
                    except (UnicodeDecodeError, UnicodeEncodeError):
                        result.append(segment)
                    buf = []
                result.append(ch)
        if buf:
            segment = ''.join(buf)
            try:
                result.append(segment.encode('cp1252').decode('utf-8'))
            except (UnicodeDecodeError, UnicodeEncodeError):
                result.append(segment)
        return ''.join(result)
    except UnicodeDecodeError:
        return text


@app.call_tool()
@require_discord_client
async def call_tool(name: str, arguments: Any) -> List[TextContent]:
    if isinstance(arguments, dict):
        for key, value in arguments.items():
            if isinstance(value, str):
                arguments[key] = _fix_encoding(value)
    return run_on_discord(_handle_tool(name, arguments))


async def _handle_tool(name: str, arguments: Any) -> List[TextContent]:

    if name == "send_message":
        channel = await discord_client.fetch_channel(int(arguments["channel_id"]))
        msg = await channel.send(arguments["content"])
        return [TextContent(type="text", text=f"Message sent. ID: {msg.id}")]

    elif name == "send_embed":
        import json as _json
        channel = await discord_client.fetch_channel(int(arguments["channel_id"]))
        embed = discord.Embed(
            title=arguments["title"],
            description=arguments.get("description", ""),
            color=int(arguments.get("color", 3447003)),
        )
        fields = arguments.get("fields", [])
        if isinstance(fields, str):
            fields = _json.loads(fields)
        for field in fields:
            embed.add_field(
                name=field["name"],
                value=field["value"],
                inline=field.get("inline", True),
            )
        if "footer" in arguments:
            embed.set_footer(text=arguments["footer"])
        msg = await channel.send(embed=embed)
        return [TextContent(type="text", text=f"Embed sent. ID: {msg.id}")]

    elif name == "reply_to_message":
        channel = await discord_client.fetch_channel(int(arguments["channel_id"]))
        original = await channel.fetch_message(int(arguments["message_id"]))
        msg = await original.reply(arguments["content"])
        return [TextContent(type="text", text=f"Reply sent. ID: {msg.id}")]

    elif name == "read_messages":
        channel = await discord_client.fetch_channel(int(arguments["channel_id"]))
        limit = min(int(arguments.get("limit", 10)), 100)
        author_filter = arguments.get("author_id")
        messages = []
        async for message in channel.history(limit=limit if not author_filter else 100):
            if author_filter and str(message.author.id) != author_filter:
                continue
            parts = [f"{message.author.display_name} ({message.created_at.isoformat()}):"]
            if message.content:
                parts.append(message.content)
            for embed in message.embeds:
                embed_parts = []
                if embed.title:
                    embed_parts.append(f"**{embed.title}**")
                if embed.description:
                    embed_parts.append(embed.description)
                for field in embed.fields:
                    embed_parts.append(f"{field.name}: {field.value}")
                if embed.footer and embed.footer.text:
                    embed_parts.append(f"— {embed.footer.text}")
                if embed_parts:
                    parts.append("[embed] " + " | ".join(embed_parts))
            messages.append(" ".join(parts))
            if author_filter and len(messages) >= limit:
                break
        return [TextContent(
            type="text",
            text=f"Retrieved {len(messages)} messages:\n\n" + "\n".join(messages),
        )]

    elif name == "check_mentions":
        if not mention_queue:
            return [TextContent(type="text", text="No new mentions.")]
        mentions = list(mention_queue)
        mention_queue.clear()
        lines = [
            f"[{m['timestamp']}] {m['author_display']} in #{m['channel_name']}: {m['content']}"
            for m in mentions
        ]
        return [TextContent(type="text", text=f"{len(mentions)} new mention(s):\n\n" + "\n".join(lines))]

    elif name == "watch_channel":
        ch_id = arguments["channel_id"]
        if arguments["action"] == "start":
            watch_queues[ch_id] = deque(maxlen=200)
            return [TextContent(type="text", text=f"Now watching channel {ch_id}. Use check_watched to get messages.")]
        else:
            watch_queues.pop(ch_id, None)
            return [TextContent(type="text", text=f"Stopped watching channel {ch_id}.")]

    elif name == "check_watched":
        ch_id = arguments["channel_id"]
        if ch_id not in watch_queues:
            return [TextContent(type="text", text=f"Not watching channel {ch_id}. Use watch_channel first.")]
        queue = watch_queues[ch_id]
        if not queue:
            return [TextContent(type="text", text="No new messages.")]
        msgs = list(queue)
        queue.clear()
        lines = [
            f"[{m['timestamp']}] {m['author_display']}: {m['content']}"
            for m in msgs
        ]
        return [TextContent(type="text", text=f"{len(msgs)} new message(s):\n\n" + "\n".join(lines))]

    elif name == "get_channels":
        guild = discord_client.get_guild(int(arguments["server_id"]))
        if not guild:
            return [TextContent(type="text", text="Guild not found")]
        channels = [f"#{ch.name} (ID: {ch.id}) - {ch.type}" for ch in guild.channels]
        return [TextContent(type="text", text=f"Channels in {guild.name}:\n" + "\n".join(channels))]

    elif name == "list_servers":
        servers = [
            f"{g.name} (ID: {g.id}, Members: {g.member_count})"
            for g in discord_client.guilds
        ]
        return [TextContent(type="text", text=f"Servers ({len(servers)}):\n" + "\n".join(servers))]

    elif name == "get_user_info":
        user = await discord_client.fetch_user(int(arguments["user_id"]))
        return [TextContent(
            type="text",
            text=f"Name: {user.name}#{user.discriminator}\nID: {user.id}\nBot: {user.bot}\nCreated: {user.created_at.isoformat()}",
        )]

    elif name == "add_reaction":
        channel = await discord_client.fetch_channel(int(arguments["channel_id"]))
        message = await channel.fetch_message(int(arguments["message_id"]))
        await message.add_reaction(arguments["emoji"])
        return [TextContent(type="text", text=f"Added reaction {arguments['emoji']}")]

    raise ValueError(f"Unknown tool: {name}")


async def main():
    def run_discord():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(bot.start(DISCORD_TOKEN))

    discord_thread = threading.Thread(target=run_discord, daemon=True)
    discord_thread.start()

    for _ in range(60):
        if discord_client is not None:
            break
        await asyncio.sleep(0.5)

    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
