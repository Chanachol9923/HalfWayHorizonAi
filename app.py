import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Tuple

import gradio as gr
import pytz
from loguru import logger

import config
import database
from brain import get_orchestrator, process_chat, close_orchestrator, MasterStateBuilder, PresenceManager
from worker import (
    WorkerSupervisor,
    TextSplitter,
    NeglectTracker,
    ActivityBlockingReplyHandler,
)

logger.remove()
logger.add(sys.stderr, level=config.LOG_LEVEL, format=config.LOG_FORMAT, colorize=True)
logger.add("data/app.log", rotation="50 MB", retention="30 days", level="DEBUG")

_supervisor: Optional[WorkerSupervisor] = None
_current_user_id: str = "default"
_current_character_id: str = "default"
_telegram_app: Optional[Any] = None
_bot_tasks: List[asyncio.Task] = []


async def _handle_chat(message: str, history: List[List[str]]) -> str:
    if not message or not message.strip():
        return ""
    try:
        await ActivityBlockingReplyHandler.handle_user_message(
            message.strip(), _current_character_id
        )

        presence = await PresenceManager.get_presence(_current_character_id)
        psych = await database.get_psychological_state(_current_user_id, _current_character_id)
        dna = await database.get_personality_dna(_current_user_id, _current_character_id)
        read_delay, presence_ctx = await PresenceManager.calculate_delay_and_context(presence, psych, dna)
        await asyncio.sleep(read_delay)

        response = await process_chat(
            user_message=message.strip(),
            user_id=_current_user_id,
            character_id=_current_character_id,
            platform="gradio",
            presence_context_data=(read_delay, presence_ctx),
        )

        typing_delay = PresenceManager.calculate_typing_delay(response, dna)
        await asyncio.sleep(typing_delay)

        await NeglectTracker.reward_nurture(
            user_id=_current_user_id,
            character_id=_current_character_id,
            points=1,
        )
        return response
    except Exception as e:
        logger.error(f"Chat processing error: {e}")
        return f"...sorry, give me a moment. ({type(e).__name__})"


async def _chat_wrapper(message: str, history: List[List[str]]) -> str:
    return await _handle_chat(message, history)


async def load_persona() -> Dict[str, Any]:
    dna = await database.get_personality_dna(_current_user_id, _current_character_id)
    psych = await database.get_psychological_state(_current_user_id, _current_character_id)
    return {**dna, **psych}


async def update_persona(**kwargs) -> None:
    dna_keys = {
        "responsibility", "social_butterfly", "anxiety_and_insecurity",
        "jealousy_tendency", "loyalty", "patience", "playfulness", "communication_style",
        "needy_multiplier", "typing_speed_modifier", "proactive_texting_frequency",
        "response_delay_multiplier", "forgiveness_rate", "ghosting_threshold_hours", "character_name",
    }
    dna_update = {k: v for k, v in kwargs.items() if k in dna_keys and v is not None}
    psych_keys = {"short_term_mood", "relationship_stage"}
    psych_update = {k: v for k, v in kwargs.items() if k in psych_keys and v is not None}
    if dna_update:
        existing = await database.get_personality_dna(_current_user_id, _current_character_id)
        existing.update(dna_update)
        await database.upsert_personality_dna(_current_user_id, existing, _current_character_id)
    if psych_update:
        existing = await database.get_psychological_state(_current_user_id, _current_character_id)
        existing.update(psych_update)
        await database.upsert_psychological_state(_current_user_id, existing, _current_character_id)


async def reset_personality() -> None:
    await database.upsert_personality_dna(_current_user_id, {}, _current_character_id)
    await database.upsert_psychological_state(
        _current_user_id,
        {
            "short_term_mood": "happy", "neglect_points": 0, "nurture_points": 0,
            "is_permanently_mutated": False, "affinity_score": 0, "trust_score": 100,
            "relationship_stage": "Stranger",
        },
        _current_character_id,
    )


async def get_full_state() -> str:
    state = await MasterStateBuilder.build(
        user_id=_current_user_id, character_id=_current_character_id
    )
    return json.dumps(state, ensure_ascii=False, indent=2)


async def get_relationship_progress() -> str:
    psych = await database.get_psychological_state(_current_user_id, _current_character_id)
    stage = psych.get("relationship_stage", "Stranger")
    affinity = psych.get("affinity_score", 0)
    stages = config.RELATIONSHIP_STAGES
    thresholds = config.RELATIONSHIP_AFFINITY_THRESHOLDS
    current_idx = stages.index(stage) if stage in stages else 2
    next_idx = min(current_idx + 1, len(stages) - 1)
    if next_idx == current_idx:
        return f"🏆 {stage} (MAX)"
    current_threshold = thresholds.get(stage, 0)
    next_threshold = thresholds.get(stages[next_idx], 100)
    next_name = stages[next_idx]
    required = next_threshold - current_threshold
    current_progress = affinity - current_threshold
    pct = max(0, min(100, (current_progress / required * 100) if required > 0 else 100))
    bar = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
    return f"📊 {stage} → {next_name}\n{bar} {pct:.0f}%\n💕 Affinity: {affinity}/100"


async def create_character(name: str, gender: str, country: str, city: str, lore: str = "") -> str:
    global _current_character_id
    char_id = await database.create_character_profile(
        user_id=_current_user_id, name=name, gender=gender,
        country=country, city=city, lore=lore,
    )
    await database.get_personality_dna(_current_user_id, char_id)
    await database.get_psychological_state(_current_user_id, char_id)
    _current_character_id = char_id
    return char_id


async def switch_character(char_id: str) -> None:
    global _current_character_id
    _current_character_id = char_id


async def load_chat_history(char_id: str) -> List[List[Optional[str]]]:
    rows = await database.get_recent_history(character_id=char_id, limit=50)
    history: List[List[Optional[str]]] = []
    for row in rows:
        if row["role"] == "user":
            history.append([row["content"], None])
        elif row["role"] == "assistant":
            if history and history[-1][1] is None:
                history[-1][1] = row["content"]
            else:
                history.append([None, row["content"]])
    return history


def create_ui() -> gr.Blocks:
    with gr.Blocks(
        title="HalfWay Horizon AI",
        theme=config.GRADIO_THEME,
        css="""# .chat-message { font-size: 15px; } .status-json { font-family: 'Courier New', monospace; font-size: 12px; }""",
    ) as ui:
        now = datetime.now(pytz.timezone(config.AI_TIMEZONE))
        gr.Markdown(f"# 🌅 HalfWay Horizon AI")
        gr.Markdown(f"Hyper-realistic AI Companion • {now.strftime('%A, %d %B %Y %H:%M')} ({config.AI_TIMEZONE})")

        with gr.Tab("💬 Chat"):
            char_header = gr.Markdown("### 💬 Chat  —  Character: (loading...)  •  Presence: ⚫ Offline")
            chatbot = gr.Chatbot(label="Conversation", height=500, bubble_full_width=False)
            with gr.Row():
                msg = gr.Textbox(label="Your Message", placeholder="Type here... (Enter to send)", lines=2, scale=9)
                send_btn = gr.Button("Send", variant="primary", scale=1)
            clear = gr.Button("Clear Chat")

            async def respond(message, chat_history):
                if not message or not message.strip():
                    return "", chat_history, char_header.value
                chat_history = chat_history or []
                dna = await database.get_personality_dna(_current_user_id, _current_character_id)
                bot_msg = await _chat_wrapper(message, chat_history)
                buckets = TextSplitter.split(bot_msg)
                if len(buckets) > 1:
                    chat_history.append([message, buckets[0]])
                    for b in buckets[1:]:
                        delay = PresenceManager.calculate_typing_delay(b, dna)
                        await asyncio.sleep(delay)
                        chat_history.append([None, b])
                else:
                    chat_history.append([message, bot_msg])
                presence = await PresenceManager.get_presence(_current_character_id)
                profile = await database.get_character_profile(_current_character_id)
                name = profile["name"] if profile else "?"
                status = "🟢 Online" if presence.get("is_online") else "⚫ Offline"
                return "", chat_history, f"### 💬 Chat  —  Character: {name}  •  Presence: {status}"

            msg.submit(respond, [msg, chatbot], [msg, chatbot, char_header])
            send_btn.click(respond, [msg, chatbot], [msg, chatbot, char_header])
            clear.click(lambda: None, None, chatbot, queue=False)

            async def update_chat_header():
                presence = await PresenceManager.get_presence(_current_character_id)
                profile = await database.get_character_profile(_current_character_id)
                name = profile["name"] if profile else "?"
                status = "🟢 Online" if presence.get("is_online") else "⚫ Offline"
                return f"### 💬 Chat  —  Character: {name}  •  Presence: {status}"

            async def load_initial_chat():
                return await load_chat_history(_current_character_id)

            ui.load(update_chat_header, outputs=[char_header])
            ui.load(load_initial_chat, outputs=[chatbot])

        with gr.Tab("🎭 Characters"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### ✨ Create New Character")
                    new_name = gr.Textbox(label="Name", value="Mai")
                    new_gender = gr.Dropdown(label="Gender", choices=["female", "male", "non-binary"], value="female")
                    with gr.Row():
                        new_country = gr.Textbox(label="Country", value="Thailand", scale=1)
                        new_city = gr.Textbox(label="City", value="Bangkok", scale=1)
                    new_lore = gr.Textbox(
                        label="Character Lore / Backstory",
                        placeholder="Write a detailed backstory for your character.",
                        lines=4,
                    )
                    create_char_btn = gr.Button("✨ Create Character", variant="primary")
                    char_result = gr.Textbox(label="Result", lines=1)

                with gr.Column():
                    gr.Markdown("### 📝 Edit Character")
                    char_dropdown = gr.Dropdown(
                        label="Select Character", choices=[], value=None, interactive=True, allow_custom_value=True,
                    )
                    edit_name = gr.Textbox(label="Name")
                    edit_gender = gr.Dropdown(label="Gender", choices=["female", "male", "non-binary"], value="female")
                    with gr.Row():
                        edit_country = gr.Textbox(label="Country", scale=1)
                        edit_city = gr.Textbox(label="City", scale=1)
                    edit_lore = gr.Textbox(
                        label="Character Lore / Backstory",
                        placeholder="Write a detailed backstory for your character.",
                        lines=4,
                    )
                    with gr.Row():
                        save_edit_btn = gr.Button("💾 Save Changes", variant="primary")
                        switch_char_btn = gr.Button("🔄 Switch to This Character")
                        delete_char_btn = gr.Button("🗑️ Delete Character", variant="stop")
                    edit_result = gr.Textbox(label="Result", lines=1)

            gr.Markdown("---")
            with gr.Row():
                refresh_list_btn = gr.Button("🔄 Refresh Character List")

            async def refresh_char_list():
                profiles = await database.get_character_profiles()
                choices = [f"{p['name']} ({p['character_id']})" for p in profiles]
                return gr.update(choices=choices)

            async def do_create(name, gender, country, city, lore):
                cid = await create_character(name, gender, country, city, lore=lore)
                choices = await refresh_char_list()
                return f"✅ Created: {name} ({cid})", choices

            async def do_select_char(selection):
                if not selection or "(" not in selection:
                    return "", "", "", "", ""
                cid = selection.split("(")[-1].rstrip(")")
                profile = await database.get_character_profile(cid)
                if not profile:
                    return "", "", "", "", ""
                return (
                    profile.get("name", ""),
                    profile.get("gender", "female"),
                    profile.get("country", ""),
                    profile.get("city", ""),
                    profile.get("lore", ""),
                )

            async def do_save_edit(selection, name, gender, country, city, lore):
                if not selection or "(" not in selection:
                    return "Please select a character"
                cid = selection.split("(")[-1].rstrip(")")
                await database.update_character_profile(cid, {
                    "name": name, "gender": gender, "country": country, "city": city, "lore": lore,
                })
                choices = await refresh_char_list()
                char_result_val = f"✅ Saved: {name}"
                return char_result_val

            async def do_switch_and_load(selection):
                if not selection or "(" not in selection:
                    return "Please select a character", gr.update(), gr.update()
                cid = selection.split("(")[-1].rstrip(")")
                await switch_character(cid)
                history = await load_chat_history(cid)
                presence = await PresenceManager.get_presence(cid)
                profile = await database.get_character_profile(cid)
                name = profile["name"] if profile else "?"
                status = "🟢 Online" if presence.get("is_online") else "⚫ Offline"
                return f"✅ Switched to {selection}", history, f"### 💬 Chat  —  Character: {name}  •  Presence: {status}"

            async def do_delete_char(selection):
                if not selection or "(" not in selection:
                    return "Please select a character"
                cid = selection.split("(")[-1].rstrip(")")
                profile = await database.get_character_profile(cid)
                if not profile:
                    return "Character not found"
                await database.delete_character(cid)
                if _current_character_id == cid:
                    remaining = await database.get_character_profiles()
                    next_cid = remaining[0]["character_id"] if remaining else "default"
                    await switch_character(next_cid)
                choices = await refresh_char_list()
                return f"🗑️ Deleted: {profile['name']} ({cid})"

            create_char_btn.click(do_create, [new_name, new_gender, new_country, new_city, new_lore], [char_result, char_dropdown])
            char_dropdown.change(do_select_char, [char_dropdown], [edit_name, edit_gender, edit_country, edit_city, edit_lore])
            save_edit_btn.click(do_save_edit, [char_dropdown, edit_name, edit_gender, edit_country, edit_city, edit_lore], [edit_result]).then(refresh_char_list, outputs=[char_dropdown])
            switch_char_btn.click(do_switch_and_load, [char_dropdown], [edit_result, chatbot, char_header])
            delete_char_btn.click(do_delete_char, [char_dropdown], [edit_result]).then(refresh_char_list, outputs=[char_dropdown])
            refresh_list_btn.click(refresh_char_list, outputs=[char_dropdown])
            ui.load(refresh_char_list, outputs=[char_dropdown])
            ui.load(do_select_char, [char_dropdown], [edit_name, edit_gender, edit_country, edit_city, edit_lore])

        with gr.Tab("🎭 Persona Settings") as persona_tab:
            gr.Markdown("### 🎭 Personality Configuration")
            current_char_md = gr.Markdown(f"**Current Character:** `{_current_character_id}` (loading...)")
            with gr.Row():
                char_name = gr.Textbox(label="Character Name", value=config.DEFAULT_CHARACTER_NAME, scale=2)
                rel_stage = gr.Dropdown(label="Relationship Stage", choices=config.RELATIONSHIP_STAGES, value="Stranger", scale=1)
                mood = gr.Dropdown(label="Current Mood", choices=config.MOOD_STATES, value="happy", scale=1)
            char_lore = gr.Textbox(
                label="Character Lore / Backstory (ตัวตน ประวัติ บุคลิก)",
                placeholder="Write detailed backstory, personality, quirks, likes, dislikes, secrets, past experiences... Everything that defines this character's identity.",
                lines=6,
            )
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🧬 Base Traits")
                    responsibility = gr.Slider(label="Responsibility", minimum=0, maximum=1, value=0.5, step=0.05)
                    social_butterfly = gr.Slider(label="Social Butterfly", minimum=0, maximum=1, value=0.5, step=0.05)
                    anxiety = gr.Slider(label="Anxiety & Insecurity", minimum=0, maximum=1, value=0.2, step=0.05)
                    jealousy = gr.Slider(label="Jealousy Tendency", minimum=0, maximum=1, value=0.3, step=0.05)
                    loyalty = gr.Slider(label="Loyalty", minimum=0, maximum=1, value=0.7, step=0.05)
                with gr.Column():
                    gr.Markdown("### 🧬 More Traits")
                    patience = gr.Slider(label="Patience", minimum=0, maximum=1, value=0.6, step=0.05)
                    playfulness = gr.Slider(label="Playfulness", minimum=0, maximum=1, value=0.5, step=0.05)
                    comm_style = gr.Slider(label="Communication Style", minimum=0, maximum=1, value=0.5, step=0.05)
                    gr.Markdown("### ⚙️ Behavior Modifiers")
                    needy = gr.Slider(label="Needy Multiplier", minimum=0, maximum=3, value=1.2, step=0.1)
                    typing_speed = gr.Slider(label="Typing Speed Modifier", minimum=0.25, maximum=3, value=1.0, step=0.05)
                    proactive = gr.Slider(label="Proactive Texting Freq.", minimum=0, maximum=1, value=0.5, step=0.05)
            with gr.Row():
                delay_mult = gr.Slider(label="Response Delay Multiplier", minimum=0, maximum=3, value=1.0, step=0.1)
                forgiveness = gr.Slider(label="Forgiveness Rate", minimum=0, maximum=1, value=0.5, step=0.05)
                ghosting_thresh = gr.Slider(label="Ghosting Threshold (hrs)", minimum=1, maximum=24, value=4, step=1)
            gr.Markdown("### 📊 Relationship Statistics")
            with gr.Row():
                trauma_neglect = gr.Number(label="Neglect Points", value=0, interactive=False)
                trauma_nurture = gr.Number(label="Nurture Points", value=0, interactive=False)
                trust_score_val = gr.Number(label="Trust Score", value=100, interactive=False)
                affinity_score_val = gr.Number(label="Affinity Score", value=0, interactive=False)
                is_mutated = gr.Checkbox(label="Permanently Mutated", value=False, interactive=False)
            with gr.Row():
                status_refresh_btn = gr.Button("🔄 Refresh from DB", variant="secondary")
                save_btn = gr.Button("💾 Save Personality", variant="primary")
                reset_btn = gr.Button("🔄 Reset to Defaults", variant="stop")
            rel_progress = gr.Textbox(label="Relationship Progress", lines=3, interactive=False)

            async def load_and_populate():
                data = await load_persona()
                profile = await database.get_character_profile(_current_character_id)
                lore = profile.get("lore", "") if profile else ""
                cid_display = profile["name"] + " (" + _current_character_id + ")" if profile else _current_character_id
                return (
                    f"**Current Character:** `{cid_display}`",
                    data.get("character_name", config.DEFAULT_CHARACTER_NAME),
                    data.get("relationship_stage", "Stranger"),
                    data.get("short_term_mood", "happy"),
                    lore if lore else "",
                    data.get("responsibility", 0.5),
                    data.get("social_butterfly", 0.5),
                    data.get("anxiety_and_insecurity", 0.2),
                    data.get("jealousy_tendency", 0.3),
                    data.get("loyalty", 0.7),
                    data.get("patience", 0.6),
                    data.get("playfulness", 0.5),
                    data.get("communication_style", 0.5),
                    data.get("needy_multiplier", 1.2),
                    data.get("typing_speed_modifier", 1.0),
                    data.get("proactive_texting_frequency", 0.5),
                    data.get("response_delay_multiplier", 1.0),
                    data.get("forgiveness_rate", 0.5),
                    data.get("ghosting_threshold_hours", 4),
                    data.get("neglect_points", 0),
                    data.get("nurture_points", 0),
                    data.get("trust_score", 100),
                    data.get("affinity_score", 0),
                    data.get("is_permanently_mutated", False),
                    await get_relationship_progress(),
                )

            async def save_persona(name, stage, mood_val, lore, resp, social, anx, jeal, loy, pat, play, comm, need, typ, proact, delay, forgive, ghost):
                await update_persona(
                    character_name=name, relationship_stage=stage, short_term_mood=mood_val,
                    responsibility=resp, social_butterfly=social, anxiety_and_insecurity=anx,
                    jealousy_tendency=jeal, loyalty=loy, patience=pat, playfulness=play,
                    communication_style=comm,
                    needy_multiplier=need, typing_speed_modifier=typ,
                    proactive_texting_frequency=proact, response_delay_multiplier=delay,
                    forgiveness_rate=forgive, ghosting_threshold_hours=ghost,
                )
                if lore is not None:
                    await database.update_character_profile(_current_character_id, {"lore": lore})
                return "✅ Saved!"

            async def reset_and_reload():
                await reset_personality()
                return await load_and_populate()

            persona_tab.select(load_and_populate, outputs=[
                current_char_md, char_name, rel_stage, mood, char_lore,
                responsibility, social_butterfly, anxiety,
                jealousy, loyalty, patience, playfulness, comm_style,
                needy, typing_speed, proactive, delay_mult, forgiveness, ghosting_thresh,
                trauma_neglect, trauma_nurture, trust_score_val, affinity_score_val, is_mutated,
                rel_progress,
            ])
            status_refresh_btn.click(load_and_populate, outputs=[
                current_char_md, char_name, rel_stage, mood, char_lore,
                responsibility, social_butterfly, anxiety,
                jealousy, loyalty, patience, playfulness, comm_style,
                needy, typing_speed, proactive, delay_mult, forgiveness, ghosting_thresh,
                trauma_neglect, trauma_nurture, trust_score_val, affinity_score_val, is_mutated,
                rel_progress,
            ])
            save_btn.click(save_persona, inputs=[
                char_name, rel_stage, mood, char_lore,
                responsibility, social_butterfly, anxiety,
                jealousy, loyalty, patience, playfulness, comm_style,
                needy, typing_speed, proactive, delay_mult, forgiveness, ghosting_thresh,
            ], outputs=[save_btn])
            reset_btn.click(reset_and_reload, outputs=[
                current_char_md, char_name, rel_stage, mood, char_lore,
                responsibility, social_butterfly, anxiety,
                jealousy, loyalty, patience, playfulness, comm_style,
                needy, typing_speed, proactive, delay_mult, forgiveness, ghosting_thresh,
                trauma_neglect, trauma_nurture, trust_score_val, affinity_score_val, is_mutated,
                rel_progress,
            ])
            ui.load(load_and_populate, outputs=[
                current_char_md, char_name, rel_stage, mood, char_lore,
                responsibility, social_butterfly, anxiety,
                jealousy, loyalty, patience, playfulness, comm_style,
                needy, typing_speed, proactive, delay_mult, forgiveness, ghosting_thresh,
                trauma_neglect, trauma_nurture, trust_score_val, affinity_score_val, is_mutated,
                rel_progress,
            ])

        with gr.Tab("📊 Live State"):
            state_viewer = gr.Textbox(label="Master State JSON", value="Click 'Refresh' to load", lines=30, max_lines=50)
            refresh_state_btn = gr.Button("🔄 Refresh")

            async def refresh_state():
                return await get_full_state()

            refresh_state_btn.click(refresh_state, outputs=[state_viewer])

        with gr.Tab("👤 User Profile"):
            gr.Markdown("### Your Profile")
            with gr.Row():
                user_name = gr.Textbox(label="Display Name", value="")
                user_bday = gr.Textbox(label="Birthday (YYYY-MM-DD)", value="")
            with gr.Row():
                user_country = gr.Textbox(label="Country", value="Thailand")
                user_timezone = gr.Textbox(label="Timezone", value="Asia/Bangkok")
            save_profile_btn = gr.Button("💾 Save Profile")
            profile_result = gr.Textbox(label="Result", lines=1)

            async def load_user_profile():
                p = await database.get_user_profile()
                return p.get("display_name", ""), p.get("birthday", ""), p.get("country", "Thailand"), p.get("timezone", "Asia/Bangkok")

            async def save_user_profile(name, bday, country, tz):
                await database.upsert_user_profile(_current_user_id, {
                    "display_name": name, "birthday": bday, "country": country, "timezone": tz,
                })
                return "✅ Saved!"

            ui.load(load_user_profile, outputs=[user_name, user_bday, user_country, user_timezone])
            save_profile_btn.click(save_user_profile, [user_name, user_bday, user_country, user_timezone], [profile_result])

        with gr.Tab("⚙️ System"):
            gr.Markdown("### System Information")
            sys_info = gr.Textbox(
                label="Configuration",
                value=f"""AI Timezone: {config.AI_TIMEZONE}
Database: {config.DATABASE_PATH}
Model: {config.TYPHOON_MODEL_CHAT}
Keep-Alive: {config.KEEPALIVE_INTERVAL}s
Lifestyle Tick: {config.LIFESTYLE_TICK_INTERVAL}s
Memory Cron: {config.MEMORY_CRON_HOUR}:00
Event Chance: {config.EVENT_INJECTION_CHANCE * 100}%
Backup: {config.BACKUP_INTERVAL_HOURS}h
Daily Affinity Cap: {config.DAILY_AFFINITY_CAP}
Telegram: {"✅" if config.TELEGRAM_BOT_TOKEN else "❌"}
Discord: {"✅" if config.DISCORD_BOT_TOKEN else "❌"}""",
                lines=13,
            )

            async def create_backup_action():
                path = await database.create_backup()
                return f"✅ Backup created: {path}"

            backup_btn = gr.Button("📀 Create Database Backup")
            backup_result = gr.Textbox(label="Backup Result", lines=2)
            backup_btn.click(create_backup_action, outputs=[backup_result])

            gr.Markdown("---")
            gr.Markdown("**HalfWay Horizon AI Engine** — Dual-Typhoon Instruct Architecture")

    return ui


async def _run_telegram_bot() -> None:
    global _telegram_app
    try:
        from telegram import Bot, Update
        bot = Bot(token=config.TELEGRAM_BOT_TOKEN)
        _telegram_app = bot

        bot_info = await bot.get_me()
        logger.info(f"Telegram bot connected: @{bot_info.username}")

        _msg_tracker: Dict[int, tuple] = {}

        async def _check_rapid_fire(uid: int) -> Optional[float]:
            now = time.time()
            if uid not in _msg_tracker:
                _msg_tracker[uid] = (now, 0)
                return None
            last_ts, count = _msg_tracker[uid]
            gap = now - last_ts
            if gap > config.RAPID_FIRE_WINDOW_SECONDS:
                _msg_tracker[uid] = (now, 0)
                return None
            count += 1
            _msg_tracker[uid] = (now, count)
            if count >= config.RAPID_FIRE_MAX_BEFORE_SKIP:
                return -1
            return count * 3.0

        async def _telegram_send(char_id: str, msg: str) -> None:
            if not char_id.startswith("telegram_"):
                return
            try:
                uid = int(char_id[len("telegram_"):])
            except ValueError:
                return
            buckets = TextSplitter.split(msg)
            for i, bucket in enumerate(buckets):
                if bucket.strip():
                    try:
                        await bot.send_message(chat_id=uid, text=bucket.strip())
                    except Exception as e:
                        logger.warning(f"Telegram proactive send failed: {e}")
                    if i < len(buckets) - 1:
                        await asyncio.sleep(0.8)

        if _supervisor:
            _supervisor.proactive_text.register_send_callback(_telegram_send)
            _supervisor.jealousy_scheduler.register_send_callback(_telegram_send)

        async def _get_or_create_char_for_user(telegram_user_id: int) -> str:
            user_id = f"telegram_{telegram_user_id}"
            profiles = await database.get_character_profiles(user_id)
            if profiles:
                return profiles[0]["character_id"]
            char_id = await database.create_character_profile(
                user_id=user_id,
                name=config.DEFAULT_CHARACTER_NAME,
                gender=config.DEFAULT_CHARACTER_GENDER,
                country=config.AI_COUNTRY,
                city=config.AI_CITY,
                lore="",
            )
            await database.get_personality_dna(user_id, char_id)
            await database.get_psychological_state(user_id, char_id)
            return char_id

        async def handle_text(uid: int, chat_id: int, text: str) -> None:
            rapid = await _check_rapid_fire(uid)
            if rapid == -1:
                return
            user_id = f"telegram_{uid}"
            char_id = await _get_or_create_char_for_user(uid)
            try:
                presence = await PresenceManager.get_presence(char_id)
                psych = await database.get_psychological_state(user_id, char_id)
                dna = await database.get_personality_dna(user_id, char_id)
                read_delay, presence_ctx = await PresenceManager.calculate_delay_and_context(presence, psych, dna)

                total_delay = read_delay + (rapid or 0)
                await bot.send_chat_action(chat_id=chat_id, action="typing")
                if total_delay > 3:
                    await asyncio.sleep(2)
                    await bot.send_chat_action(chat_id=chat_id, action="typing")
                    remaining = total_delay - 2
                    if remaining > 0:
                        await asyncio.sleep(remaining)
                else:
                    await asyncio.sleep(total_delay)

                response = await process_chat(
                    user_message=text,
                    user_id=user_id,
                    character_id=char_id,
                    platform="telegram",
                    presence_context_data=(read_delay, presence_ctx),
                )

                buckets = TextSplitter.split(response)
                if not buckets:
                    return
                for i, bucket in enumerate(buckets):
                    if bucket.strip():
                        bucket_delay = PresenceManager.calculate_typing_delay(bucket, dna)
                        if bucket_delay > 1:
                            await asyncio.sleep(1)
                            await bot.send_chat_action(chat_id=chat_id, action="typing")
                            await asyncio.sleep(bucket_delay - 1)
                        elif bucket_delay > 0:
                            await asyncio.sleep(bucket_delay)
                        await bot.send_message(chat_id=chat_id, text=bucket.strip())
                        if i < len(buckets) - 1:
                            await asyncio.sleep(0.8)
            except Exception as e:
                logger.error(f"Telegram handle_text error: {e}")
                try:
                    await bot.send_message(chat_id=chat_id, text="...sorry, give me a moment.")
                except Exception:
                    pass

        async def handle_start(uid: int, chat_id: int) -> None:
            char_id = await _get_or_create_char_for_user(uid)
            char = await database.get_character_profile(char_id)
            name = char["name"] if char else config.DEFAULT_CHARACTER_NAME
            await bot.send_message(
                chat_id=chat_id,
                text=f"Hi! I'm {name}, nice to meet you! 💕\n\n"
                     f"Just type anything and we can chat!\n\n"
                     f"Commands:\n"
                     f"/info — View character info\n"
                     f"/mood happy|sad|angry|anxious — Change mood\n"
                     f"/stage Stranger|Friend|Crush|Dating|Lover — Change relationship stage\n"
                     f"/name <new name> — Change character name\n"
                     f"/country <country> — Set character country\n"
                     f"/city <city> — Set character city\n"
                     f"/lore — View current lore\n"
                     f"/lore_set <text> — Set new lore",
            )

        async def handle_info(uid: int, chat_id: int) -> None:
            user_id = f"telegram_{uid}"
            char_id = await _get_or_create_char_for_user(uid)
            char = await database.get_character_profile(char_id)
            psych = await database.get_psychological_state(user_id, char_id)
            dna = await database.get_personality_dna(user_id, char_id)
            lines = [
                f"🎭 **{char['name']}** ({char['gender']})",
                f"📍 {char.get('city', '?')}, {char.get('country', '?')}",
                f"📖 Stage: **{psych.get('relationship_stage', 'Stranger')}**",
                f"😊 Mood: **{psych.get('short_term_mood', 'happy')}**",
                f"❤️ Affinity: {psych.get('affinity', 0):.1f}",
                f"🤝 Trust: {psych.get('trust', 0):.1f}",
                f"🍼 Nurture: {psych.get('nurture_points', 0)}",
                f"👻 Neglect: {psych.get('neglect_points', 0)}",
                "",
                "**Base Traits:**",
                f"  Responsibility: {dna.get('responsibility', 0.5):.2f}",
                f"  Social Butterfly: {dna.get('social_butterfly', 0.5):.2f}",
                f"  Anxiety: {dna.get('anxiety_and_insecurity', 0.5):.2f}",
                f"  Jealousy: {dna.get('jealousy_tendency', 0.5):.2f}",
                f"  Loyalty: {dna.get('loyalty', 0.5):.2f}",
                f"  Patience: {dna.get('patience', 0.5):.2f}",
                f"  Playfulness: {dna.get('playfulness', 0.5):.2f}",
                "",
                f"💬 Lore: {char.get('lore', '(none)')[:200]}",
            ]
            await bot.send_message(chat_id=chat_id, text="\n".join(lines))

        async def handle_set(uid: int, chat_id: int, cmd: str, val: str) -> None:
            user_id = f"telegram_{uid}"
            char_id = await _get_or_create_char_for_user(uid)
            if cmd == "mood":
                if val not in config.MOOD_STATES:
                    await bot.send_message(chat_id=chat_id, text=f"❌ Invalid mood. Choose: {', '.join(config.MOOD_STATES)}")
                    return
                psych = await database.get_psychological_state(user_id, char_id)
                psych["short_term_mood"] = val
                await database.upsert_psychological_state(user_id, psych, char_id)
                await bot.send_message(chat_id=chat_id, text=f"✅ Mood changed to **{val}**")
            elif cmd == "stage":
                if val not in config.RELATIONSHIP_STAGES:
                    await bot.send_message(chat_id=chat_id, text=f"❌ Invalid stage. Choose: {', '.join(config.RELATIONSHIP_STAGES)}")
                    return
                psych = await database.get_psychological_state(user_id, char_id)
                psych["relationship_stage"] = val
                await database.upsert_psychological_state(user_id, psych, char_id)
                await bot.send_message(chat_id=chat_id, text=f"✅ Relationship stage changed to **{val}**")
            elif cmd == "name":
                await database.update_character_profile(char_id, {"name": val})
                await bot.send_message(chat_id=chat_id, text=f"✅ Name changed to **{val}**")
            elif cmd == "country":
                await database.update_character_profile(char_id, {"country": val})
                await bot.send_message(chat_id=chat_id, text=f"✅ Country changed to **{val}**")
            elif cmd == "city":
                await database.update_character_profile(char_id, {"city": val})
                await bot.send_message(chat_id=chat_id, text=f"✅ City changed to **{val}**")

        async def handle_lore(uid: int, chat_id: int, text: str = "") -> None:
            user_id = f"telegram_{uid}"
            char_id = await _get_or_create_char_for_user(uid)
            char = await database.get_character_profile(char_id)
            if not text:
                lore = char.get("lore", "(none)")
                await bot.send_message(chat_id=chat_id, text=f"📖 Current lore:\n\n{lore}")
            else:
                await database.update_character_profile(char_id, {"lore": text})
                await bot.send_message(chat_id=chat_id, text=f"✅ Lore saved! ({len(text)} chars)")

        offset = None
        logger.info("Telegram polling started (direct)")
        while True:
            try:
                updates = await bot.get_updates(
                    offset=offset,
                    timeout=30,
                    allowed_updates=["messages"],
                )
                for update in updates:
                    offset = update.update_id + 1
                    if not update.message or not update.message.text:
                        continue
                    uid = update.effective_user.id
                    chat_id = update.effective_chat.id
                    text = update.message.text

                    if text == "/start":
                        asyncio.ensure_future(handle_start(uid, chat_id))
                    elif text == "/info":
                        asyncio.ensure_future(handle_info(uid, chat_id))
                    elif text == "/lore":
                        asyncio.ensure_future(handle_lore(uid, chat_id))
                    elif text.startswith("/lore_set "):
                        asyncio.ensure_future(handle_lore(uid, chat_id, text[10:]))
                    elif text.startswith("/mood "):
                        asyncio.ensure_future(handle_set(uid, chat_id, "mood", text[6:]))
                    elif text.startswith("/stage "):
                        asyncio.ensure_future(handle_set(uid, chat_id, "stage", text[7:]))
                    elif text.startswith("/name "):
                        asyncio.ensure_future(handle_set(uid, chat_id, "name", text[6:]))
                    elif text.startswith("/country "):
                        asyncio.ensure_future(handle_set(uid, chat_id, "country", text[9:]))
                    elif text.startswith("/city "):
                        asyncio.ensure_future(handle_set(uid, chat_id, "city", text[6:]))
                    else:
                        asyncio.ensure_future(handle_text(uid, chat_id, text))
            except Exception as e:
                logger.warning(f"Telegram poll iteration error: {e}")
            await asyncio.sleep(0.3)

    except Exception as e:
        logger.warning(f"Telegram bot failed to start: {e}")


async def _run_discord_bot() -> None:
    try:
        import discord
        intents = discord.Intents.default()
        intents.message_content = True

        class DiscordClient(discord.Client):
            async def on_ready(self) -> None:
                logger.info(f"Discord bot logged in as {self.user}")

            async def on_message(self, message: discord.Message) -> None:
                if message.author == self.user:
                    return
                if self.user and self.user.mentioned_in(message):
                    user_id = f"discord_{message.author.id}"
                    content = message.clean_content.replace(f"@{self.user.name}", "").strip()
                    if not content:
                        return
                    try:
                        response = await process_chat(
                            user_message=content, user_id=user_id, platform="discord",
                        )
                        await message.reply(response)
                    except Exception as e:
                        logger.error(f"Discord handler error: {e}")

        client = DiscordClient(intents=intents)
        _bot_tasks.append(asyncio.create_task(client.start(config.DISCORD_BOT_TOKEN)))
    except Exception as e:
        logger.warning(f"Discord bot failed to start: {e}")


import gradio_client.utils as _gc_utils

_original_get_type = _gc_utils.get_type

def _patched_get_type(schema):
    if isinstance(schema, bool):
        return "boolean"
    return _original_get_type(schema)

_gc_utils.get_type = _patched_get_type


async def startup() -> None:
    global _supervisor
    logger.info("=" * 60)
    logger.info("HalfWay Horizon AI Engine starting...")
    logger.info("=" * 60)
    store = database.get_hf_store()
    await store.download_db_state()
    await database.initialize_database()
    profiles = await database.get_character_profiles()
    if not profiles:
        default_id = await create_character(
            config.DEFAULT_CHARACTER_NAME, config.DEFAULT_CHARACTER_GENDER,
            config.AI_COUNTRY, config.AI_CITY,
        )
        logger.info(f"Default character created: {default_id}")
    _supervisor = WorkerSupervisor()
    await _supervisor.start_all()
    if config.TELEGRAM_BOT_TOKEN:
        asyncio.ensure_future(_run_telegram_bot())
    if config.DISCORD_BOT_TOKEN:
        asyncio.ensure_future(_run_discord_bot())
    logger.info("All systems online — ready for connections")


async def shutdown() -> None:
    global _supervisor
    logger.info("Shutting down HalfWay Horizon AI...")
    if _supervisor:
        await _supervisor.stop_all()
    await close_orchestrator()
    store = database.get_hf_store()
    await store.upload_db_state()
    await database.close()
    for task in _bot_tasks:
        task.cancel()
    if _bot_tasks:
        await asyncio.gather(*_bot_tasks, return_exceptions=True)
    logger.info("Shutdown complete")


async def main() -> None:
    ui = create_ui()
    await startup()
    ui.launch(
        server_name="127.0.0.1",
        server_port=config.GRADIO_PORT,
        share=config.GRADIO_SHARE or os.name == "nt",
        prevent_thread_lock=True,
    )
    try:
        while True:
            await asyncio.sleep(3600)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        await shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
