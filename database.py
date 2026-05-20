import aiosqlite
import asyncio
import json
import os
import shutil
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import aiofiles
from loguru import logger

import config

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
BACKUP_DIR = Path(config.BACKUP_DIR)
BACKUP_DIR.mkdir(parents=True, exist_ok=True)


def _ensure_db_dir() -> None:
    db_path = Path(config.DATABASE_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)


_db_connection: Optional[aiosqlite.Connection] = None
_db_lock = asyncio.Lock()


async def get_connection() -> aiosqlite.Connection:
    global _db_connection
    if _db_connection is None:
        _ensure_db_dir()
        _db_connection = await aiosqlite.connect(config.DATABASE_PATH)
        _db_connection.row_factory = aiosqlite.Row
        await _db_connection.execute("PRAGMA journal_mode=WAL")
        await _db_connection.execute("PRAGMA foreign_keys=ON")
        await _db_connection.execute("PRAGMA busy_timeout=5000")
    return _db_connection


async def close_connection() -> None:
    global _db_connection
    if _db_connection is not None:
        await _db_connection.close()
        _db_connection = None


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    role TEXT NOT NULL CHECK(role IN ('user','assistant','system')),
    content TEXT NOT NULL,
    platform TEXT NOT NULL DEFAULT 'gradio',
    user_id TEXT NOT NULL DEFAULT 'default',
    character_id TEXT NOT NULL DEFAULT 'default',
    metadata_json TEXT DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_conversations_user_char
    ON conversations(user_id, character_id, created_at);

CREATE TABLE IF NOT EXISTS character_profiles (
    character_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL DEFAULT 'default',
    name TEXT NOT NULL,
    gender TEXT NOT NULL DEFAULT 'female',
    country TEXT NOT NULL DEFAULT 'Thailand',
    city TEXT NOT NULL DEFAULT 'Bangkok',
    timezone TEXT NOT NULL DEFAULT 'Asia/Bangkok',
    avatar_url TEXT DEFAULT NULL,
    lore TEXT DEFAULT '',
    personality TEXT DEFAULT '',
    perspective TEXT DEFAULT '',
    textstyle TEXT DEFAULT '',
    is_active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_characters_user
    ON character_profiles(user_id, is_active);

CREATE TABLE IF NOT EXISTS crystallized_memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_type TEXT NOT NULL CHECK(memory_type IN ('factual','emotional','subtext')),
    content TEXT NOT NULL,
    source_date TEXT NOT NULL,
    importance REAL NOT NULL DEFAULT 0.5,
    user_id TEXT NOT NULL DEFAULT 'default',
    character_id TEXT NOT NULL DEFAULT 'default',
    is_active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_memories_user_char_active
    ON crystallized_memories(user_id, character_id, is_active);

CREATE TABLE IF NOT EXISTS psychological_state (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    character_id TEXT NOT NULL DEFAULT 'default',
    short_term_mood TEXT NOT NULL DEFAULT 'happy',
    neglect_points INTEGER NOT NULL DEFAULT 0,
    nurture_points INTEGER NOT NULL DEFAULT 0,
    is_permanently_mutated INTEGER NOT NULL DEFAULT 0,
    mutation_event TEXT DEFAULT NULL,
    last_mutation_date TEXT DEFAULT NULL,
    affinity_score REAL NOT NULL DEFAULT 0,
    trust_score REAL NOT NULL DEFAULT 100,
    relationship_stage TEXT NOT NULL DEFAULT 'Stranger',
    daily_affinity_gained REAL NOT NULL DEFAULT 0,
    last_affinity_reset TEXT DEFAULT NULL,
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(user_id, character_id)
);

CREATE TABLE IF NOT EXISTS personality_dna (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    character_id TEXT NOT NULL DEFAULT 'default',
    responsibility REAL NOT NULL DEFAULT 0.5,
    social_butterfly REAL NOT NULL DEFAULT 0.5,
    anxiety_and_insecurity REAL NOT NULL DEFAULT 0.2,
    jealousy_tendency REAL NOT NULL DEFAULT 0.3,
    loyalty REAL NOT NULL DEFAULT 0.7,
    patience REAL NOT NULL DEFAULT 0.6,
    playfulness REAL NOT NULL DEFAULT 0.5,
    communication_style REAL NOT NULL DEFAULT 0.5,
    needy_multiplier REAL NOT NULL DEFAULT 1.2,
    typing_speed_modifier REAL NOT NULL DEFAULT 1.0,
    proactive_texting_frequency REAL NOT NULL DEFAULT 0.5,
    response_delay_multiplier REAL NOT NULL DEFAULT 1.0,
    forgiveness_rate REAL NOT NULL DEFAULT 0.5,
    ghosting_threshold_hours REAL NOT NULL DEFAULT 4,
    character_name TEXT NOT NULL DEFAULT 'Mai',
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(user_id, character_id)
);

CREATE TABLE IF NOT EXISTS active_promises (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    promise_id TEXT NOT NULL UNIQUE,
    user_id TEXT NOT NULL DEFAULT 'default',
    character_id TEXT NOT NULL DEFAULT 'default',
    type TEXT NOT NULL DEFAULT 'none',
    description TEXT NOT NULL DEFAULT '',
    is_breaking INTEGER NOT NULL DEFAULT 0,
    source TEXT NOT NULL DEFAULT 'user',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    expires_at TEXT DEFAULT NULL
);

CREATE INDEX IF NOT EXISTS idx_promises_user_char
    ON active_promises(user_id, character_id);

CREATE TABLE IF NOT EXISTS itinerary (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL DEFAULT 'default',
    character_id TEXT NOT NULL DEFAULT 'default',
    plan_name TEXT NOT NULL DEFAULT 'Daily Routine',
    current_phase_index INTEGER NOT NULL DEFAULT 0,
    phases_json TEXT NOT NULL DEFAULT '[]',
    is_active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS activity_blocks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL DEFAULT 'default',
    character_id TEXT NOT NULL DEFAULT 'default',
    activity_type TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ends_at TEXT NOT NULL,
    description TEXT DEFAULT '',
    is_active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_activity_blocks_active
    ON activity_blocks(user_id, character_id, is_active);

CREATE TABLE IF NOT EXISTS subtext_analysis (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL DEFAULT 'default',
    character_id TEXT NOT NULL DEFAULT 'default',
    user_message TEXT NOT NULL,
    detected_intent TEXT NOT NULL DEFAULT 'none',
    intent_validity REAL NOT NULL DEFAULT 1.0,
    analysis TEXT DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS holiday_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL DEFAULT 'default',
    character_id TEXT NOT NULL DEFAULT 'default',
    holiday_name TEXT NOT NULL,
    holiday_date TEXT NOT NULL,
    behavior_type TEXT NOT NULL DEFAULT 'festive',
    is_celebrated INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS user_profile (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL UNIQUE,
    display_name TEXT DEFAULT NULL,
    birthday TEXT DEFAULT NULL,
    country TEXT DEFAULT 'Thailand',
    timezone TEXT NOT NULL DEFAULT 'Asia/Bangkok',
    preferences_json TEXT DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS user_settings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL UNIQUE,
    timezone TEXT NOT NULL DEFAULT 'Asia/Bangkok',
    location_country TEXT DEFAULT 'Thailand',
    platform TEXT NOT NULL DEFAULT 'gradio',
    chat_id TEXT DEFAULT NULL,
    metadata_json TEXT DEFAULT '{}',
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS app_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


async def initialize_database() -> None:
    conn = await get_connection()
    async with _db_lock:
        await conn.executescript(SCHEMA_SQL)
        await conn.commit()
    await _migrate_schema()
    logger.info("Database schema initialized successfully")


async def _migrate_schema() -> None:
    conn = await get_connection()
    async with _db_lock:
        for col in ["lore", "personality", "perspective", "textstyle"]:
            try:
                await conn.execute(f"ALTER TABLE character_profiles ADD COLUMN {col} TEXT DEFAULT ''")
                await conn.commit()
                logger.info(f"Migration: added {col} column to character_profiles")
            except Exception:
                pass


async def save_message(
    role: str,
    content: str,
    platform: str = "gradio",
    user_id: str = "default",
    character_id: str = "default",
    metadata: Optional[Dict[str, Any]] = None,
) -> int:
    conn = await get_connection()
    async with _db_lock:
        cursor = await conn.execute(
            "INSERT INTO conversations (role, content, platform, user_id, character_id, metadata_json) VALUES (?, ?, ?, ?, ?, ?)",
            (role, content, platform, user_id, character_id, json.dumps(metadata or {}, ensure_ascii=False)),
        )
        await conn.commit()
        return cursor.lastrowid


async def get_recent_history(
    user_id: str = "default",
    character_id: str = "default",
    limit: int = 50,
    include_system: bool = False,
) -> List[Dict[str, Any]]:
    conn = await get_connection()
    if include_system:
        rows = await conn.execute_fetchall(
            "SELECT role, content FROM conversations WHERE user_id = ? AND character_id = ? ORDER BY id DESC LIMIT ?",
            (user_id, character_id, limit),
        )
    else:
        rows = await conn.execute_fetchall(
            "SELECT role, content FROM conversations WHERE user_id = ? AND character_id = ? AND role != 'system' ORDER BY id DESC LIMIT ?",
            (user_id, character_id, limit),
        )
    rows.reverse()
    return [{"role": r[0], "content": r[1]} for r in rows]


async def save_crystallized_memory(
    memory_type: str,
    content: str,
    source_date: Optional[str] = None,
    importance: float = 0.5,
    user_id: str = "default",
    character_id: str = "default",
) -> int:
    conn = await get_connection()
    source_date = source_date or datetime.now(timezone.utc).isoformat()
    async with _db_lock:
        cursor = await conn.execute(
            "INSERT INTO crystallized_memories (memory_type, content, source_date, importance, user_id, character_id) VALUES (?, ?, ?, ?, ?, ?)",
            (memory_type, content, source_date, importance, user_id, character_id),
        )
        await conn.commit()
        return cursor.lastrowid


async def get_active_memories(
    user_id: str = "default",
    character_id: str = "default",
    limit: int = 30,
    min_importance: float = 0.0,
) -> List[Dict[str, Any]]:
    conn = await get_connection()
    rows = await conn.execute_fetchall(
        """SELECT memory_type, content, source_date, importance
           FROM crystallized_memories
           WHERE user_id = ? AND character_id = ? AND is_active = 1 AND importance >= ?
           ORDER BY importance DESC, id DESC
           LIMIT ?""",
        (user_id, character_id, min_importance, limit),
    )
    return [
        {
            "memory_type": r[0],
            "content": r[1],
            "source_date": r[2],
            "importance": r[3],
        }
        for r in rows
    ]


async def deactivate_old_memories(user_id: str = "default", character_id: str = "default", keep_count: int = 50) -> int:
    conn = await get_connection()
    async with _db_lock:
        cursor = await conn.execute(
            """UPDATE crystallized_memories SET is_active = 0
               WHERE user_id = ? AND character_id = ? AND id NOT IN (
                   SELECT id FROM crystallized_memories
                   WHERE user_id = ? AND character_id = ? ORDER BY importance DESC LIMIT ?
               )""",
            (user_id, character_id, user_id, character_id, keep_count),
        )
        await conn.commit()
        return cursor.rowcount


async def consolidate_daily_memories(
    user_id: str = "default",
    character_id: str = "default",
) -> Dict[str, List[str]]:
    conn = await get_connection()
    today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    factual_rows = await conn.execute_fetchall(
        """SELECT content FROM conversations
           WHERE user_id = ? AND character_id = ? AND role = 'user'
             AND created_at >= ? AND created_at < ?
           ORDER BY id ASC""",
        (user_id, character_id, (today_start - timedelta(days=1)).isoformat(), today_start.isoformat()),
    )
    fact_texts = [r[0] for r in factual_rows]
    return {"factual": fact_texts, "emotional": [], "subtext": []}


async def get_psychological_state(user_id: str = "default", character_id: str = "default") -> Dict[str, Any]:
    conn = await get_connection()
    cursor = await conn.execute(
        "SELECT * FROM psychological_state WHERE user_id = ? AND character_id = ?", (user_id, character_id)
    )
    row = await cursor.fetchone()
    if row is None:
        default_state = {
            "short_term_mood": "happy",
            "neglect_points": 0,
            "nurture_points": 0,
            "is_permanently_mutated": False,
            "mutation_event": None,
            "last_mutation_date": None,
            "affinity_score": 0.0,
            "trust_score": 100.0,
            "relationship_stage": "Stranger",
            "daily_affinity_gained": 0.0,
            "last_affinity_reset": None,
        }
        await upsert_psychological_state(user_id, default_state, character_id)
        return default_state

    return {
        "short_term_mood": row["short_term_mood"],
        "neglect_points": row["neglect_points"],
        "nurture_points": row["nurture_points"],
        "is_permanently_mutated": bool(row["is_permanently_mutated"]),
        "mutation_event": row["mutation_event"],
        "last_mutation_date": row["last_mutation_date"],
        "affinity_score": row["affinity_score"],
        "trust_score": row["trust_score"],
        "relationship_stage": row["relationship_stage"],
        "daily_affinity_gained": row["daily_affinity_gained"],
        "last_affinity_reset": row["last_affinity_reset"],
    }


async def upsert_psychological_state(
    user_id: str,
    state: Dict[str, Any],
    character_id: str = "default",
) -> None:
    conn = await get_connection()
    async with _db_lock:
        await conn.execute(
            """INSERT INTO psychological_state
               (user_id, character_id, short_term_mood, neglect_points, nurture_points,
                is_permanently_mutated, mutation_event, last_mutation_date,
                affinity_score, trust_score, relationship_stage,
                daily_affinity_gained, last_affinity_reset, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
               ON CONFLICT(user_id, character_id) DO UPDATE SET
               short_term_mood=excluded.short_term_mood,
               neglect_points=excluded.neglect_points,
               nurture_points=excluded.nurture_points,
               is_permanently_mutated=excluded.is_permanently_mutated,
               mutation_event=excluded.mutation_event,
               last_mutation_date=excluded.last_mutation_date,
               affinity_score=excluded.affinity_score,
               trust_score=excluded.trust_score,
               relationship_stage=excluded.relationship_stage,
               daily_affinity_gained=excluded.daily_affinity_gained,
               last_affinity_reset=excluded.last_affinity_reset,
               updated_at=datetime('now')""",
            (
                user_id,
                character_id,
                state.get("short_term_mood", "happy"),
                state.get("neglect_points", 0),
                state.get("nurture_points", 0),
                int(state.get("is_permanently_mutated", False)),
                state.get("mutation_event"),
                state.get("last_mutation_date"),
                state.get("affinity_score", 0),
                state.get("trust_score", 100),
                state.get("relationship_stage", "Stranger"),
                state.get("daily_affinity_gained", 0),
                state.get("last_affinity_reset"),
            ),
        )
        await conn.commit()


async def reset_daily_affinity(user_id: str = "default", character_id: str = "default") -> None:
    conn = await get_connection()
    async with _db_lock:
        await conn.execute(
            """UPDATE psychological_state SET
               daily_affinity_gained = 0,
               last_affinity_reset = datetime('now'),
               updated_at = datetime('now')
               WHERE user_id = ? AND character_id = ?""",
            (user_id, character_id),
        )
        await conn.commit()


async def get_personality_dna(user_id: str = "default", character_id: str = "default") -> Dict[str, Any]:
    conn = await get_connection()
    cursor = await conn.execute(
        "SELECT * FROM personality_dna WHERE user_id = ? AND character_id = ?", (user_id, character_id)
    )
    row = await cursor.fetchone()
    if row is None:
        defaults = {
            "responsibility": 0.5,
            "social_butterfly": 0.5,
            "anxiety_and_insecurity": 0.2,
            "jealousy_tendency": 0.3,
            "loyalty": 0.7,
            "patience": 0.6,
            "playfulness": 0.5,
            "communication_style": 0.5,
            "needy_multiplier": 1.2,
            "typing_speed_modifier": 1.0,
            "proactive_texting_frequency": 0.5,
            "response_delay_multiplier": 1.0,
            "forgiveness_rate": 0.5,
            "ghosting_threshold_hours": 4,
            "character_name": config.DEFAULT_CHARACTER_NAME,
        }
        await upsert_personality_dna(user_id, defaults, character_id)
        return defaults

    return {
        "responsibility": row["responsibility"],
        "social_butterfly": row["social_butterfly"],
        "anxiety_and_insecurity": row["anxiety_and_insecurity"],
        "jealousy_tendency": row["jealousy_tendency"],
        "loyalty": row["loyalty"],
        "patience": row["patience"],
        "playfulness": row["playfulness"],
        "communication_style": row["communication_style"],
        "needy_multiplier": row["needy_multiplier"],
        "typing_speed_modifier": row["typing_speed_modifier"],
        "proactive_texting_frequency": row["proactive_texting_frequency"],
        "response_delay_multiplier": row["response_delay_multiplier"],
        "forgiveness_rate": row["forgiveness_rate"],
        "ghosting_threshold_hours": row["ghosting_threshold_hours"],
        "character_name": row["character_name"],
    }


async def upsert_personality_dna(
    user_id: str,
    dna: Dict[str, Any],
    character_id: str = "default",
) -> None:
    conn = await get_connection()
    async with _db_lock:
        await conn.execute(
            """INSERT INTO personality_dna
               (user_id, character_id, responsibility, social_butterfly, anxiety_and_insecurity,
                jealousy_tendency, loyalty, patience, playfulness, communication_style,
                needy_multiplier, typing_speed_modifier, proactive_texting_frequency,
                response_delay_multiplier, forgiveness_rate, ghosting_threshold_hours,
                character_name, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
               ON CONFLICT(user_id, character_id) DO UPDATE SET
               responsibility=excluded.responsibility,
               social_butterfly=excluded.social_butterfly,
               anxiety_and_insecurity=excluded.anxiety_and_insecurity,
               jealousy_tendency=excluded.jealousy_tendency,
               loyalty=excluded.loyalty,
               patience=excluded.patience,
               playfulness=excluded.playfulness,
               communication_style=excluded.communication_style,
               needy_multiplier=excluded.needy_multiplier,
               typing_speed_modifier=excluded.typing_speed_modifier,
               proactive_texting_frequency=excluded.proactive_texting_frequency,
               response_delay_multiplier=excluded.response_delay_multiplier,
               forgiveness_rate=excluded.forgiveness_rate,
               ghosting_threshold_hours=excluded.ghosting_threshold_hours,
               character_name=excluded.character_name,
               updated_at=datetime('now')""",
            (
                user_id,
                character_id,
                dna.get("responsibility", 0.5),
                dna.get("social_butterfly", 0.5),
                dna.get("anxiety_and_insecurity", 0.2),
                dna.get("jealousy_tendency", 0.3),
                dna.get("loyalty", 0.7),
                dna.get("patience", 0.6),
                dna.get("playfulness", 0.5),
                dna.get("communication_style", 0.5),
                dna.get("needy_multiplier", 1.2),
                dna.get("typing_speed_modifier", 1.0),
                dna.get("proactive_texting_frequency", 0.5),
                dna.get("response_delay_multiplier", 1.0),
                dna.get("forgiveness_rate", 0.5),
                dna.get("ghosting_threshold_hours", 4),
                dna.get("character_name", config.DEFAULT_CHARACTER_NAME),
            ),
        )
        await conn.commit()


async def get_active_promises(user_id: str = "default", character_id: str = "default") -> List[Dict[str, Any]]:
    conn = await get_connection()
    rows = await conn.execute_fetchall(
        "SELECT promise_id, type, description, is_breaking, source, created_at, expires_at FROM active_promises WHERE user_id = ? AND character_id = ?",
        (user_id, character_id),
    )
    return [
        {
            "promise_id": r[0],
            "type": r[1],
            "description": r[2],
            "is_breaking_promise": bool(r[3]),
            "source": r[4],
            "created_at": r[5],
            "expires_at": r[6],
        }
        for r in rows
    ]


async def update_promise_breaking(
    promise_id: str,
    is_breaking: bool,
) -> None:
    conn = await get_connection()
    async with _db_lock:
        await conn.execute(
            "UPDATE active_promises SET is_breaking = ? WHERE promise_id = ?",
            (int(is_breaking), promise_id),
        )
        await conn.commit()


async def clear_expired_promises() -> int:
    conn = await get_connection()
    async with _db_lock:
        cursor = await conn.execute(
            "DELETE FROM active_promises WHERE expires_at IS NOT NULL AND expires_at < datetime('now')"
        )
        await conn.commit()
        return cursor.rowcount


async def get_active_itinerary(user_id: str = "default", character_id: str = "default") -> Optional[Dict[str, Any]]:
    conn = await get_connection()
    cursor = await conn.execute(
        "SELECT * FROM itinerary WHERE user_id = ? AND character_id = ? AND is_active = 1 ORDER BY id DESC LIMIT 1",
        (user_id, character_id),
    )
    row = await cursor.fetchone()
    if row is None:
        return None
    return {
        "plan_name": row["plan_name"],
        "current_phase_index": row["current_phase_index"],
        "phases": json.loads(row["phases_json"]),
        "is_active": bool(row["is_active"]),
    }


async def save_itinerary(
    plan_name: str,
    phases: List[Dict[str, Any]],
    user_id: str = "default",
    character_id: str = "default",
) -> None:
    conn = await get_connection()
    async with _db_lock:
        await conn.execute(
            """INSERT INTO itinerary (plan_name, phases_json, user_id, character_id)
               VALUES (?, ?, ?, ?)""",
            (plan_name, json.dumps(phases, ensure_ascii=False), user_id, character_id),
        )
        await conn.commit()


async def update_itinerary_phase(
    phase_index: int,
    status: str,
    user_id: str = "default",
    character_id: str = "default",
) -> None:
    itinerary = await get_active_itinerary(user_id, character_id)
    if itinerary is None:
        return
    phases = itinerary["phases"]
    if 0 <= phase_index < len(phases):
        phases[phase_index]["status"] = status
        conn = await get_connection()
        async with _db_lock:
            await conn.execute(
                "UPDATE itinerary SET phases_json = ?, current_phase_index = ? WHERE user_id = ? AND character_id = ? AND is_active = 1",
                (json.dumps(phases, ensure_ascii=False), phase_index, user_id, character_id),
            )
            await conn.commit()


async def create_activity_block(
    activity_type: str,
    duration_minutes: int,
    user_id: str = "default",
    character_id: str = "default",
    description: str = "",
) -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    ends_at = now + timedelta(minutes=duration_minutes)
    conn = await get_connection()
    async with _db_lock:
        cursor = await conn.execute(
            "INSERT INTO activity_blocks (user_id, character_id, activity_type, started_at, ends_at, description) VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, character_id, activity_type, now.isoformat(), ends_at.isoformat(), description),
        )
        await conn.commit()
        return {
            "id": cursor.lastrowid,
            "activity_type": activity_type,
            "started_at": now.isoformat(),
            "ends_at": ends_at.isoformat(),
            "description": description,
        }


async def get_active_activity_block(user_id: str = "default", character_id: str = "default") -> Optional[Dict[str, Any]]:
    conn = await get_connection()
    cursor = await conn.execute(
        """SELECT * FROM activity_blocks
           WHERE user_id = ? AND character_id = ? AND is_active = 1
             AND ends_at > datetime('now')
           ORDER BY id DESC LIMIT 1""",
        (user_id, character_id),
    )
    row = await cursor.fetchone()
    if row is None:
        return None
    return {
        "id": row["id"],
        "activity_type": row["activity_type"],
        "started_at": row["started_at"],
        "ends_at": row["ends_at"],
        "description": row["description"],
    }


async def deactivate_activity_block(block_id: int) -> None:
    conn = await get_connection()
    async with _db_lock:
        await conn.execute(
            "UPDATE activity_blocks SET is_active = 0 WHERE id = ?",
            (block_id,),
        )
        await conn.commit()


async def save_subtext_analysis(
    user_message: str,
    detected_intent: str,
    intent_validity: float,
    analysis: str,
    user_id: str = "default",
    character_id: str = "default",
) -> int:
    conn = await get_connection()
    async with _db_lock:
        cursor = await conn.execute(
            "INSERT INTO subtext_analysis (user_id, character_id, user_message, detected_intent, intent_validity, analysis) VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, character_id, user_message, detected_intent, intent_validity, analysis),
        )
        await conn.commit()
        return cursor.lastrowid


async def get_recent_subtext_analyses(
    user_id: str = "default",
    character_id: str = "default",
    limit: int = 10,
) -> List[Dict[str, Any]]:
    conn = await get_connection()
    rows = await conn.execute_fetchall(
        """SELECT user_message, detected_intent, intent_validity, analysis, created_at
           FROM subtext_analysis
           WHERE user_id = ? AND character_id = ?
           ORDER BY id DESC LIMIT ?""",
        (user_id, character_id, limit),
    )
    return [
        {
            "user_message": r[0],
            "detected_intent": r[1],
            "intent_validity": r[2],
            "analysis": r[3],
            "created_at": r[4],
        }
        for r in rows
    ]


async def create_character_profile(
    user_id: str,
    name: str,
    gender: str = "female",
    country: str = "Thailand",
    city: str = "Bangkok",
    timezone: str = "Asia/Bangkok",
    lore: str = "",
    personality: str = "",
    perspective: str = "",
    textstyle: str = "",
) -> str:
    character_id = str(uuid.uuid4())[:8]
    conn = await get_connection()
    async with _db_lock:
        await conn.execute(
            "INSERT INTO character_profiles (character_id, user_id, name, gender, country, city, timezone, lore, personality, perspective, textstyle) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (character_id, user_id, name, gender, country, city, timezone, lore, personality, perspective, textstyle),
        )
        await conn.commit()
    return character_id


async def get_character_profiles(user_id: Optional[str] = None) -> List[Dict[str, Any]]:
    conn = await get_connection()
    if user_id:
        rows = await conn.execute_fetchall(
            "SELECT * FROM character_profiles WHERE user_id = ? AND is_active = 1 ORDER BY created_at ASC",
            (user_id,),
        )
    else:
        rows = await conn.execute_fetchall(
            "SELECT * FROM character_profiles WHERE is_active = 1 ORDER BY created_at ASC",
        )
    return [
        {
            "character_id": r["character_id"],
            "user_id": r["user_id"],
            "name": r["name"],
            "gender": r["gender"],
            "country": r["country"],
            "city": r["city"],
            "timezone": r["timezone"],
            "is_active": bool(r["is_active"]),
            "lore": r["lore"] if "lore" in r.keys() else "",
            "personality": r["personality"] if "personality" in r.keys() else "",
            "perspective": r["perspective"] if "perspective" in r.keys() else "",
            "textstyle": r["textstyle"] if "textstyle" in r.keys() else "",
        }
        for r in rows
    ]


async def get_character_profile(character_id: str) -> Optional[Dict[str, Any]]:
    conn = await get_connection()
    cursor = await conn.execute(
        "SELECT * FROM character_profiles WHERE character_id = ?",
        (character_id,),
    )
    row = await cursor.fetchone()
    if row is None:
        return None
    return {
        "character_id": row["character_id"],
        "user_id": row["user_id"],
        "name": row["name"],
        "gender": row["gender"],
        "country": row["country"],
        "city": row["city"],
        "timezone": row["timezone"],
        "lore": row["lore"] if "lore" in row.keys() else "",
        "personality": row["personality"] if "personality" in row.keys() else "",
        "perspective": row["perspective"] if "perspective" in row.keys() else "",
        "textstyle": row["textstyle"] if "textstyle" in row.keys() else "",
        "is_active": bool(row["is_active"]),
    }


async def update_character_profile(character_id: str, updates: Dict[str, Any]) -> None:
    allowed = {"name", "gender", "country", "city", "timezone", "avatar_url", "is_active", "lore", "personality", "perspective", "textstyle"}
    to_set = {k: v for k, v in updates.items() if k in allowed}
    if not to_set:
        return
    to_set["updated_at"] = datetime.now(timezone.utc).isoformat()
    sets = ", ".join(f"{k} = ?" for k in to_set)
    vals = list(to_set.values())
    vals.append(character_id)
    conn = await get_connection()
    async with _db_lock:
        await conn.execute(
            f"UPDATE character_profiles SET {sets} WHERE character_id = ?",
            vals,
        )
        await conn.commit()


async def delete_character(character_id: str) -> None:
    conn = await get_connection()
    async with _db_lock:
        tables = [
            "conversations",
            "crystallized_memories",
            "psychological_state",
            "personality_dna",
            "active_promises",
            "itinerary",
            "activity_blocks",
            "subtext_analysis",
        ]
        for table in tables:
            await conn.execute(
                f"DELETE FROM {table} WHERE character_id = ?", (character_id,)
            )
        await conn.execute(
            "DELETE FROM character_profiles WHERE character_id = ?", (character_id,)
        )
        await conn.commit()


async def get_user_profile(user_id: str = "default") -> Dict[str, Any]:
    conn = await get_connection()
    cursor = await conn.execute(
        "SELECT * FROM user_profile WHERE user_id = ?", (user_id,)
    )
    row = await cursor.fetchone()
    if row is None:
        defaults = {
            "display_name": None,
            "birthday": None,
            "country": "Thailand",
            "timezone": config.DEFAULT_USER_TIMEZONE,
            "preferences": {},
        }
        await upsert_user_profile(user_id, defaults)
        return defaults
    return {
        "display_name": row["display_name"],
        "birthday": row["birthday"],
        "country": row["country"],
        "timezone": row["timezone"],
        "preferences": json.loads(row["preferences_json"]) if row["preferences_json"] else {},
    }


async def upsert_user_profile(user_id: str, profile: Dict[str, Any]) -> None:
    conn = await get_connection()
    async with _db_lock:
        await conn.execute(
            """INSERT INTO user_profile (user_id, display_name, birthday, country, timezone, preferences_json, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
               ON CONFLICT(user_id) DO UPDATE SET
               display_name=excluded.display_name,
               birthday=excluded.birthday,
               country=excluded.country,
               timezone=excluded.timezone,
               preferences_json=excluded.preferences_json,
               updated_at=datetime('now')""",
            (
                user_id,
                profile.get("display_name"),
                profile.get("birthday"),
                profile.get("country", "Thailand"),
                profile.get("timezone", config.DEFAULT_USER_TIMEZONE),
                json.dumps(profile.get("preferences", {}), ensure_ascii=False),
            ),
        )
        await conn.commit()


async def get_user_settings(user_id: str = "default") -> Dict[str, Any]:
    conn = await get_connection()
    cursor = await conn.execute(
        "SELECT * FROM user_settings WHERE user_id = ?", (user_id,)
    )
    row = await cursor.fetchone()
    if row is None:
        defaults = {
            "timezone": config.DEFAULT_USER_TIMEZONE,
            "location_country": "Thailand",
            "platform": "gradio",
            "chat_id": None,
            "metadata": {},
        }
        await upsert_user_settings(user_id, defaults)
        return defaults
    return {
        "timezone": row["timezone"],
        "location_country": row["location_country"],
        "platform": row["platform"],
        "chat_id": row["chat_id"],
        "metadata": json.loads(row["metadata_json"]) if row["metadata_json"] else {},
    }


async def upsert_user_settings(user_id: str, settings: Dict[str, Any]) -> None:
    conn = await get_connection()
    async with _db_lock:
        await conn.execute(
            """INSERT INTO user_settings (user_id, timezone, location_country, platform, chat_id, metadata_json, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
               ON CONFLICT(user_id) DO UPDATE SET
               timezone=excluded.timezone,
               location_country=excluded.location_country,
               platform=excluded.platform,
               chat_id=excluded.chat_id,
               metadata_json=excluded.metadata_json,
               updated_at=datetime('now')""",
            (
                user_id,
                settings.get("timezone", config.DEFAULT_USER_TIMEZONE),
                settings.get("location_country", "Thailand"),
                settings.get("platform", "gradio"),
                settings.get("chat_id"),
                json.dumps(settings.get("metadata", {}), ensure_ascii=False),
            ),
        )
        await conn.commit()


async def check_and_reset_daily_affinity(user_id: str = "default", character_id: str = "default") -> bool:
    psych = await get_psychological_state(user_id, character_id)
    last_reset = psych.get("last_affinity_reset")
    if last_reset:
        reset_date = datetime.fromisoformat(last_reset)
        if datetime.now(timezone.utc).date() == reset_date.date():
            return False
    await reset_daily_affinity(user_id, character_id)
    return True


async def set_app_state(key: str, value: str) -> None:
    conn = await get_connection()
    async with _db_lock:
        await conn.execute(
            "INSERT INTO app_state (key, value, updated_at) VALUES (?, ?, datetime('now')) ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=datetime('now')",
            (key, value),
        )
        await conn.commit()


async def get_app_state(key: str, default: Optional[str] = None) -> Optional[str]:
    conn = await get_connection()
    cursor = await conn.execute(
        "SELECT value FROM app_state WHERE key = ?", (key,)
    )
    row = await cursor.fetchone()
    return row[0] if row else default


async def create_backup() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = BACKUP_DIR / f"companion_backup_{timestamp}.db"
    conn = await get_connection()
    async with _db_lock:
        await conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        await conn.backup(aiosqlite.connect(str(backup_path)))
    logger.info(f"Database backup created: {backup_path}")
    backups = sorted(BACKUP_DIR.glob("companion_backup_*.db"), reverse=True)
    while len(backups) > config.BACKUP_MAX_COUNT:
        backups.pop().unlink()
    if config.ENABLE_EXTERNAL_BACKUP and config.EXTERNAL_DATABASE_URL:
        asyncio.ensure_future(_sync_to_external(str(backup_path)))
    return str(backup_path)


async def _sync_to_external(backup_path: str) -> None:
    try:
        import httpx
        if config.EXTERNAL_DATABASE_URL and "supabase" in config.EXTERNAL_DATABASE_URL:
            async with aiofiles.open(backup_path, "rb") as f:
                data = await f.read()
            async with httpx.AsyncClient(timeout=60) as client:
                await client.post(
                    f"{config.EXTERNAL_DATABASE_URL}/rest/v1/rpc/upload_backup",
                    headers={"apikey": config.SUPABASE_SERVICE_KEY or ""},
                    files={"file": ("backup.db", data)},
                )
            logger.info("Backup synced to external database")
    except Exception as e:
        logger.warning(f"External backup sync failed: {e}")


async def periodic_backup_task() -> None:
    while True:
        await asyncio.sleep(config.BACKUP_INTERVAL_HOURS * 3600)
        try:
            await create_backup()
            await clear_expired_promises()
        except Exception as e:
            logger.error(f"Periodic backup failed: {e}")


async def close() -> None:
    try:
        await create_backup()
    except Exception as e:
        logger.warning(f"Final backup failed: {e}")
    await close_connection()
    logger.info("Database connection closed")
