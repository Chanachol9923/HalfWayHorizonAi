import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

TYPHOON_API_KEY: str = os.getenv("TYPHOON_API_KEY", "")
TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_API_PROXY: str = os.getenv("TELEGRAM_API_PROXY", "")
DISCORD_BOT_TOKEN: str = os.getenv("DISCORD_BOT_TOKEN", "")

HEADLESS_MODE: bool = os.getenv("HEADLESS_MODE", "true").lower() == "true"
GRADIO_PORT: int = int(os.getenv("GRADIO_PORT", "7860"))
GRADIO_SHARE: bool = os.getenv("GRADIO_SHARE", "false").lower() == "true"

TYPHOON_API_BASE: str = "https://api.opentyphoon.ai/v1"
TYPHOON_MODEL_WORLD: str = os.getenv("TYPHOON_MODEL", "typhoon-v2.5-30b-a3b-instruct")
TYPHOON_MODEL_CHAT: str = os.getenv("TYPHOON_MODEL", "typhoon-v2.5-30b-a3b-instruct")
TYPHOON_TIMEOUT: int = 60
TYPHOON_MAX_RETRIES: int = 3

AI_TIMEZONE: str = os.getenv("AI_TIMEZONE", "Asia/Bangkok")
DEFAULT_USER_TIMEZONE: str = "Asia/Bangkok"
AI_COUNTRY: str = os.getenv("AI_COUNTRY", "Thailand")
AI_CITY: str = os.getenv("AI_CITY", "Bangkok")

DATABASE_PATH: str = os.getenv("DATABASE_PATH", "data/companion.db")
DATA_DIR: str = os.getenv("DATA_DIR", "data")
BACKUP_DIR: str = os.getenv("BACKUP_DIR", "data/backups")
BACKUP_INTERVAL_HOURS: int = 6
BACKUP_MAX_COUNT: int = 48

EXTERNAL_DATABASE_URL: Optional[str] = os.getenv("EXTERNAL_DATABASE_URL", None)
ENABLE_EXTERNAL_BACKUP: bool = os.getenv("ENABLE_EXTERNAL_BACKUP", "false").lower() == "true"
SUPABASE_SERVICE_KEY: Optional[str] = os.getenv("SUPABASE_SERVICE_KEY", None)

DEFAULT_CHARACTER_NAME: str = os.getenv("CHARACTER_NAME", "Ellie")
DEFAULT_CHARACTER_GENDER: str = os.getenv("CHARACTER_GENDER", "female")

DEFAULT_PERSONALITY: Dict[str, Any] = {
    "base_traits": {
        "responsibility": 0.5,
        "social_butterfly": 0.5,
        "anxiety_and_insecurity": 0.2,
        "jealousy_tendency": 0.3,
        "loyalty": 0.7,
        "patience": 0.6,
        "playfulness": 0.5,
        "communication_style": 0.5,
    },
    "sliders": {
        "needy_multiplier": 1.2,
        "typing_speed_modifier": 1.0,
        "proactive_texting_frequency": 0.5,
        "response_delay_multiplier": 1.0,
        "forgiveness_rate": 0.5,
    },
}

TYPING_BASE_DELAY_PER_CHAR: float = 0.08
MIN_TYPING_DELAY: float = 0.5
MAX_TYPING_DELAY: float = 8.0

# Presence / Online status
PRESENCE_WORKER_INTERVAL: int = 30
PRESENCE_SESSION_MIN: int = 120       # minimum online session (seconds)
PRESENCE_SESSION_MAX: int = 600       # maximum online session (seconds)
PRESENCE_OFFLINE_MIN: int = 180       # minimum offline period (seconds)
PRESENCE_OFFLINE_MAX: int = 1800      # maximum offline period (seconds)
ONLINE_BASE_CHANCE: float = 0.35      # base probability AI is online at any time
CHECK_NOTIFICATION_BASE_CHANCE: float = 0.55  # base chance AI checks notification when offline
OFFLINE_CHECK_DELAY_MIN: float = 4.0    # delay when AI checks notification (seconds)
OFFLINE_CHECK_DELAY_MAX: float = 10.0
OFFLINE_IGNORE_DELAY_MIN: float = 20.0  # delay when AI doesn't check immediately
OFFLINE_IGNORE_DELAY_MAX: float = 60.0

GHOSTING_CHECK_INTERVAL: int = 300
LIFESTYLE_TICK_INTERVAL: int = 60
MEMORY_CRON_HOUR: int = 3
EVENT_INJECTION_CHANCE: float = 0.2
KEEPALIVE_INTERVAL: int = 900
PING_TIMEOUT: int = 10
PROACTIVE_TEXT_CHECK_INTERVAL: int = 300
PROACTIVE_MIN_INTERVAL_HOURS: int = 3

RAPID_FIRE_WINDOW_SECONDS: float = 3.0
RAPID_FIRE_MAX_BEFORE_SKIP: int = 5
JEALOUSY_TEST_INTERVAL: int = 7200
ACTIVITY_BLOCK_CHECK_INTERVAL: int = 30

NEGLECT_MUTATION_THRESHOLD: int = 50
NURTURE_MUTATION_RECOVERY_THRESHOLD: int = 100
TRAUMA_MUTATION_COOLDOWN_DAYS: int = 14
DAILY_AFFINITY_CAP: float = 5.0
AFFINITY_DECAY_PER_DAY: float = 0.5
TRUST_DECAY_PER_DAY: float = 0.2

RELATIONSHIP_STAGES: List[str] = [
    "Hate", "Dislike", "Stranger", "Acquaintance",
    "Friend", "Close_Friend", "Crush", "Dating",
    "Lover", "Fiance", "Spouse",
]

RELATIONSHIP_AFFINITY_THRESHOLDS: Dict[str, float] = {
    "Hate": -100,
    "Dislike": -50,
    "Stranger": 0,
    "Acquaintance": 10,
    "Friend": 25,
    "Close_Friend": 40,
    "Crush": 55,
    "Dating": 70,
    "Lover": 85,
    "Fiance": 95,
    "Spouse": 110,
}

MOOD_STATES: List[str] = [
    "happy", "tired", "annoyed", "clingy", "pouty",
    "anxious", "sad", "playful", "romantic", "jealous",
    "grateful", "lonely", "excited", "moody",
]

ACTIVITY_TYPES: Dict[str, int] = {
    "relaxing": 0,
    "showering": 15,
    "eating": 20,
    "commuting": 30,
    "working": 60,
    "studying": 60,
    "shopping": 45,
    "cooking": 30,
    "exercising": 40,
    "sleeping": 480,
    "movie": 120,
    "with_friends": 90,
    "on_phone": 15,
    "reading": 30,
}

EVENT_TYPES: List[str] = [
    "none", "peer_pressure", "transit_delay", "battery_low",
    "third_party_trigger", "weather_surprise", "lost_item",
    "bumped_into_ex", "family_call",
]

TEXT_SPLIT_PATTERN: str = r'(?<=[.!?])\s+|(?<=[,;:])\s+|(?<=ก็)\s+|(?<=นะ)\s+|(?<=คะ)\s+|(?<=ครับ)\s+|(?<=จ้ะ)\s+'

GRADIO_THEME: str = "soft"

VERSION: str = "3.0.2"

USER_INTENTS: Dict[str, List[str]] = {
    "going_to_sleep": ["sleep", "眠", "going to bed", "good night", "ฝันดี", "นอน", "เข้านอน"],
    "going_to_work": ["work", "office", "job", "meeting", "ไปทำงาน", "ทำงาน", "เข้าออฟฟิศ"],
    "going_to_study": ["study", "class", "school", "university", "เรียน", "สอบ", "เข้าเรียน"],
    "going_out": ["going out", "hang out", "meet friends", "party", "ออกไป", "ไปเที่ยว"],
    "going_to_shower": ["shower", "bath", "อาบน้ำ", "อาบ"],
    "going_to_eat": ["eat", "food", "กินข้าว", "กิน", "ทานข้าว"],
    "going_to_commute": ["driving", "commute", "on the way", "เดินทาง", "ขับรถ"],
    "apology": ["sorry", "ขอโทษ", "โทษที", "sorry na", "โทษนะ"],
}

LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT: str = "{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name} | {message}"

HOLIDAYS: Dict[str, Dict[str, Any]] = {
    "valentines_day": {"month": 2, "day": 14, "thai": "วันวาเลนไทน์", "behavior": "romantic"},
    "christmas": {"month": 12, "day": 25, "thai": "วันคริสต์มาส", "behavior": "festive"},
    "new_year": {"month": 1, "day": 1, "thai": "วันปีใหม่", "behavior": "festive"},
    "songkran": {"month": 4, "day": 13, "thai": "วันสงกรานต์", "behavior": "playful"},
    "user_birthday": {"month": None, "day": None, "thai": "วันเกิด", "behavior": "celebratory"},
}

THIRD_PARTY_NAMES: List[str] = [
    "a senior from the club", "an old classmate", "a colleague from work",
    "a neighbor", "someone from the gym", "a friend's friend",
]

WEATHER_API_ENABLED: bool = os.getenv("WEATHER_API_ENABLED", "false").lower() == "true"
WEATHER_API_KEY: Optional[str] = os.getenv("WEATHER_API_KEY", None)
