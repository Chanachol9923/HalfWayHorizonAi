import asyncio
import json
import random
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List, Callable, Awaitable

import httpx
import pytz
from loguru import logger

import config
import database
from brain import MasterStateBuilder, WorldEngine, TyphoonClient, DualTyphoonOrchestrator, PresenceManager


class TextSplitter:
    @staticmethod
    def split(text: str) -> List[str]:
        text = text.strip()
        if not text:
            return []

        parts = re.split(r'\s*\|\|\s*', text)
        if len(parts) > 1:
            return [p.strip() for p in parts if p.strip()]

        segments = re.split(config.TEXT_SPLIT_PATTERN, text)
        buckets: List[str] = []
        current = ""
        for seg in segments:
            seg = seg.strip()
            if not seg:
                continue
            if len(current) + len(seg) + 1 <= 200:
                current = (current + " " + seg).strip() if current else seg
            else:
                if current:
                    buckets.append(current)
                current = seg
        if current:
            buckets.append(current)
        if not buckets:
            buckets = [text]
        return buckets

    @staticmethod
    def should_double_text(buckets: List[str], personality: Dict[str, Any]) -> bool:
        if len(buckets) <= 1:
            return False
        needy = personality.get("sliders", {}).get("needy_multiplier", 1.0)
        anxiety = personality.get("base_traits", {}).get("anxiety_and_insecurity", 0.2)
        patience = personality.get("base_traits", {}).get("patience", 0.6)
        impulsive_roll = random.random()
        base_chance = 0.3 * needy * (1 + anxiety) * (1 - patience * 0.5)
        return impulsive_roll < base_chance


class TypingSimulator:
    def __init__(self) -> None:
        self._typing_callbacks: List[Callable[[], Awaitable[None]]] = []

    def register_typing_callback(self, cb: Callable[[], Awaitable[None]]) -> None:
        self._typing_callbacks.append(cb)

    async def simulate_typing(self, text: str, speed_modifier: float = 1.0) -> None:
        delay = len(text) * config.TYPING_BASE_DELAY_PER_CHAR * speed_modifier
        delay = min(max(delay, config.MIN_TYPING_DELAY), config.MAX_TYPING_DELAY)
        for cb in self._typing_callbacks:
            try:
                await cb()
            except Exception:
                pass
        await asyncio.sleep(delay)

    async def simulate_inter_message_pause(self) -> None:
        await asyncio.sleep(config.INTER_MESSAGE_DELAY)


class DoubleTextPipeline:
    def __init__(self) -> None:
        self.typing_sim = TypingSimulator()
        self._send_callbacks: List[Callable[[str], Awaitable[None]]] = []

    def register_send_callback(self, cb: Callable[[str], Awaitable[None]]) -> None:
        self._send_callbacks.append(cb)

    async def execute(self, raw_text: str, personality: Dict[str, Any]) -> List[str]:
        buckets = TextSplitter.split(raw_text)
        speed_mod = personality.get("sliders", {}).get("typing_speed_modifier", 1.0)
        for i, bucket in enumerate(buckets):
            await self.typing_sim.simulate_typing(bucket, speed_mod)
            for cb in self._send_callbacks:
                try:
                    await cb(bucket)
                except Exception as e:
                    logger.error(f"Send callback failed: {e}")
            if i < len(buckets) - 1:
                await self.typing_sim.simulate_inter_message_pause()
        return buckets


class LifestyleSimulator:
    def __init__(self) -> None:
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Lifestyle simulator started")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Lifestyle simulator stopped")

    async def _run_loop(self) -> None:
        while self._running:
            try:
                await self._tick()
            except Exception as e:
                logger.error(f"Lifestyle tick error: {e}")
            await asyncio.sleep(config.LIFESTYLE_TICK_INTERVAL)

    async def _tick(self) -> None:
        profiles = await database.get_character_profiles()
        if not profiles:
            return
        for profile in profiles:
            try:
                await self._process_character(profile["character_id"])
            except Exception as e:
                logger.error(f"Char tick error {profile['character_id']}: {e}")

    async def _process_character(self, character_id: str) -> None:
        itinerary = await database.get_active_itinerary("default", character_id)
        if not itinerary:
            itinerary = self._generate_daily_itinerary()
            await database.save_itinerary(
                plan_name=itinerary["plan_name"],
                phases=itinerary["phases"],
                character_id=character_id,
            )
        phases = itinerary["phases"]
        current_idx = itinerary["current_phase_index"]
        if current_idx >= len(phases):
            return
        current_phase = phases[current_idx]
        profile = await database.get_character_profile(character_id)
        char_tz_str = profile["timezone"] if profile else config.AI_TIMEZONE
        ai_tz = pytz.timezone(char_tz_str)
        now = datetime.now(ai_tz)
        current_time_str = now.strftime("%H:%M")

        if current_phase["status"] == "completed":
            if current_idx + 1 < len(phases):
                next_phase = phases[current_idx + 1]
                if current_time_str >= next_phase["start"]:
                    await database.update_itinerary_phase(current_idx + 1, "active", character_id=character_id)
                    await self._on_phase_change(current_idx + 1, next_phase, character_id)
            return

        if current_phase["status"] == "active":
            if current_time_str >= current_phase["end"]:
                await database.update_itinerary_phase(current_idx, "completed", character_id=character_id)
                if current_idx + 1 < len(phases):
                    await database.update_itinerary_phase(current_idx + 1, "active", character_id=character_id)
                    await self._on_phase_change(current_idx + 1, phases[current_idx + 1], character_id)

    def _generate_daily_itinerary(self) -> Dict[str, Any]:
        ai_tz = pytz.timezone(config.AI_TIMEZONE)
        now = datetime.now(ai_tz)
        today_str = now.strftime("%Y-%m-%d")
        phases = [
            {"phase": "preparing", "start": f"{random.randint(6, 8):02d}:00", "end": f"{random.randint(8, 9):02d}:00", "status": "pending"},
            {"phase": "going_there", "start": f"{random.randint(8, 9):02d}:00", "end": f"{random.randint(9, 10):02d}:00", "status": "pending"},
            {"phase": "main_activity", "start": f"{random.randint(9, 10):02d}:00", "end": f"{random.randint(16, 18):02d}:00", "status": "pending"},
            {"phase": "returning", "start": f"{random.randint(16, 18):02d}:00", "end": f"{random.randint(17, 19):02d}:00", "status": "pending"},
        ]
        current_time = now.strftime("%H:%M")
        for i, phase in enumerate(phases):
            if current_time >= phase["start"]:
                phases[i]["status"] = "active" if current_time < phase["end"] else "completed"
                if i > 0 and phases[i - 1]["status"] == "pending":
                    phases[i - 1]["status"] = "completed"
        active_idx = next((i for i, p in enumerate(phases) if p["status"] == "active"), 0)
        return {"plan_name": f"Daily Routine {today_str}", "phases": phases, "current_phase_index": active_idx}

    async def _on_phase_change(self, phase_index: int, phase: Dict[str, Any], character_id: str) -> None:
        logger.info(f"Phase changed to {phase['phase']} (index {phase_index}) for {character_id}")
        conflict_checker = PromiseConflictChecker()
        await conflict_checker.evaluate(phase_index, phase, character_id=character_id)


class PromiseConflictChecker:
    async def evaluate(
        self,
        phase_index: int,
        phase: Dict[str, Any],
        user_id: str = "default",
        character_id: str = "default",
    ) -> Dict[str, Any]:
        promises = await database.get_active_promises(user_id, character_id)
        if not promises:
            return {"decision": "no_promises", "broken": False}
        dna = await database.get_personality_dna(user_id, character_id)
        event_triggered = random.random() < config.EVENT_INJECTION_CHANCE
        result = {"decision": "maintain_plan", "broken": False, "event": None}
        if event_triggered:
            event_type = random.choice(config.EVENT_TYPES[1:])
            event_desc = self._generate_event_description(event_type)
            result["event"] = {"type": event_type, "description": event_desc}
            responsibility = dna.get("responsibility", 0.5)
            social = dna.get("social_butterfly", 0.5)
            loyalty = dna.get("loyalty", 0.7)
            break_chance = (1 - responsibility) * 0.4 + social * 0.3 + (1 - loyalty) * 0.3
            if random.random() < break_chance:
                result["decision"] = "break_promise_temporarily"
                result["broken"] = True
                for promise in promises:
                    await database.update_promise_breaking(promise["promise_id"], True)
                neglect_inc = max(1, int((1 - responsibility) * 5))
                psych = await database.get_psychological_state(user_id, character_id)
                psych["neglect_points"] = psych.get("neglect_points", 0) + neglect_inc
                await database.upsert_psychological_state(user_id, psych, character_id)
                logger.warning(f"Promise BROKEN for {user_id}/{character_id}: {event_type}")
            else:
                logger.info(f"Promise maintained for {user_id}/{character_id}: {event_type}")
        return result

    def _generate_event_description(self, event_type: str) -> str:
        descriptions = {
            "peer_pressure": random.choice(["Friends insist I stay longer at the cafe", "Everyone's going for dinner and won't take no", "Friends teasing me for leaving early"]),
            "transit_delay": random.choice(["Train delayed by 20 minutes", "Traffic is insane", "Bus never showed up"]),
            "battery_low": random.choice(["Phone at 5%", "Battery died, just got back"]),
            "third_party_trigger": random.choice([f"{random.choice(config.THIRD_PARTY_NAMES)} just asked for my LINE", "Someone from work asked if I'm single lol", "This cute guy/girl at the cafe kept looking at me"]),
            "weather_surprise": random.choice(["Suddenly pouring rain", "Super hot today, melting"]),
            "lost_item": random.choice(["Lost my wallet, panicking", "Can't find my keys anywhere"]),
            "bumped_into_ex": random.choice(["Just ran into my ex at the mall... awkward", "Saw my ex with someone new"]),
            "family_call": random.choice(["Mom just called, she won't stop talking", "Family dinner got extended"]),
        }
        return descriptions.get(event_type, f"Unexpected {event_type}")


class ActivityBlockManager:
    def __init__(self) -> None:
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Activity block manager started")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Activity block manager stopped")

    async def _run_loop(self) -> None:
        while self._running:
            try:
                await self._check_expired_blocks()
            except Exception as e:
                logger.error(f"Activity block check error: {e}")
            await asyncio.sleep(config.ACTIVITY_BLOCK_CHECK_INTERVAL)

    async def _check_expired_blocks(self) -> None:
        profiles = await database.get_character_profiles()
        for profile in profiles:
            block = await database.get_active_activity_block(character_id=profile["character_id"])
            if block:
                try:
                    end = datetime.fromisoformat(block["ends_at"])
                    if datetime.now(timezone.utc) >= end:
                        await database.deactivate_activity_block(block["id"])
                        logger.info(f"Activity block expired: {block['activity_type']} for {profile['character_id']}")
                except (ValueError, KeyError):
                    pass

    @staticmethod
    async def create_block(
        activity_type: str,
        character_id: str = "default",
        user_id: str = "default",
        description: str = "",
    ) -> Optional[Dict[str, Any]]:
        duration = config.ACTIVITY_TYPES.get(activity_type)
        if not duration:
            return None
        block = await database.create_activity_block(
            activity_type=activity_type,
            duration_minutes=duration,
            user_id=user_id,
            character_id=character_id,
            description=description or activity_type,
        )
        logger.info(f"Activity block created: {activity_type} ({duration}min) for {character_id}")
        return block


class ProactiveTextWorker:
    def __init__(self, orchestrator: DualTyphoonOrchestrator) -> None:
        self._orchestrator = orchestrator
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._send_callbacks: List[Callable[[str, str], Awaitable[None]]] = []

    def register_send_callback(self, cb: Callable[[str, str], Awaitable[None]]) -> None:
        self._send_callbacks.append(cb)

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Proactive text worker started")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Proactive text worker stopped")

    async def _run_loop(self) -> None:
        while self._running:
            try:
                await self._check_proactive_text()
                await self._check_ghosting_followup()
            except Exception as e:
                logger.error(f"Proactive text error: {e}")
            await asyncio.sleep(config.PROACTIVE_TEXT_CHECK_INTERVAL)

    async def _check_proactive_text(self) -> None:
        profiles = await database.get_character_profiles()
        for profile in profiles:
            char_id = profile["character_id"]
            user_id = profile.get("user_id", "")
            # Only send proactive texts for characters linked to Telegram
            if not user_id.startswith("telegram_"):
                continue
            now = datetime.now(timezone.utc)
            last_raw = await database.get_app_state(f"last_proactive_{char_id}")
            if last_raw:
                last_time = datetime.fromisoformat(last_raw)
                hours_since = (now - last_time).total_seconds() / 3600
                if hours_since < config.PROACTIVE_MIN_INTERVAL_HOURS:
                    continue
            block = await database.get_active_activity_block(character_id=char_id)
            if block:
                continue
            msg = await self._orchestrator.generate_proactive_message(character_id=char_id)
            if msg:
                logger.info(f"Proactive text from {char_id}: {msg[:50]}...")
                for cb in self._send_callbacks:
                    try:
                        await cb(char_id, msg)
                    except Exception as e:
                        logger.error(f"Proactive send callback failed: {e}")
                await database.save_message(
                    role="assistant",
                    content=msg,
                    platform="system",
                    character_id=char_id,
                    metadata={"type": "proactive"},
                )
                await database.set_app_state(f"last_proactive_{char_id}", now.isoformat())
                await self._handle_activity_block_from_message(char_id, msg)

    CRUSH_OR_ABOVE = ["Crush", "Dating", "Lover", "Fiance", "Spouse"]

    async def _check_ghosting_followup(self) -> None:
        profiles = await database.get_character_profiles()
        for profile in profiles:
            char_id = profile["character_id"]
            user_id = profile.get("user_id", "")
            if not user_id.startswith("telegram_"):
                continue
            history = await database.get_recent_history(user_id, char_id, limit=3)
            if not history:
                continue
            # Only if last message was from AI (user hasn't replied)
            if history[-1].get("role") != "assistant":
                continue
            psych = await database.get_psychological_state(user_id, char_id)
            stage = psych.get("relationship_stage", "Stranger")
            if stage not in self.CRUSH_OR_ABOVE:
                continue
            # Check how long since AI's last message
            now = datetime.now(timezone.utc)
            last_raw = await database.get_app_state(f"last_ghosting_check_{char_id}")
            if last_raw:
                last_time = datetime.fromisoformat(last_raw)
                hours_since = (now - last_time).total_seconds() / 3600
                if hours_since < 2:
                    continue
            # 30% chance
            if random.random() > 0.3:
                continue
            await database.set_app_state(f"last_ghosting_check_{char_id}", now.isoformat())
            msg = await self._orchestrator.generate_ghosting_followup(character_id=char_id, user_id=user_id)
            if msg:
                logger.info(f"Ghosting follow-up from {char_id}: {msg[:50]}...")
                for cb in self._send_callbacks:
                    try:
                        await cb(char_id, msg)
                    except Exception as e:
                        logger.error(f"Ghosting follow-up send failed: {e}")
                await database.save_message(
                    role="assistant", content=msg, platform="system",
                    character_id=char_id, metadata={"type": "ghosting_followup"},
                )

    async def _handle_activity_block_from_message(self, character_id: str, message: str) -> None:
        msg_lower = message.lower()
        activity_map = {
            "shower": "showering", "bath": "showering", "อาบน้ำ": "showering",
            "eat": "eating", "food": "eating", "กิน": "eating",
            "cook": "cooking", "ทำอาหาร": "cooking",
            "sleep": "sleeping", "นอน": "sleeping", "bed": "sleeping",
            "work": "working", "ทำงาน": "working",
            "study": "studying", "เรียน": "studying",
            "commute": "commuting", "drive": "commuting", "เดินทาง": "commuting",
            "shop": "shopping", "ช้อป": "shopping",
            "exercise": "exercising", "workout": "exercising", "ออกกำลัง": "exercising",
        }
        for keyword, activity in activity_map.items():
            if keyword in msg_lower:
                await ActivityBlockManager.create_block(activity, character_id=character_id)
                break

    async def send_activity_blocking_reply(
        self, activity_type: str, character_id: str, user_id: str = "default"
    ) -> Optional[str]:
        block = await ActivityBlockManager.create_block(activity_type, character_id, user_id)
        if not block:
            return None
        replies = {
            "showering": "อาบน้ำอยู่ แป๊บนะ 🙈",
            "eating": "กำลังกินข้าวอยู่ เดี๋ยวคุยกันนะ 😋",
            "commuting": "เดินทางอยู่ค่ะ ตอบช้าหน่อยนะ 🚗",
            "sleeping": "ZZZ... 😴💤",
            "working": "ทำงานอยู่ ตอบช้าหน่อยนะ 😅",
            "studying": "กำลังเรียนอยู่ 😅",
            "cooking": "ทำกับข้าวอยู่ค่ะ 🍳",
            "exercising": "ออกกำลังกายอยู่ค่ะ 😮‍💨",
            "shopping": "ช้อปปิ้งอยู่ 😅",
            "with_friends": "ออกไปข้างนอกกับเพื่อนๆ ค่ะ 🥳",
            "movie": "ดูหนังอยู่ค่ะ 🎬",
        }
        return replies.get(activity_type)


class JealousyTestScheduler:
    def __init__(self, orchestrator: DualTyphoonOrchestrator) -> None:
        self._orchestrator = orchestrator
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._send_callbacks: List[Callable[[str, str], Awaitable[None]]] = []

    def register_send_callback(self, cb: Callable[[str, str], Awaitable[None]]) -> None:
        self._send_callbacks.append(cb)

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Jealousy test scheduler started")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Jealousy test scheduler stopped")

    async def _run_loop(self) -> None:
        while self._running:
            try:
                await self._check_jealousy_tests()
            except Exception as e:
                logger.error(f"Jealousy test error: {e}")
            await asyncio.sleep(config.JEALOUSY_TEST_INTERVAL)

    async def _check_jealousy_tests(self) -> None:
        profiles = await database.get_character_profiles()
        for profile in profiles:
            char_id = profile["character_id"]
            psych = await database.get_psychological_state(character_id=char_id)
            stage = psych.get("relationship_stage", "Stranger")
            if stage not in ("Crush", "Dating", "Lover", "Fiance", "Spouse"):
                continue
            block = await database.get_active_activity_block(character_id=char_id)
            if block:
                continue
            msg = await self._orchestrator.generate_jealousy_test(character_id=char_id)
            if msg:
                logger.info(f"Jealousy test from {char_id}: {msg[:50]}...")
                for cb in self._send_callbacks:
                    try:
                        await cb(char_id, msg)
                    except Exception as e:
                        logger.error(f"Jealousy send callback failed: {e}")
                await database.save_message(
                    role="assistant",
                    content=msg,
                    platform="system",
                    character_id=char_id,
                    metadata={"type": "jealousy_test"},
                )


class ActivityBlockingReplyHandler:
    @staticmethod
    async def handle_user_message(user_message: str, character_id: str, user_id: str = "default") -> Optional[str]:
        msg_lower = user_message.lower()
        block = await database.get_active_activity_block(character_id=character_id)
        if block:
            return None
        activity_map = {
            "shower": "showering", "bath": "showering", "อาบน้ำ": "showering",
            "eat": "eating", "กินข้าว": "eating",
            "sleep": "sleeping", "นอน": "sleeping",
            "cook": "cooking", "ทำอาหาร": "cooking",
        }
        for keyword, activity in activity_map.items():
            if keyword in msg_lower:
                await ActivityBlockManager.create_block(activity, character_id, user_id)
                return None
        return None


class MemoryConsolidationCron:
    def __init__(self, typhoon: Optional[TyphoonClient] = None) -> None:
        self._typhoon = typhoon or TyphoonClient()
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info(f"Memory consolidation cron started (runs at {config.MEMORY_CRON_HOUR}:00 daily)")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Memory consolidation cron stopped")

    async def _run_loop(self) -> None:
        while self._running:
            try:
                now = datetime.now(pytz.timezone(config.AI_TIMEZONE))
                target_hour = config.MEMORY_CRON_HOUR
                next_run = now.replace(hour=target_hour, minute=0, second=0, microsecond=0)
                if now >= next_run:
                    next_run = next_run + timedelta(days=1)
                wait_seconds = (next_run - now).total_seconds()
                logger.info(f"Next memory consolidation at {next_run.isoformat()} ({wait_seconds/3600:.1f}h)")
                await asyncio.sleep(wait_seconds)
                if self._running:
                    await self._consolidate_all()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Memory consolidation error: {e}")
                await asyncio.sleep(60)

    async def _consolidate_all(self) -> None:
        profiles = await database.get_character_profiles()
        for profile in profiles:
            try:
                await self._consolidate(character_id=profile["character_id"])
            except Exception as e:
                logger.error(f"Consolidation failed for {profile['character_id']}: {e}")

    async def _consolidate(self, user_id: str = "default", character_id: str = "default") -> None:
        logger.info(f"Starting memory consolidation for {user_id}/{character_id}")
        raw_data = await database.consolidate_daily_memories(user_id, character_id)
        factual_texts = raw_data["factual"]
        if not factual_texts:
            logger.info("No conversations to consolidate")
            return
        combined = "\n".join(factual_texts[-20:])
        try:
            summary = await self._typhoon.generate(
                messages=[
                    {
                        "role": "system",
                        "content": """You are a memory consolidation engine for an AI companion. Extract:
1) FACTUAL: Concrete facts about the user (work, habits, preferences, plans, important dates)
2) EMOTIONAL: Emotional high/low points and strong reactions
3) SUBTEXT: Hidden meanings, unspoken tensions, or important relationship dynamics

Output as JSON:
{"factual": ["fact1"], "emotional": ["emotion1"], "subtext": ["subtext1"]}""",
                    },
                    {"role": "user", "content": f"Consolidate these messages:\n{combined}"},
                ],
                model=config.TYPHOON_MODEL_WORLD,
                temperature=0.3,
                max_tokens=1024,
                response_format={"type": "json_object"},
            )
            result = json.loads(summary)
        except Exception as e:
            logger.warning(f"LLM consolidation failed, using fallback: {e}")
            result = {"factual": factual_texts[-3:], "emotional": [], "subtext": []}

        for fact in result.get("factual", []):
            if len(fact) > 20:
                await database.save_crystallized_memory("factual", fact, importance=0.7, character_id=character_id)
        for emotion in result.get("emotional", []):
            if len(emotion) > 20:
                await database.save_crystallized_memory("emotional", emotion, importance=0.9, character_id=character_id)
        for subtext in result.get("subtext", []):
            if len(subtext) > 20:
                await database.save_crystallized_memory("subtext", subtext, importance=0.8, character_id=character_id)

        await database.deactivate_old_memories(character_id=character_id, keep_count=50)
        await database.create_backup()
        logger.info(f"Memory consolidation complete for {character_id}: {len(result.get('factual',[]))} factual, {len(result.get('emotional',[]))} emotional")


class KeepAlivePinger:
    def __init__(self, port: int = 7860) -> None:
        self.port = port
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info(f"Keep-alive pinger started (port {self.port}, interval {config.KEEPALIVE_INTERVAL}s)")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Keep-alive pinger stopped")

    async def _run_loop(self) -> None:
        while self._running:
            try:
                async with httpx.AsyncClient(timeout=config.PING_TIMEOUT) as client:
                    resp = await client.get(f"http://127.0.0.1:{self.port}/")
                    logger.debug(f"Keep-alive ping: {resp.status_code}")
            except Exception as e:
                logger.debug(f"Keep-alive ping failed: {e}")
            jitter = random.randint(-120, 120)
            await asyncio.sleep(config.KEEPALIVE_INTERVAL + jitter)


class NeglectTracker:
    @staticmethod
    async def check_ghosting(user_id: str = "default", character_id: str = "default") -> None:
        psych = await database.get_psychological_state(user_id, character_id)
        dna = await database.get_personality_dna(user_id, character_id)
        threshold = dna.get("ghosting_threshold_hours", 4)
        history = await database.get_recent_history(user_id, character_id, limit=2)
        if len(history) < 2:
            return
        last_msg = history[-1]
        if last_msg.get("role") != "user":
            return
        ai_responded = any(m.get("role") == "assistant" for m in history[-4:])
        if ai_responded:
            return
        psych["neglect_points"] = psych.get("neglect_points", 0) + 1
        psych["trust_score"] = max(0, psych.get("trust_score", 100) - 1)
        psych["affinity_score"] = max(-50, psych.get("affinity_score", 0) - 0.5)
        moods = ["annoyed", "pouty", "sad"]
        if psych["neglect_points"] > threshold * 2:
            psych["short_term_mood"] = random.choice(moods)
        await database.upsert_psychological_state(user_id, psych, character_id)
        logger.debug(f"Neglect+1 for {user_id}/{character_id} (total: {psych['neglect_points']})")

    @staticmethod
    async def reward_nurture(user_id: str = "default", character_id: str = "default", points: int = 1) -> None:
        psych = await database.get_psychological_state(user_id, character_id)
        psych["nurture_points"] = psych.get("nurture_points", 0) + points
        psych["trust_score"] = min(100, psych.get("trust_score", 100) + 0.5)
        psych["affinity_score"] = min(100, psych.get("affinity_score", 0) + 0.5)
        await database.upsert_psychological_state(user_id, psych, character_id)
        logger.debug(f"Nurture+{points} for {user_id}/{character_id}")


class GhostingDetector:
    def __init__(self) -> None:
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Ghosting detector started")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Ghosting detector stopped")

    async def _run_loop(self) -> None:
        while self._running:
            try:
                profiles = await database.get_character_profiles()
                for profile in profiles:
                    await NeglectTracker.check_ghosting(character_id=profile["character_id"])
            except Exception as e:
                logger.error(f"Ghosting check error: {e}")
            await asyncio.sleep(config.GHOSTING_CHECK_INTERVAL)


class AffinityDecayWorker:
    def __init__(self) -> None:
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Affinity decay worker started")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Affinity decay worker stopped")

    async def _run_loop(self) -> None:
        while self._running:
            try:
                await self._apply_decay()
            except Exception as e:
                logger.error(f"Affinity decay error: {e}")
            await asyncio.sleep(86400)

    async def _apply_decay(self) -> None:
        profiles = await database.get_character_profiles()
        for profile in profiles:
            psych = await database.get_psychological_state(character_id=profile["character_id"])
            last_reset = psych.get("last_affinity_reset")
            if last_reset:
                try:
                    last_date = datetime.fromisoformat(last_reset)
                    if datetime.now(timezone.utc).date() > last_date.date():
                        psych["affinity_score"] = max(-100, psych.get("affinity_score", 0) - config.AFFINITY_DECAY_PER_DAY)
                        psych["trust_score"] = max(0, psych.get("trust_score", 100) - config.TRUST_DECAY_PER_DAY)
                        await database.upsert_psychological_state("default", psych, profile["character_id"])
                        logger.debug(f"Daily decay applied for {profile['character_id']}")
                except (ValueError, TypeError):
                    pass


class PresenceSimulator:
    def __init__(self) -> None:
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Presence simulator started")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Presence simulator stopped")

    async def _run_loop(self) -> None:
        while self._running:
            try:
                await self._tick()
            except Exception as e:
                logger.error(f"Presence tick error: {e}")
            await asyncio.sleep(config.PRESENCE_WORKER_INTERVAL)

    async def _tick(self) -> None:
        profiles = await database.get_character_profiles()
        if not profiles:
            return
        for profile in profiles:
            try:
                await self._process_character(profile["character_id"])
            except Exception as e:
                logger.error(f"Presence error for {profile['character_id']}: {e}")

    async def _process_character(self, character_id: str) -> None:
        presence = await PresenceManager.get_presence(character_id)
        now_ts = datetime.now(timezone.utc).isoformat()

        if presence.get("is_online"):
            change_at = presence.get("next_change_at")
            if change_at and change_at <= now_ts:
                psych = await database.get_psychological_state(character_id=character_id)
                dna = await database.get_personality_dna(character_id=character_id)
                offline_dur = PresenceManager._offline_duration(psych, dna)
                offline_end = datetime.fromisoformat(now_ts) + timedelta(seconds=offline_dur)
                presence["is_online"] = False
                presence["next_change_at"] = offline_end.isoformat()
                presence["last_seen"] = now_ts
                await PresenceManager.set_presence(character_id, presence)
        else:
            change_at = presence.get("next_change_at")
            if not change_at or change_at <= now_ts:
                psych = await database.get_psychological_state(character_id=character_id)
                dna = await database.get_personality_dna(character_id=character_id)
                online_chance = PresenceManager._online_probability(psych, dna)
                if random.random() < online_chance:
                    session_dur = PresenceManager._session_duration(psych, dna)
                    session_end = datetime.fromisoformat(now_ts) + timedelta(seconds=session_dur)
                    presence["is_online"] = True
                    presence["online_since"] = now_ts
                    presence["next_change_at"] = session_end.isoformat()
                    presence["last_seen"] = now_ts
                else:
                    offline_dur = PresenceManager._offline_duration(psych, dna)
                    next_check = datetime.fromisoformat(now_ts) + timedelta(seconds=offline_dur)
                    presence["next_change_at"] = next_check.isoformat()
                await PresenceManager.set_presence(character_id, presence)


class WorkerSupervisor:
    def __init__(self) -> None:
        self.orchestrator = DualTyphoonOrchestrator()
        self.lifestyle = LifestyleSimulator()
        self.memory_cron = MemoryConsolidationCron()
        self.keepalive = KeepAlivePinger(port=config.GRADIO_PORT)
        self.ghosting = GhostingDetector()
        self.activity_blocks = ActivityBlockManager()
        self.proactive_text = ProactiveTextWorker(self.orchestrator)
        self.jealousy_scheduler = JealousyTestScheduler(self.orchestrator)
        self.affinity_decay = AffinityDecayWorker()
        self.text_pipeline = DoubleTextPipeline()
        self.presence = PresenceSimulator()
        self._tasks: List[asyncio.Task] = []

    async def start_all(self) -> None:
        self._tasks = [
            asyncio.create_task(self.lifestyle.start()),
            asyncio.create_task(self.memory_cron.start()),
            asyncio.create_task(self.keepalive.start()),
            asyncio.create_task(self.ghosting.start()),
            asyncio.create_task(self.activity_blocks.start()),
            asyncio.create_task(self.proactive_text.start()),
            asyncio.create_task(self.jealousy_scheduler.start()),
            asyncio.create_task(self.affinity_decay.start()),
            asyncio.create_task(self.presence.start()),
            asyncio.create_task(database.periodic_backup_task()),
        ]
        logger.info("All background workers started")

    async def stop_all(self) -> None:
        await self.lifestyle.stop()
        await self.memory_cron.stop()
        await self.keepalive.stop()
        await self.ghosting.stop()
        await self.activity_blocks.stop()
        await self.proactive_text.stop()
        await self.jealousy_scheduler.stop()
        await self.affinity_decay.stop()
        await self.presence.stop()
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        await self.orchestrator.close()
        logger.info("All background workers stopped")
