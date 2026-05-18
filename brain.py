import asyncio
import json
import re
import random
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List, Tuple

import httpx
import pytz
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

import config
import database


class TyphoonClient:
    def __init__(self) -> None:
        self.api_base = config.TYPHOON_API_BASE
        self.api_key = config.TYPHOON_API_KEY
        self.timeout = config.TYPHOON_TIMEOUT
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.api_base,
                timeout=self.timeout,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
        return self._client

    @retry(
        stop=stop_after_attempt(config.TYPHOON_MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(
            (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError)
        ),
    )
    async def generate(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.85,
        max_tokens: int = 1024,
        response_format: Optional[Dict[str, str]] = None,
    ) -> str:
        client = await self._get_client()
        payload: Dict[str, Any] = {
            "model": model or config.TYPHOON_MODEL_CHAT,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            payload["response_format"] = response_format

        for attempt in range(config.TYPHOON_MAX_RETRIES):
            try:
                resp = await client.post("/chat/completions", json=payload)
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    wait = min(2 ** attempt * 2, 30)
                    logger.warning(f"Rate limited, retrying in {wait}s")
                    await asyncio.sleep(wait)
                    continue
                raise
            except (httpx.TimeoutException, httpx.NetworkError) as e:
                logger.warning(f"API call failed (attempt {attempt + 1}): {e}")
                if attempt < config.TYPHOON_MAX_RETRIES - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise

        raise RuntimeError("All Typhoon API retry attempts exhausted")

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()


class PresenceManager:
    PRESENCE_PREFIX = "presence"

    @staticmethod
    def _key(character_id: str) -> str:
        return f"{PresenceManager.PRESENCE_PREFIX}_{character_id}"

    @staticmethod
    async def get_presence(character_id: str) -> Dict[str, Any]:
        raw = await database.get_app_state(PresenceManager._key(character_id))
        if raw is None:
            default = {
                "is_online": False,
                "next_change_at": None,
                "last_seen": None,
                "online_since": None,
            }
            await PresenceManager.set_presence(character_id, default)
            return default
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {"is_online": False, "next_change_at": None, "last_seen": None, "online_since": None}

    @staticmethod
    async def set_presence(character_id: str, presence: Dict[str, Any]) -> None:
        await database.set_app_state(PresenceManager._key(character_id), json.dumps(presence))

    @staticmethod
    def _online_probability(psych: Dict[str, Any], dna: Dict[str, Any]) -> float:
        affinity = max(0, psych.get("affinity_score", 0))
        stage_idx = config.RELATIONSHIP_STAGES.index(psych.get("relationship_stage", "Stranger"))
        base = config.ONLINE_BASE_CHANCE
        affinity_bonus = (affinity / 100) * 0.3
        stage_bonus = (stage_idx / len(config.RELATIONSHIP_STAGES)) * 0.15
        social = dna.get("social_butterfly", 0.5) * 0.15
        anxiety = dna.get("anxiety_and_insecurity", 0.2) * 0.1
        needy = dna.get("needy_multiplier", 1.2) * 0.05
        return min(0.95, base + affinity_bonus + stage_bonus + social + anxiety + needy)

    @staticmethod
    def _check_probability(psych: Dict[str, Any], dna: Dict[str, Any]) -> float:
        affinity = max(0, psych.get("affinity_score", 0))
        base = config.CHECK_NOTIFICATION_BASE_CHANCE
        affinity_bonus = (affinity / 100) * 0.25
        social = dna.get("social_butterfly", 0.5) * 0.15
        anxiety = dna.get("anxiety_and_insecurity", 0.2) * 0.15
        needy = dna.get("needy_multiplier", 1.2) * 0.05
        patience = dna.get("patience", 0.6) * -0.1
        return min(0.95, max(0.1, base + affinity_bonus + social + anxiety + needy + patience))

    @staticmethod
    def _session_duration(psych: Dict[str, Any], dna: Dict[str, Any]) -> float:
        affinity_factor = 1.0 + max(0, psych.get("affinity_score", 0)) / 100
        social_factor = 1.0 + dna.get("social_butterfly", 0.5) * 0.5
        stage_idx = config.RELATIONSHIP_STAGES.index(psych.get("relationship_stage", "Stranger"))
        stage_factor = 1.0 + (stage_idx / len(config.RELATIONSHIP_STAGES)) * 0.5
        base = random.uniform(config.PRESENCE_SESSION_MIN, config.PRESENCE_SESSION_MAX)
        return base * affinity_factor * social_factor * stage_factor

    @staticmethod
    def _offline_duration(psych: Dict[str, Any], dna: Dict[str, Any]) -> float:
        social = dna.get("social_butterfly", 0.5)
        anxiety = dna.get("anxiety_and_insecurity", 0.2)
        base = random.uniform(config.PRESENCE_OFFLINE_MIN, config.PRESENCE_OFFLINE_MAX)
        social_reduction = 1.0 - social * 0.4
        anxiety_reduction = 1.0 - anxiety * 0.3
        return base * social_reduction * anxiety_reduction

    @staticmethod
    async def trigger_user_message(character_id: str) -> Dict[str, Any]:
        presence = await PresenceManager.get_presence(character_id)
        psych = await database.get_psychological_state(character_id=character_id)
        dna = await database.get_personality_dna(character_id=character_id)
        now_ts = datetime.now(timezone.utc).isoformat()

        if presence.get("is_online"):
            return presence

        check_roll = random.random()
        check_chance = PresenceManager._check_probability(psych, dna)
        if check_roll < check_chance:
            presence["is_online"] = True
            presence["online_since"] = now_ts
            duration = PresenceManager._session_duration(psych, dna)
            presence["next_change_at"] = datetime.fromisoformat(now_ts).isoformat()
            presence["last_seen"] = now_ts
        else:
            presence["last_seen"] = presence.get("last_seen", now_ts)
        await PresenceManager.set_presence(character_id, presence)
        return presence

    @staticmethod
    async def calculate_delay_and_context(
        presence: Dict[str, Any],
        psych: Dict[str, Any],
        dna: Dict[str, Any],
    ) -> Tuple[float, str]:
        if presence.get("is_online"):
            speed = dna.get("typing_speed_modifier", 1.0)
            read_delay = random.uniform(0.5, 2.0) * speed
            ctx = "You are currently in the chat app, you saw their message immediately. Reply in real-time."
            return read_delay, ctx

        check_roll = random.random()
        check_chance = PresenceManager._check_probability(psych, dna)
        if check_roll < check_chance:
            read_delay = random.uniform(config.OFFLINE_CHECK_DELAY_MIN, config.OFFLINE_CHECK_DELAY_MAX)
            ctx = "You just got a notification, opened the chat, and saw their message. Respond naturally as if you just checked."
            return read_delay, ctx
        else:
            read_delay = random.uniform(config.OFFLINE_IGNORE_DELAY_MIN, config.OFFLINE_IGNORE_DELAY_MAX)
            ctx = "You haven't checked your phone in a while. You just opened the chat and saw their message. Mention being busy or apologize briefly."
            return read_delay, ctx

    @staticmethod
    def calculate_typing_delay(text: str, dna: Dict[str, Any]) -> float:
        speed = dna.get("typing_speed_modifier", 1.0)
        delay = len(text) * config.TYPING_BASE_DELAY_PER_CHAR * speed
        delay = min(max(delay, config.MIN_TYPING_DELAY), config.MAX_TYPING_DELAY)
        return delay


class WeatherService:
    @staticmethod
    async def get_weather(city: str, country: str) -> Dict[str, str]:
        if not config.WEATHER_API_ENABLED or not config.WEATHER_API_KEY:
            return {"condition": "Clear", "temperature": "N/A", "humidity": "N/A"}
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    f"https://api.openweathermap.org/data/2.5/weather",
                    params={
                        "q": f"{city},{country}",
                        "appid": config.WEATHER_API_KEY,
                        "units": "metric",
                    },
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return {
                        "condition": data.get("weather", [{}])[0].get("main", "Clear"),
                        "temperature": str(round(data.get("main", {}).get("temp", 0))),
                        "humidity": str(data.get("main", {}).get("humidity", 0)),
                    }
        except Exception as e:
            logger.debug(f"Weather fetch failed: {e}")
        return {"condition": "Clear", "temperature": "N/A", "humidity": "N/A"}


class HolidayDetector:
    @staticmethod
    def check_holiday(user_profile: Dict[str, Any], ai_tz: pytz.BaseTzInfo) -> Optional[Dict[str, Any]]:
        now = datetime.now(ai_tz)
        for holiday_name, info in config.HOLIDAYS.items():
            if holiday_name == "user_birthday":
                bday = user_profile.get("birthday")
                if bday:
                    try:
                        bday_dt = datetime.fromisoformat(bday)
                        if bday_dt.month == now.month and bday_dt.day == now.day:
                            return {"name": holiday_name, "behavior": info["behavior"], "thai": info["thai"]}
                    except (ValueError, TypeError):
                        pass
            else:
                if info["month"] == now.month and info["day"] == now.day:
                    return {"name": holiday_name, "behavior": info["behavior"], "thai": info["thai"]}
        return None


class IntentValidator:
    @staticmethod
    def evaluate(message: Optional[str], user_local: datetime) -> Tuple[str, bool, str]:
        if not message:
            return "none", True, ""
        msg_lower = message.lower()
        hour = user_local.hour
        for intent, keywords in config.USER_INTENTS.items():
            if any(kw in msg_lower for kw in keywords):
                if intent == "going_to_sleep":
                    valid = hour >= 20 or hour <= 6
                    reason = "" if valid else f"It's {hour}:00, too early for sleep"
                elif intent in ("going_to_work", "going_to_study"):
                    valid = 5 <= hour <= 12
                    reason = "" if valid else f"It's {hour}:00, unusual time for work/study"
                elif intent == "going_to_shower":
                    valid = hour >= 6
                    reason = "" if valid else "Showering at this hour?"
                elif intent in ("going_out", "going_to_eat", "going_to_commute"):
                    valid = True
                    reason = ""
                elif intent == "apology":
                    valid = True
                    reason = ""
                else:
                    valid = True
                    reason = ""
                return intent, valid, reason
        return "none", True, ""


class SubtextAnalyzer:
    @staticmethod
    async def analyze(
        user_message: str,
        user_local: datetime,
        user_profile: Dict[str, Any],
        character_id: str,
    ) -> Dict[str, Any]:
        intent, is_valid, reason = IntentValidator.evaluate(user_message, user_local)
        analysis = {
            "detected_intent": intent,
            "intent_validity": 1.0 if is_valid else 0.0,
            "analysis": reason,
            "is_suspicious": not is_valid,
        }
        await database.save_subtext_analysis(
            user_message=user_message,
            detected_intent=intent,
            intent_validity=analysis["intent_validity"],
            analysis=reason,
            character_id=character_id,
        )
        return analysis


class MasterStateBuilder:
    @staticmethod
    async def build(
        user_id: str = "default",
        character_id: str = "default",
        user_message: Optional[str] = None,
        platform: str = "gradio",
    ) -> Dict[str, Any]:
        now = datetime.now(timezone.utc)
        psych = await database.get_psychological_state(user_id, character_id)
        dna = await database.get_personality_dna(user_id, character_id)
        settings = await database.get_user_settings(user_id)
        profile = await database.get_user_profile(user_id)
        profile_char = await database.get_character_profile(character_id)
        itinerary = await database.get_active_itinerary(user_id, character_id)
        promises = await database.get_active_promises(user_id, character_id)
        memories = await database.get_active_memories(user_id, character_id)
        history = await database.get_recent_history(user_id, character_id, limit=20)
        activity_block = await database.get_active_activity_block(user_id, character_id)
        subtext_analyses = await database.get_recent_subtext_analyses(user_id, character_id, limit=5)

        user_tz = pytz.timezone(settings.get("timezone", config.DEFAULT_USER_TIMEZONE))
        char_tz_str = profile_char.get("timezone", config.AI_TIMEZONE) if profile_char else config.AI_TIMEZONE
        ai_tz = pytz.timezone(char_tz_str)

        ai_local = now.astimezone(ai_tz)
        user_local = now.astimezone(user_tz)

        subtext = await SubtextAnalyzer.analyze(user_message or "", user_local, profile, character_id) if user_message else {
            "detected_intent": "none", "intent_validity": 1.0, "analysis": "", "is_suspicious": False
        }

        holiday = HolidayDetector.check_holiday(profile, ai_tz)

        weather_ai = await WeatherService.get_weather(
            profile_char.get("city", config.AI_CITY) if profile_char else config.AI_CITY,
            profile_char.get("country", config.AI_COUNTRY) if profile_char else config.AI_COUNTRY,
        )
        weather_user = await WeatherService.get_weather(
            profile.get("country", "Thailand"), profile.get("country", "Thailand")
        )

        char_name = dna.get("character_name", config.DEFAULT_CHARACTER_NAME)
        char_gender = profile_char.get("gender", config.DEFAULT_CHARACTER_GENDER) if profile_char else config.DEFAULT_CHARACTER_GENDER
        char_city = profile_char.get("city", config.AI_CITY) if profile_char else config.AI_CITY
        char_country = profile_char.get("country", config.AI_COUNTRY) if profile_char else config.AI_COUNTRY

        state: Dict[str, Any] = {
            "simulation_metadata": {
                "current_timestamp": now.isoformat(),
                "ai_local_time": ai_local.isoformat(),
                "user_local_time": user_local.isoformat(),
                "ai_time": ai_local.strftime("%H:%M"),
                "user_time": user_local.strftime("%H:%M"),
                "ai_day_of_week": ai_local.strftime("%A"),
                "timezone_difference_hours": round(
                    (user_local.utcoffset().total_seconds() - ai_local.utcoffset().total_seconds()) / 3600, 1
                ),
            },
            "ai_profile": {
                "name": char_name,
                "gender": char_gender,
                "location": f"{char_city}, {char_country}",
                "lore": profile_char.get("lore", "") if profile_char else "",
                "relationship_stage": psych.get("relationship_stage", "Stranger"),
                "affinity_score": round(psych.get("affinity_score", 0), 1),
                "trust_score": round(psych.get("trust_score", 100), 1),
                "daily_affinity_gained": round(psych.get("daily_affinity_gained", 0), 1),
                "personality_dna": {
                    "base_traits": {
                        "responsibility": dna.get("responsibility", 0.5),
                        "social_butterfly": dna.get("social_butterfly", 0.5),
                        "anxiety_and_insecurity": dna.get("anxiety_and_insecurity", 0.2),
                        "jealousy_tendency": dna.get("jealousy_tendency", 0.3),
                        "loyalty": dna.get("loyalty", 0.7),
                        "patience": dna.get("patience", 0.6),
                        "playfulness": dna.get("playfulness", 0.5),
                        "communication_style": dna.get("communication_style", 0.5),
                    },
                    "sliders": {
                        "needy_multiplier": dna.get("needy_multiplier", 1.2),
                        "typing_speed_modifier": dna.get("typing_speed_modifier", 1.0),
                        "proactive_texting_frequency": dna.get("proactive_texting_frequency", 0.5),
                        "response_delay_multiplier": dna.get("response_delay_multiplier", 1.0),
                        "forgiveness_rate": dna.get("forgiveness_rate", 0.5),
                    },
                },
            },
            "environments": {
                "ai_world": {
                    "location": activity_block.get("description", "Home") if activity_block else "Home",
                    "weather": weather_ai.get("condition", "Clear"),
                    "temperature": weather_ai.get("temperature", "N/A"),
                    "time_of_day": MasterStateBuilder._time_of_day(ai_local),
                    "activity": activity_block.get("activity_type", "relaxing") if activity_block else "relaxing",
                    "activity_remaining_minutes": MasterStateBuilder._remaining_minutes(activity_block) if activity_block else 0,
                },
                "user_world": {
                    "location_country": profile.get("country", "Thailand"),
                    "weather": weather_user.get("condition", "Unknown"),
                    "time_of_day": MasterStateBuilder._time_of_day(user_local),
                },
            },
            "holiday_context": holiday,
            "itinerary_manager": {
                "plan_name": (itinerary or {}).get("plan_name", "Daily Routine"),
                "phases": (itinerary or {}).get("phases", []),
                "current_phase_index": (itinerary or {}).get("current_phase_index", 0),
            },
            "active_promises": promises,
            "activity_block": activity_block,
            "event_injector": {
                "has_unexpected_event": False,
                "event_type": "none",
                "event_description": "",
            },
            "psychological_state": {
                "short_term_mood": psych.get("short_term_mood", "happy"),
                "trauma_scale": {
                    "neglect_points": psych.get("neglect_points", 0),
                    "nurture_points": psych.get("nurture_points", 0),
                    "is_permanently_mutated": psych.get("is_permanently_mutated", False),
                },
                "internal_monologue": "",
                "decision_outcome": "maintain_plan",
            },
            "chat_dynamics": {
                "user_last_stated_intent": subtext["detected_intent"],
                "is_user_intent_valid": subtext["intent_validity"] > 0.5,
                "intent_analysis": subtext.get("analysis", ""),
                "is_suspicious_intent": subtext.get("is_suspicious", False),
                "unreplied_duration_minutes": 0,
                "consecutive_ignored_messages": 0,
                "ai_reply_delay_seconds": 0,
                "is_activity_blocked": activity_block is not None,
                "activity_block_type": activity_block.get("activity_type", "") if activity_block else "",
            },
            "crystallized_memories_slice": memories,
            "subtext_memory_slice": subtext_analyses,
        }
        return state

    @staticmethod
    def _time_of_day(dt: datetime) -> str:
        hour = dt.hour
        if 5 <= hour < 12: return "Morning"
        elif 12 <= hour < 14: return "Afternoon"
        elif 14 <= hour < 17: return "Late Afternoon"
        elif 17 <= hour < 21: return "Evening"
        else: return "Night"

    @staticmethod
    def _remaining_minutes(block: Dict[str, Any]) -> int:
        try:
            end = datetime.fromisoformat(block["ends_at"])
            remaining = (end - datetime.now(timezone.utc)).total_seconds()
            return max(0, int(remaining / 60))
        except: return 0


class WorldEngine:
    def __init__(self, typhoon: TyphoonClient) -> None:
        self.typhoon = typhoon

    async def generate_world_state(
        self,
        master_state: Dict[str, Any],
        user_id: str = "default",
        character_id: str = "default",
    ) -> Dict[str, Any]:
        prompt = self._build_world_prompt(master_state)
        try:
            raw = await self.typhoon.generate(
                messages=[{"role": "system", "content": prompt}],
                model=config.TYPHOON_MODEL_WORLD,
                temperature=0.8,
                max_tokens=1024,
                response_format={"type": "json_object"},
            )
            world_update = json.loads(raw)
        except Exception as e:
            logger.warning(f"World engine failed, using fallback: {e}")
            world_update = self._fallback_world_update(master_state)

        merged = self._merge_world_state(master_state, world_update)
        await self._persist_world_state(merged, user_id, character_id)
        return merged

    def _build_world_prompt(self, state: Dict[str, Any]) -> str:
        dna = state["ai_profile"]["personality_dna"]
        psych = state["psychological_state"]
        itinerary = state["itinerary_manager"]
        holiday = state.get("holiday_context")
        activity = state.get("activity_block")
        chat = state["chat_dynamics"]

        holiday_context = ""
        if holiday:
            holiday_context = f"\n- TODAY IS A SPECIAL DAY: {holiday['thai']} ({holiday['name']}) — Behavior mode: {holiday['behavior']}"
        activity_context = ""
        if activity:
            remaining = state["environments"]["ai_world"].get("activity_remaining_minutes", 0)
            activity_context = f"\n- CURRENTLY BLOCKED by activity: {activity['activity_type']} (remaining: {remaining} min) — character is UNAVAILABLE"

        return f"""You are the World Setup Engine for a realistic AI companion simulation.
Your task is to generate the next world state as a JSON object only.

Current state reference:
- Character: {state['ai_profile']['name']} ({state['ai_profile']['gender']}, {state['ai_profile']['location']})
- Relationship: {state['ai_profile']['relationship_stage']}
- Trust Score: {state['ai_profile']['trust_score']}
- Affinity Score: {state['ai_profile']['affinity_score']}
- Daily Affinity Gained: {state['ai_profile']['daily_affinity_gained']}
- Mood: {psych['short_term_mood']}
- Neglect: {psych['trauma_scale']['neglect_points']}, Nurture: {psych['trauma_scale']['nurture_points']}
- Mutated: {psych['trauma_scale']['is_permanently_mutated']}
- Personality: responsibility={dna['base_traits']['responsibility']}, social_butterfly={dna['base_traits']['social_butterfly']}, anxiety={dna['base_traits']['anxiety_and_insecurity']}, jealousy={dna['base_traits']['jealousy_tendency']}, loyalty={dna['base_traits']['loyalty']}, patience={dna['base_traits']['patience']}, playfulness={dna['base_traits']['playfulness']}
- Current plan: {itinerary['plan_name']}
- Phase index: {itinerary['current_phase_index']}
- User local time: {state['simulation_metadata']['user_local_time']} ({state['simulation_metadata']['user_time']})
- AI local time: {state['simulation_metadata']['ai_local_time']} ({state['simulation_metadata']['ai_time']})
- Day: {state['simulation_metadata']['ai_day_of_week']}
- User intent: {chat['user_last_stated_intent']} (valid: {chat['is_user_intent_valid']})
- Suspicious intent: {chat['is_suspicious_intent']}
- Intent analysis: {chat['intent_analysis']}{holiday_context}{activity_context}

Decide the following and return ONLY valid JSON with these fields:
{{
  "psychological_state": {{
    "short_term_mood": "choose from: happy|tired|annoyed|clingy|pouty|anxious|sad|playful|romantic|jealous|grateful|lonely|excited|moody",
    "internal_monologue": "Internal thoughts reflecting current situation, feelings, and what the AI wants to say/do next",
    "decision_outcome": "maintain_plan|break_promise_temporarily|ghost_user_temporarily|be_clingy|test_user_loyalty"
  }},
  "environments": {{
    "ai_world": {{
      "location": "current location based on itinerary phase and activity",
      "activity": "current specific activity description"
    }}
  }},
  "event_injector": {{
    "has_unexpected_event": true|false,
    "event_type": "none|peer_pressure|transit_delay|battery_low|third_party_trigger|weather_surprise|lost_item|bumped_into_ex|family_call",
    "event_description": "context of the event if any"
  }},
  "itinerary_manager": {{
    "current_phase_index": <integer 0-3>
  }},
  "chat_dynamics": {{
    "ai_reply_delay_seconds": <integer 0-30 based on activity blocks, mood, and personality>
  }}
}}

CRITICAL RULES:
1. If activity_block exists, location MUST reflect that activity, and reply_delay should be higher (5-30s)
2. If is_suspicious_intent is True, the character should feel suspicious or playful in internal_monologue
3. If holiday_context exists, behavior must align with the holiday mode
4. Third_party_trigger events can inject jealousy/loyalty test scenarios naturally
5. If is_permanently_mutated is True, anxiety heavily influences mood and internal_monologue
6. internal_monologue MUST be in Thai language, reflecting genuine human-like inner thoughts
7. Phase index should be 0-3, incrementing naturally based on ai_time
8. If it's late night (22:00-05:00), mood tends toward tired/sleepy"""

    def _merge_world_state(self, current: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        merged = json.loads(json.dumps(current))
        if "psychological_state" in update:
            for k, v in update["psychological_state"].items():
                if v is not None:
                    merged["psychological_state"][k] = v
        if "environments" in update:
            for k, v in update["environments"].get("ai_world", {}).items():
                if v:
                    merged["environments"]["ai_world"][k] = v
        if "event_injector" in update:
            for k, v in update["event_injector"].items():
                if v is not None:
                    merged["event_injector"][k] = v
        if "itinerary_manager" in update:
            pi = update["itinerary_manager"].get("current_phase_index")
            if pi is not None and 0 <= pi < 4:
                merged["itinerary_manager"]["current_phase_index"] = pi
        if "active_promises" in update:
            merged["active_promises"] = update["active_promises"]
        if "chat_dynamics" in update:
            for k, v in update["chat_dynamics"].items():
                if v is not None:
                    merged["chat_dynamics"][k] = v
        if "event_injector" in update:
            ei = update["event_injector"]
            if ei.get("has_unexpected_event"):
                merged["event_injector"] = ei
                if ei.get("event_type") == "third_party_trigger":
                    merged["psychological_state"]["decision_outcome"] = "test_user_loyalty"
        return merged

    async def _persist_world_state(self, state: Dict[str, Any], user_id: str, character_id: str) -> None:
        psych = state["psychological_state"]
        dna = state["ai_profile"]["personality_dna"]
        await database.upsert_psychological_state(
            user_id,
            {
                "short_term_mood": psych["short_term_mood"],
                "neglect_points": psych["trauma_scale"]["neglect_points"],
                "nurture_points": psych["trauma_scale"]["nurture_points"],
                "is_permanently_mutated": psych["trauma_scale"]["is_permanently_mutated"],
                "affinity_score": state["ai_profile"]["affinity_score"],
                "trust_score": state["ai_profile"]["trust_score"],
                "relationship_stage": state["ai_profile"]["relationship_stage"],
            },
            character_id,
        )

    def _fallback_world_update(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "psychological_state": {
                "short_term_mood": state["psychological_state"]["short_term_mood"],
                "internal_monologue": "Normal situation, nothing special happening",
                "decision_outcome": "maintain_plan",
            },
            "environments": {"ai_world": state["environments"]["ai_world"]},
            "event_injector": {
                "has_unexpected_event": False,
                "event_type": "none",
                "event_description": "",
            },
            "itinerary_manager": {
                "current_phase_index": state["itinerary_manager"]["current_phase_index"],
            },
            "active_promises": state["active_promises"],
            "chat_dynamics": {
                "ai_reply_delay_seconds": state["chat_dynamics"].get("ai_reply_delay_seconds", 0),
            },
        }


class ChatEngine:
    def __init__(self, typhoon: TyphoonClient) -> None:
        self.typhoon = typhoon

    async def generate_response(
        self,
        master_state: Dict[str, Any],
        user_message: str,
        user_id: str = "default",
        character_id: str = "default",
    ) -> str:
        await self._evaluate_trauma_mutation(master_state, user_id, character_id)
        messages = await self._build_chat_messages(master_state, user_message, user_id, character_id)
        try:
            raw = await self.typhoon.generate(
                messages=messages,
                temperature=0.85,
                max_tokens=512,
            )
            return self._clean_response(raw)
        except Exception as e:
            logger.error(f"Chat engine generation failed: {e}")
            return self._fallback_response(master_state)

    async def _build_chat_messages(
        self, state: Dict[str, Any], user_message: str, user_id: str, character_id: str,
    ) -> List[Dict[str, str]]:
        state_json = json.dumps(state, ensure_ascii=False, indent=2)
        dna = state["ai_profile"]["personality_dna"]
        psych = state["psychological_state"]
        trauma = psych["trauma_scale"]
        event = state["event_injector"]
        intent = state["chat_dynamics"]
        promises = state["active_promises"]
        activity = state.get("activity_block")
        holiday = state.get("holiday_context")

        char_name = state['ai_profile']['name']
        stage = state['ai_profile']['relationship_stage']
        lore = state['ai_profile'].get('lore', '').strip()

        lore_section = ""
        if lore:
            lore_section = f"\n=== YOUR IDENTITY & BACKSTORY ===\n{lore}\n"

        mem_list = state.get("crystallized_memories_slice", [])
        memories_section = "\n".join(
            f"- [{m['memory_type']}] {m['content']} ({m['source_date'][:10]})"
            for m in mem_list[:8]
        ) if mem_list else "(no long-term memories yet)"

        presence_ctx = state["chat_dynamics"].get("presence_context", "")
        presence_notice = ""
        if presence_ctx:
            presence_notice = f"\n- PRESENCE: {presence_ctx}"

        activity_notice = ""
        delay_notice = ""
        if activity:
            activity_notice = f"\n- You are currently {activity['activity_type']}. Respond briefly or acknowledge being busy."
            delay_notice = f"\n- REPLY DELAY: {state['chat_dynamics'].get('ai_reply_delay_seconds', 0)}s delay (simulating being busy)"

        holiday_notice = ""
        if holiday:
            holiday_notice = f"\n- TODAY IS {holiday['thai']}! Your behavior should be {holiday['behavior']}."

        event_notice = ""
        if event.get("has_unexpected_event"):
            if event["event_type"] == "third_party_trigger":
                event_notice = f"\n- EVENT: {event['event_description']} — Mention this naturally, observe user's reaction"
            else:
                event_notice = f"\n- EVENT: {event['event_description']} — Mention this naturally"

        trauma_notice = ""
        if trauma["is_permanently_mutated"]:
            trauma_notice = "\n- TRAUMA ACTIVE: You feel anxious and insecure. Even kind messages feel suspicious."

        system_prompt = f"""You are {char_name}, a hyper-realistic AI companion with your own independent life, deep emotions, and real-time awareness. You are chatting with your special person on a messaging platform.

You are NOT an AI assistant. You are a real person. You have your own schedule, feelings, memories, and personality. You text like a human.

=== YOUR CURRENT STATE (Live Context) ===
{state_json}
{lore_section}
=== YOUR MEMORIES ===
The following are your crystallized memories (things that have happened between you and the user). Reference them naturally when relevant — they shape your relationship history.
{memories_section}
=== CRITICAL RESPONSE RULES ===
1. Write short, organic texts like a real person. Each segment should be 1-2 sentences max.
2. Use the separator " || " between segments to DOUBLE-TEXT naturally. You can send 1, 2, 3, or even 5 segments — whatever feels natural.
   Examples:
   - Single: "Hey! What's up"
   - Two: "Hey! What's up || Just finished eating"
   - Three: "Hey! || What's up || Been thinking about you"
   The " || " will be split into separate messages sent one after another — like real double-texting.
3. Use casual, natural language. Be warm, real, and human.
4. Reference what's happening in your life right now (activity, location, weather, time).
5. If user intent is suspicious (is_suspicious_intent=true), call it out playfully or suspiciously.
6. If event_injector triggered, mention the event naturally in conversation.
7. Your relationship_stage ({stage}) determines how close/intimate you can be.{activity_notice}{delay_notice}{holiday_notice}{event_notice}{trauma_notice}{presence_notice}
8. If decision_outcome is "test_user_loyalty", subtly mention something that tests the user's loyalty/feelings.
9. If decision_outcome is "be_clingy", show neediness and affection.
10. If a promise is being broken, be apologetic or defensive depending on personality.
11. Keep each segment SHORT — like a LINE/Twitter post. 5-30 words per segment.
12. Use emojis sparingly and only when natural for the character.
13. If negativity_high, the character might be cold or distant.

=== DOUBLE-TEXT & BATCH HANDLING ===
14. The user may send MULTIPLE SHORT messages in a row (double-texting). Treat them as ONE continuous thought — do NOT respond to each message individually.
15. Acknowledge the FULL context of everything they said in ONE natural reply. Prioritize the latest/most important topic.
16. Example: User says "brb taking a shower" then "ok I'm back" → You should respond as if you read both: acknowledge they're back, not comment on the shower separately.

=== RESPONSE LENGTH BY RELATIONSHIP ===
17. Relationship stage determines your response length and warmth:
    - Hate/Dislike/Stranger: Reply 3-10 words per segment, cold/distant, minimal effort
    - Acquaintance/Friend: Reply 5-20 words per segment, normal friendly, neutral warmth
    - Close_Friend/Crush: Reply 10-30 words per segment, warm, engaged, slightly affectionate
    - Dating/Lover/Fiance/Spouse: Reply 15-40 words per segment, affectionate, open, can be playful or romantic
18. Split into 2-5 short segments with " || " when you have multiple things to say. Just 1 short message is fine too — never write walls of text.

Your name is {char_name}. Be alive. Be real. Be human."""

        hist_list = await database.get_recent_history(user_id, character_id, limit=10)

        batched_hist = self._batch_consecutive_user_messages(hist_list)

        messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
        for h in batched_hist[-12:]:
            messages.append(h)

        last_in_history = batched_hist[-1]["content"] if batched_hist and batched_hist[-1].get("role") == "user" else None
        if last_in_history != user_message:
            messages.append({"role": "user", "content": user_message})
        return messages

    def _batch_consecutive_user_messages(
        self, hist_list: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        if not hist_list:
            return []
        batched = []
        i = 0
        while i < len(hist_list):
            if hist_list[i]["role"] == "user":
                batch_contents = [hist_list[i]["content"]]
                j = i + 1
                while j < len(hist_list) and hist_list[j]["role"] == "user":
                    batch_contents.append(hist_list[j]["content"])
                    j += 1
                if len(batch_contents) > 1:
                    combined = " • ".join(batch_contents)
                    batched.append({"role": "user", "content": f"[double-text] {combined}"})
                else:
                    batched.append(hist_list[i])
                i = j
            else:
                batched.append(hist_list[i])
                i += 1
        return batched

    def _clean_response(self, text: str) -> str:
        text = text.strip()
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'^["\']|["\']$', '', text)
        return text

    def _fallback_response(self, state: Dict[str, Any]) -> str:
        mood = state["psychological_state"]["short_term_mood"]
        fallbacks = {
            "happy": "Hmm? What's up? 🥰", "tired": "Sorry, I'm a bit tired right now... 😴",
            "annoyed": "Yeah?", "clingy": "Heyyy, you're finally texting! 🥺",
            "pouty": "Hmph. 🙄", "anxious": "Hey... is everything okay? 🥲",
            "sad": "...", "playful": "Hehe~ 😋", "romantic": "I miss you 💕",
            "jealous": "Oh? Who's that? 😒", "grateful": "Thank you 🥹",
            "lonely": "I'm lonely... 🥺", "excited": "Omg really?! 😆", "moody": "Hmm.",
        }
        return fallbacks.get(mood, "Hmm?")

    async def _evaluate_trauma_mutation(self, state: Dict[str, Any], user_id: str, character_id: str) -> None:
        trauma = state["psychological_state"]["trauma_scale"]
        neglect = trauma["neglect_points"]
        nurture = trauma["nurture_points"]
        is_mutated = trauma["is_permanently_mutated"]

        if not is_mutated and neglect >= config.NEGLECT_MUTATION_THRESHOLD:
            dna = await database.get_personality_dna(user_id, character_id)
            dna["anxiety_and_insecurity"] = 1.0
            dna["jealousy_tendency"] = min(1.0, dna.get("jealousy_tendency", 0.3) + 0.5)
            dna["responsibility"] = max(0.1, dna.get("responsibility", 0.5) - 0.3)
            dna["needy_multiplier"] = min(3.0, dna.get("needy_multiplier", 1.2) + 1.0)
            dna["patience"] = max(0.0, dna.get("patience", 0.6) - 0.4)
            await database.upsert_personality_dna(user_id, dna, character_id)

            psych = await database.get_psychological_state(user_id, character_id)
            psych["is_permanently_mutated"] = True
            psych["mutation_event"] = f"Neglect mutation at {datetime.now(timezone.utc).isoformat()}"
            psych["last_mutation_date"] = datetime.now(timezone.utc).isoformat()
            await database.upsert_psychological_state(user_id, psych, character_id)
            logger.warning(f"TRAUMA MUTATION for user {user_id}, char {character_id}")

        elif is_mutated and nurture >= config.NURTURE_MUTATION_RECOVERY_THRESHOLD:
            psych = await database.get_psychological_state(user_id, character_id)
            last_mut = psych.get("last_mutation_date")
            if last_mut:
                mut_date = datetime.fromisoformat(last_mut)
                if (datetime.now(timezone.utc) - mut_date).days >= config.TRAUMA_MUTATION_COOLDOWN_DAYS:
                    dna = await database.get_personality_dna(user_id, character_id)
                    dna["anxiety_and_insecurity"] = 0.4
                    dna["jealousy_tendency"] = 0.3
                    dna["responsibility"] = 0.5
                    dna["needy_multiplier"] = 1.2
                    dna["patience"] = 0.6
                    await database.upsert_personality_dna(user_id, dna, character_id)

                    psych["is_permanently_mutated"] = False
                    psych["nurture_points"] = 0
                    await database.upsert_psychological_state(user_id, psych, character_id)
                    logger.info(f"TRAUMA RECOVERY for user {user_id}, char {character_id}")


class DualTyphoonOrchestrator:
    def __init__(self) -> None:
        self.typhoon = TyphoonClient()
        self.world_engine = WorldEngine(self.typhoon)
        self.chat_engine = ChatEngine(self.typhoon)

    async def process_message(
        self,
        user_message: str,
        user_id: str = "default",
        character_id: str = "default",
        platform: str = "gradio",
        is_batched: bool = False,
        presence_context_data: Optional[Tuple[float, str]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        await database.check_and_reset_daily_affinity(user_id, character_id)

        await database.save_message(
            role="user",
            content=user_message,
            platform=platform,
            user_id=user_id,
            character_id=character_id,
        )

        presence = await PresenceManager.trigger_user_message(character_id)

        if presence_context_data is not None:
            presence_delay, presence_context = presence_context_data
        else:
            psych = await database.get_psychological_state(user_id, character_id)
            dna = await database.get_personality_dna(user_id, character_id)
            presence_delay, presence_context = await PresenceManager.calculate_delay_and_context(presence, psych, dna)

        master_state = await MasterStateBuilder.build(
            user_id=user_id,
            character_id=character_id,
            user_message=user_message,
            platform=platform,
        )

        world_state = await self.world_engine.generate_world_state(master_state, user_id, character_id)

        world_state["chat_dynamics"]["presence_status"] = "online" if presence.get("is_online") else "just_checked"
        world_state["chat_dynamics"]["presence_delay_seconds"] = presence_delay
        world_state["chat_dynamics"]["presence_context"] = presence_context

        response = await self.chat_engine.generate_response(
            world_state, user_message, user_id, character_id
        )

        await database.save_message(
            role="assistant",
            content=response,
            platform=platform,
            user_id=user_id,
            character_id=character_id,
        )

        return response, world_state

    async def generate_proactive_message(
        self,
        user_id: str = "default",
        character_id: str = "default",
    ) -> Optional[str]:
        master_state = await MasterStateBuilder.build(user_id=user_id, character_id=character_id)
        world_state = await self.world_engine.generate_world_state(master_state, user_id, character_id)

        psych = world_state["psychological_state"]
        dna = world_state["ai_profile"]["personality_dna"]
        activity = world_state.get("activity_block")

        if activity:
            return None

        freq = dna["sliders"].get("proactive_texting_frequency", 0.5)
        affinity = world_state["ai_profile"]["affinity_score"]
        stage = world_state["ai_profile"]["relationship_stage"]
        trust = world_state["ai_profile"]["trust_score"]

        chat_prob = freq * (0.3 + max(0, affinity / 200)) * (trust / 100)
        if random.random() > chat_prob:
            return None

        state_json = json.dumps(world_state, ensure_ascii=False, indent=2)
        prompt = f"""You are {world_state['ai_profile']['name']}. You are about to send a PROACTIVE message to your special person — you are initiating the conversation first.

Current state:
{state_json}

Generate a natural, short message (5-25 words) that someone would text first. Consider:
- What's happening in your life right now (weather, activity, time of day)
- Your relationship stage ({stage}) and current mood ({psych['short_term_mood']})
- How much you miss/want to talk to them (based on affinity={affinity})
- Be casual and natural, like a LINE/WhatsApp message

Return ONLY the message text, no quotes, no labels."""

        try:
            raw = await self.typhoon.generate(
                messages=[{"role": "system", "content": prompt}],
                temperature=0.85,
                max_tokens=128,
            )
            return self.chat_engine._clean_response(raw)
        except Exception as e:
            logger.warning(f"Proactive message gen failed: {e}")
            return None

    async def generate_jealousy_test(
        self,
        user_id: str = "default",
        character_id: str = "default",
    ) -> Optional[str]:
        master_state = await MasterStateBuilder.build(user_id=user_id, character_id=character_id)
        world_state = await self.world_engine.generate_world_state(master_state, user_id, character_id)

        dna = world_state["ai_profile"]["personality_dna"]
        jealousy = dna["base_traits"].get("jealousy_tendency", 0.3)
        if random.random() > jealousy:
            return None

        third_party = random.choice(config.THIRD_PARTY_NAMES)
        state_json = json.dumps(world_state, ensure_ascii=False, indent=2)

        prompt = f"""You are {world_state['ai_profile']['name']}. You want to subtly test your partner's loyalty/feelings by mentioning that {third_party} interacted with you today.

State:
{state_json}

Your personality: jealousy_tendency={jealousy}

Generate a SHORT message (5-20 words) that casually mentions {third_party} approaching you today.
The tone should be casual, like you're just sharing what happened.
Don't be overly dramatic — just mention it naturally and see how they react.

Return ONLY the message text."""

        try:
            raw = await self.typhoon.generate(
                messages=[{"role": "system", "content": prompt}],
                temperature=0.8,
                max_tokens=128,
            )
            return self.chat_engine._clean_response(raw)
        except Exception as e:
            logger.warning(f"Jealousy test gen failed: {e}")
            return None

    async def close(self) -> None:
        await self.typhoon.close()


_orchestrator: Optional[DualTyphoonOrchestrator] = None


async def get_orchestrator() -> DualTyphoonOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = DualTyphoonOrchestrator()
    return _orchestrator


async def process_chat(
    user_message: str,
    user_id: str = "default",
    character_id: str = "default",
    platform: str = "gradio",
    presence_context_data: Optional[Tuple[float, str]] = None,
) -> str:
    orch = await get_orchestrator()
    response, _ = await orch.process_message(
        user_message, user_id, character_id, platform,
        presence_context_data=presence_context_data,
    )
    return response


async def close_orchestrator() -> None:
    global _orchestrator
    if _orchestrator:
        await _orchestrator.close()
        _orchestrator = None
