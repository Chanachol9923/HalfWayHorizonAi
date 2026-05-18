# HalfWay Horizon AI

Hyper-realistic time-aware AI Companion powered by **Dual-Typhoon Instruct Engine**, running 24/7 on Telegram with background lifestyle simulation, memory consolidation, and dynamic personality mutation.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Telegram Bot (PTB)                    │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│              Process Chat / World Engine                  │
│  ┌────────────┐  ┌──────────┐  ┌──────────────────────┐ │
│  │ MasterState│  │ WorldEng │  │      ChatEng         │ │
│  │   Builder  │  │ (lore +  │  │ (response + memory)  │ │
│  │            │  │  events) │  │                      │ │
│  └────────────┘  └──────────┘  └──────────────────────┘ │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│              Background Workers (asyncio)                │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────┐ │
│  │Lifestyle │ │  Memory  │ │Proactive │ │  Presence  │ │
│  │Simulator │ │Consolidat│ │  Text    │ │ Simulator  │ │
│  ├──────────┤ ├──────────┤ ├──────────┤ ├────────────┤ │
│  │ Promise  │ │ Affinity │ │ Ghosting │ │ Jealousy   │ │
│  │ Conflict │ │  Decay   │ │ Detector │ │   Test     │ │
│  └──────────┘ └──────────┘ └──────────┘ └────────────┘ │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                SQLite Database (aiosqlite)               │
│  Characters · Personalities · Psychology · Chat History │
│  Memories · Activity Blocks · Master State              │
└─────────────────────────────────────────────────────────┘
```

## Features

### 🧠 Dual-Typhoon Instruct Engine
- **World Engine**: Generates time-aware lore, events, and environmental context
- **Chat Engine**: Produces staggered double-text responses (` || ` separator)
- Typhoon model: `typhoon-v2.5-30b-a3b-instruct`

### 🎭 Dynamic Personality DNA
- 8 base traits (responsibility, social, anxiety, jealousy, loyalty, patience, playfulness, communication)
- Relationship stages: Stranger → Friend → Crush → Dating → Lover
- Permanent personality mutation triggered by trauma events
- Affinity/Trust scoring system

### 🕐 Time-Aware Presence
- Online/Offline/Away/Busy states with configurable delays
- Per-character presence tracking
- Response delays based on presence + personality

### 💬 Telegram Commands
| Command | Description |
|---------|-------------|
| `/start` | Welcome message + help |
| `/info` | View character info & stats |
| `/mood happy\|sad\|angry\|anxious` | Change mood |
| `/stage Stranger\|Friend\|Crush\|Dating\|Lover` | Change relationship stage |
| `/name <new name>` | Change character name |
| `/country <country>` | Set character country |
| `/city <city>` | Set character city |
| `/lore` | View current lore |
| `/lore_set <text>` | Set new lore |

### 🔄 Background Workers
- **Lifestyle Simulator**: Daily routines, events, activity blocks
- **Memory Consolidation**: Crystallizes short-term → long-term memory daily at 3:00
- **Proactive Text**: AI-initiated messages (minimum 3h interval)
- **Presence Simulator**: Auto-switches online/offline states
- **Affinity Decay**: Gradual relationship decay when neglected
- **Ghosting Detector**: Tracks neglect/nurture points
- **Jealousy Test Scheduler**: Periodic relationship tests
- **Promise Conflict Checker**: Detects scheduling conflicts
- **Keep-Alive Pinger**: Self-ping every 15 min

### 🚀 Rapid-Fire Protection
- 3-second window tracker
- Escalating delay (3s, 6s, 9s, 12s)
- Auto-skip at 6+ messages in window

## Quick Start

### 1. Prerequisites
- Python 3.10+
- Telegram Bot Token (from [@BotFather](https://t.me/BotFather))
- Typhoon API Key ([api.opentyphoon.ai](https://api.opentyphoon.ai))

### 2. Install
```bash
git clone https://github.com/Chanachol9923/HalfWayHorizonAi
cd HalfWayHorizonAi
pip install -r requirements.txt
```

### 3. Configure
```bash
cat > .env << 'EOF'
TYPHOON_API_KEY=sk-your-key
TELEGRAM_BOT_TOKEN=your-bot-token
HEADLESS_MODE=true
EOF
```

### 4. Run
```bash
python app.py
```

## Deploy to GCP e2-micro

### Swap (1GB RAM → 2GB)
```bash
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### Systemd Service
```ini
[Unit]
Description=HalfWay Horizon AI Companion
After=network.target

[Service]
Type=simple
User=YOUR_USER
WorkingDirectory=/home/YOUR_USER/horizon
EnvironmentFile=/home/YOUR_USER/horizon/.env
ExecStart=/home/YOUR_USER/.venv/bin/python app.py
Restart=always
RestartSec=10
MemoryMax=800M

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now horizon.service
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `TYPHOON_API_KEY` | — | Typhoon API key |
| `TELEGRAM_BOT_TOKEN` | — | Telegram bot token |
| `TELEGRAM_API_PROXY` | — | Optional proxy for Telegram API |
| `DISCORD_BOT_TOKEN` | — | Discord bot token |
| `HEADLESS_MODE` | `true` | Skip Gradio UI (saves RAM) |
| `CHARACTER_NAME` | `Ellie` | Default character name |
| `CHARACTER_GENDER` | `female` | Default character gender |
| `AI_TIMEZONE` | `Asia/Bangkok` | Character timezone |
| `AI_COUNTRY` | `Thailand` | Character country |
| `AI_CITY` | `Bangkok` | Character city |
| `TYPHOON_MODEL` | `typhoon-v2.5-30b-a3b-instruct` | Model ID |
| `GRADIO_PORT` | `7860` | Web UI port (when not headless) |
| `GRADIO_SHARE` | `false` | Create public Gradio link |

## Gradio Web UI

When `HEADLESS_MODE=false`, a full management UI is available:

- **Chat** — Web chat with staggered double-text
- **Persona Settings** — Edit all personality DNA sliders + lore + stage
- **Characters** — Create/delete characters
- **Live State** — Raw Master State JSON viewer
- **User Profile** — Edit user info (birthday, country, timezone)
- **System** — Config info + database backup

## Database

- Local SQLite via `aiosqlite`
- Tables: character_profiles, personality_dna, psychological_state, chat_history, memory, activity_blocks, user_profiles, app_state, master_state, itineraries
- Auto-backup every 6 hours to `data/backups/`

## Project Structure

```
HalfWayHorizonAi/
├── app.py          # Main entry: Gradio UI + Telegram bot + lifecycle
├── config.py       # All env vars, constants, defaults
├── database.py     # Async SQLite schema + CRUD
├── brain.py        # TyphoonClient, WorldEngine, ChatEngine, PresenceManager
├── worker.py       # All background workers + TextSplitter
├── Dockerfile      # Multi-stage Docker build (HF Spaces)
├── requirements.txt
└── .env            # Secrets (not committed)
```
