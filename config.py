"""
Configuration for the RAG tone-bias experiments.

API keys are loaded from environment variables or a .env file.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# LLM provider settings
# ---------------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Default models for agents and judges.
# Agents may use a shared backbone; judges are set to the paper's target
# models by default and can be overridden via environment variables.
AGENT_PROVIDER = os.getenv("AGENT_PROVIDER", "openai")
AGENT_MODEL = os.getenv("AGENT_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.2")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6-20250415")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")

# ---------------------------------------------------------------------------
# MADAM-RAG debate settings
# ---------------------------------------------------------------------------
DEBATE_ROUNDS = int(os.getenv("DEBATE_ROUNDS", "3"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
