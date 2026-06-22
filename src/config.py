"""Config — load .env once and expose typed settings. No secrets in code."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

DEFAULT_TARGET = "https://ui.shadcn.com/docs/forms/react-hook-form"


@dataclass
class Config:
    gemini_api_key: str
    model: str
    headless: bool
    viewport_width: int
    viewport_height: int
    max_steps: int
    step_delay: float
    target_url: str


def _as_bool(value, default=False):
    if value is None:
        return default
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def load_config():
    load_dotenv()  # makes GEMINI_API_KEY available to genai.Client() too
    return Config(
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        model=os.getenv("MODEL", "gemini-2.5-flash"),
        headless=_as_bool(os.getenv("HEADLESS"), False),
        viewport_width=int(os.getenv("VIEWPORT_WIDTH", "1280")),
        viewport_height=int(os.getenv("VIEWPORT_HEIGHT", "800")),
        max_steps=int(os.getenv("MAX_STEPS", "15")),
        step_delay=float(os.getenv("STEP_DELAY", "2")),
        target_url=os.getenv("TARGET_URL", DEFAULT_TARGET),
    )
