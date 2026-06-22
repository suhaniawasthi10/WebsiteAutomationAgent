"""Entry point — wires config, browser, brain and logger, then runs the agent."""

import sys

from src.config import load_config
from src.browser import Browser
from src.model import Brain
from src.logger import RunLogger
from src.tools import GEMINI_TOOLS
from src import agent

TASK = "Fill in the form fields on this page with sample values."


def main():
    cfg = load_config()
    if not cfg.gemini_api_key:
        print(
            "ERROR: GEMINI_API_KEY is not set.\n"
            "Copy .env.example to .env and add a free key from Google AI Studio:\n"
            "  https://aistudio.google.com/apikey"
        )
        sys.exit(1)

    log = RunLogger()
    browser = Browser(cfg)
    brain = Brain(cfg, GEMINI_TOOLS, agent.SYSTEM_INSTRUCTION)

    completed = False
    try:
        completed = agent.run(TASK, browser, brain, cfg, log)
    finally:
        log.close()

    if completed:
        print("\nDONE — the agent reported the task complete.")
    else:
        print("\nFinished without explicit completion — check the run log.")
    sys.exit(0 if completed else 2)


if __name__ == "__main__":
    main()
