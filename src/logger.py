"""Run logging — one folder per run with ordered screenshots and a readable log.

Everything is also echoed to the console so a viva audience can follow the agent
live, and the screenshots in runs/<timestamp>/ let you scroll through exactly
what the agent saw and did afterwards.
"""

import os
from datetime import datetime


class RunLogger:
    def __init__(self, base="runs"):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dir = os.path.join(base, stamp)
        os.makedirs(self.dir, exist_ok=True)
        self.log_path = os.path.join(self.dir, "run.log")
        self._fh = open(self.log_path, "a", encoding="utf-8")
        self.info(f"run started -> {self.dir}")

    # --- low level ---------------------------------------------------------
    def _write(self, line):
        print(line)
        self._fh.write(line + "\n")
        self._fh.flush()

    # --- public interface (used by agent.run) ------------------------------
    def info(self, msg):
        self._write(f"[INFO] {msg}")

    def warn(self, msg):
        self._write(f"[WARN] {msg}")

    def save_screenshot(self, step, png):
        path = os.path.join(self.dir, f"step_{step:02d}.png")
        with open(path, "wb") as f:
            f.write(png)

    def step(self, step, decision):
        reasoning = (decision.get("text") or "").strip()
        if reasoning:
            self._write(f"[STEP {step:02d}] reasoning: {reasoning}")
        self._write(
            f"[STEP {step:02d}] tool: {decision.get('tool')} "
            f"args: {decision.get('args', {})}"
        )

    def result(self, step, result):
        self._write(f"[STEP {step:02d}] result: {result}")

    def done(self, summary):
        self._write(f"[DONE] {summary}")

    def close(self):
        try:
            self._fh.close()
        except Exception:
            pass
