"""Playwright browser wrapper — the concrete implementations of every required tool.

Coordinate mapping is the detail that makes or breaks this (see context.md):
the model returns (x, y) in image-pixel space, and page.mouse.click expects CSS
pixels in the viewport. They line up 1:1 only because we:
  - launch the context with device_scale_factor=1,
  - use a fixed viewport,
  - take viewport-only screenshots (never full_page),
  - never resize the image before sending it to the model.
So whatever pixel the model points at is exactly the pixel the mouse hits.
"""

import time

from playwright.sync_api import sync_playwright
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError


class Browser:
    def __init__(self, cfg):
        self.cfg = cfg
        self._pw = None
        self.browser = None
        self.context = None
        self.page = None

    # --- lifecycle ---------------------------------------------------------
    def open_browser(self):
        self._pw = sync_playwright().start()
        self.browser = self._pw.chromium.launch(headless=self.cfg.headless)
        self.context = self.browser.new_context(
            viewport={
                "width": self.cfg.viewport_width,
                "height": self.cfg.viewport_height,
            },
            device_scale_factor=1,  # 1:1 image px == css px
        )
        self.page = self.context.new_page()
        return "browser opened"

    def close_browser(self):
        try:
            if self.context is not None:
                self.context.close()
            if self.browser is not None:
                self.browser.close()
        finally:
            if self._pw is not None:
                self._pw.stop()
                self._pw = None
        return "browser closed"

    # --- navigation --------------------------------------------------------
    def navigate_to_url(self, url):
        attempts = 3  # initial try + retry twice
        last_err = None
        for attempt in range(attempts):
            try:
                self.page.goto(url, wait_until="domcontentloaded", timeout=30000)
                return f"navigated to {url}"
            except PlaywrightTimeoutError as err:
                last_err = err
                # fail loudly only after exhausting retries; back off in between
                if attempt < attempts - 1:
                    time.sleep(2 ** attempt)
        raise RuntimeError(
            f"navigation to {url} failed after {attempts} attempts: {last_err}"
        )

    # --- perception --------------------------------------------------------
    def take_screenshot(self):
        # viewport only (NOT full_page) so coords match what the mouse can reach
        return self.page.screenshot()

    # --- actions -----------------------------------------------------------
    def click_on_screen(self, x, y):
        self._validate_coords(x, y)
        self.page.mouse.click(x, y)
        return f"clicked at ({x}, {y})"

    def double_click(self, x, y):
        self._validate_coords(x, y)
        self.page.mouse.dblclick(x, y)
        return f"double-clicked at ({x}, {y})"

    def send_keys(self, text):
        # types into whatever is focused, so the flow is: click the field, then type
        self.page.keyboard.type(text, delay=30)
        return f"typed {text!r}"

    def scroll(self, direction, amount):
        dy = amount if direction == "down" else -amount
        self.page.mouse.wheel(0, dy)
        return f"scrolled {direction} by {amount}"

    def focused_value(self):
        """Return the focused field's current text, or None if nothing typable
        is focused. Used to verify that send_keys actually landed: a misclick
        leaves nothing focused, so the keystrokes go nowhere.
        """
        return self.page.evaluate(
            """() => {
                const el = document.activeElement;
                if (!el) return null;
                const tag = (el.tagName || '').toLowerCase();
                if (tag === 'input' || tag === 'textarea') return el.value;
                if (el.isContentEditable) return el.innerText;
                return null;
            }"""
        )

    # --- helpers -----------------------------------------------------------
    def _validate_coords(self, x, y):
        w = self.cfg.viewport_width
        h = self.cfg.viewport_height
        if not (0 <= x < w and 0 <= y < h):
            raise ValueError(
                f"coordinates ({x}, {y}) are outside the {w}x{h} viewport"
            )
