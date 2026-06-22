"""The brain — a thin wrapper around Gemini, kept behind a small interface.

The rest of the code only knows `Brain.act(contents) -> {tool, args, text, content}`,
so the model could be swapped for another provider without touching the loop.
Rate-limit handling (the free tier is ~10 req/min) lives here as retry-with-
exponential-backoff on 429 / RESOURCE_EXHAUSTED.
"""

import time

from google import genai
from google.genai import types

_RETRYABLE = ("RESOURCE_EXHAUSTED", "429", "UNAVAILABLE", "503", "500")


class Brain:
    def __init__(self, cfg, tools, system):
        self.client = genai.Client()  # reads GEMINI_API_KEY from env
        self.model = cfg.model
        self.config = types.GenerateContentConfig(
            system_instruction=system,
            tools=[tools],
            # we handle the function calls ourselves, one per turn
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
            temperature=0.2,  # steadier coordinate picking
        )

    def act(self, contents):
        last_err = None
        for attempt in range(5):
            try:
                resp = self.client.models.generate_content(
                    model=self.model, contents=contents, config=self.config
                )
                return self._parse(resp)
            except Exception as err:  # noqa: BLE001 - classify then re-raise
                last_err = err
                if any(tok in str(err) for tok in _RETRYABLE):
                    time.sleep(2 ** attempt)  # backoff for the free-tier limit
                    continue
                raise
        raise RuntimeError(f"Gemini call failed after retries: {last_err}")

    def _parse(self, resp):
        if not resp.candidates:
            return self._text_decision("(model returned no candidates)")
        content = resp.candidates[0].content
        parts = (content.parts or []) if content else []
        text_bits = []
        for part in parts:
            if getattr(part, "function_call", None):
                fc = part.function_call
                return {
                    "tool": fc.name,
                    "args": dict(fc.args or {}),
                    "text": " ".join(text_bits).strip(),
                    "content": content,
                }
            if getattr(part, "text", None):
                text_bits.append(part.text)
        # no function call -> model just talked; treat its text as reasoning.
        # Only reuse the real content if it actually has parts; an empty/parts-
        # less candidate would otherwise break history trimming downstream.
        decision = self._text_decision(" ".join(text_bits).strip())
        if content is not None and content.parts:
            decision["content"] = content
        return decision

    @staticmethod
    def _text_decision(text):
        return {
            "tool": None,
            "args": {},
            "text": text,
            "content": types.Content(
                role="model", parts=[types.Part.from_text(text=text or "(empty)")]
            ),
        }
