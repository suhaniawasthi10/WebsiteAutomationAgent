"""The perceive-act loop (ReAct style).

Each turn: screenshot -> annotate with set-of-marks -> send to the brain with the
running history -> get back one tool call -> execute it -> feed the result back.
Repeat until the model calls task_complete or we hit MAX_STEPS (the infinite-loop
guard). Only the latest screenshot is kept in context to stay under the per-minute
token cap.
"""

import time

from google.genai import types

from .annotate import get_interactive_elements, draw_marks, format_elements
from .tools import build_dispatch

SYSTEM_INSTRUCTION = (
    "You are a web automation agent controlling a real browser, one tool call "
    "per turn. Your goal: fill the form's text fields with short sample values, "
    "then finish. You do NOT need to submit the form.\n"
    "Each turn you receive a screenshot with numbered red boxes over the "
    "interactive elements, plus a list giving each element's id, tag, label, and "
    "the EXACT coordinates to click.\n"
    "How to fill ONE field (repeat for each field):\n"
    "1. Make sure the field is in the element list. The description textarea is "
    "usually below the visible area, so if it is not listed, scroll down to "
    "reveal it first.\n"
    "2. Click the field using the exact coordinates given for it in the list.\n"
    "3. On the NEXT turn, call send_keys to type a short sample value.\n"
    "Critical rules:\n"
    "- NEVER call send_keys twice in a row. send_keys types into the field you "
    "most recently clicked, so to fill a second field you MUST click that field "
    "first. Typing without clicking the new field just appends to the old one.\n"
    "- Use the exact coordinates from the list; do not guess or invent any.\n"
    "- Fill ONLY fields that appear in the list (a text input and a textarea). "
    "Do not invent fields such as username or email that are not present.\n"
    "- Do exactly one action per turn, then wait for the next screenshot.\n"
    "- Once BOTH the text input and the textarea contain a value, call "
    "task_complete with a one-line summary. Do not submit and do not keep "
    "clicking once both fields are filled."
)

NO_PROGRESS_LIMIT = 3  # consecutive unchanged screenshots before we give up


def run(task, browser, brain, cfg, log):
    dispatch = build_dispatch(browser)

    browser.open_browser()
    log.info(
        f"opened browser ({cfg.viewport_width}x{cfg.viewport_height}, "
        f"headless={cfg.headless})"
    )
    browser.navigate_to_url(cfg.target_url)
    log.info(f"navigated to {cfg.target_url}")

    contents = []
    pending_response = None  # function_response Part carried to the next user turn
    last_raw = None
    no_progress = 0
    completed = False
    field_focused = False  # True once a click has focused a field, until we type

    try:
        for step in range(cfg.max_steps):
            raw = browser.take_screenshot()
            elements = get_interactive_elements(browser.page)
            annotated = draw_marks(raw, elements)
            log.save_screenshot(step, annotated)

            # no-progress guard: identical screen after an action means stuck
            if last_raw is not None and raw == last_raw:
                no_progress += 1
            else:
                no_progress = 0
            last_raw = raw
            if no_progress >= NO_PROGRESS_LIMIT:
                log.warn(
                    f"no visible progress for {NO_PROGRESS_LIMIT} steps; stopping"
                )
                break

            # this turn's user content: previous tool result + fresh observation
            user_parts = []
            if step == 0:
                user_parts.append(types.Part.from_text(text=task))
            if pending_response is not None:
                user_parts.append(pending_response)
                pending_response = None
            user_parts.append(
                types.Part.from_bytes(data=annotated, mime_type="image/png")
            )
            user_parts.append(
                types.Part.from_text(text=_observation_text(elements))
            )

            contents = _trim_images(contents)  # keep only the latest screenshot
            contents.append(types.Content(role="user", parts=user_parts))

            try:
                decision = brain.act(contents)
            except Exception as err:  # noqa: BLE001 - don't let one call kill the run
                log.warn(f"brain error: {err}")
                break
            contents.append(decision["content"])  # the model's turn
            log.step(step, decision)

            tool = decision["tool"]
            if tool == "task_complete":
                log.done(decision["args"].get("summary", ""))
                completed = True
                break
            if tool is None:
                # model just talked; next turn gives it a fresh screenshot
                time.sleep(cfg.step_delay)
                continue

            # guard the click-then-type flow: typing requires a freshly clicked
            # field, otherwise send_keys would append to the previous one
            if tool == "send_keys" and not field_focused:
                result = (
                    "ERROR: no field is focused. Click the target field first "
                    "(click_on_screen at its listed coordinates), then send_keys."
                )
            else:
                try:
                    result = dispatch[tool](**decision["args"])
                    if isinstance(result, (bytes, bytearray)):
                        result = "screenshot captured"
                except KeyError:
                    result = f"ERROR: unknown tool {tool}"
                except Exception as err:  # noqa: BLE001 - feed it back, don't crash
                    result = f"ERROR: {err}"
                if not str(result).startswith("ERROR"):
                    if tool in ("click_on_screen", "double_click"):
                        field_focused = True
                    elif tool == "send_keys":
                        field_focused = False  # consumed; must click again to type
            log.result(step, result)

            pending_response = types.Part.from_function_response(
                name=tool, response={"result": str(result)}
            )
            time.sleep(cfg.step_delay)
        else:
            log.warn("hit MAX_STEPS without completing")
    finally:
        browser.close_browser()
        log.info("closed browser")

    return completed


def _observation_text(elements):
    return (
        "Here is the current screen. Numbered red boxes mark interactive "
        "elements. Each line below gives the exact coordinates to click for that "
        "element — use them directly, do not guess.\n"
        "Elements:\n" + format_elements(elements)
    )


def _trim_images(contents):
    """Drop stale screenshots so only the newest stays in context."""
    trimmed = []
    for content in contents:
        kept = [
            p for p in content.parts if getattr(p, "inline_data", None) is None
        ]
        if kept:
            trimmed.append(types.Content(role=content.role, parts=kept))
    return trimmed
