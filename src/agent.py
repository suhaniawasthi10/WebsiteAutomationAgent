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
    "You are a web automation agent that controls a real browser by issuing one "
    "tool call at a time. Each turn you receive a screenshot of the current "
    "viewport with numbered red boxes marking interactive elements, plus a list "
    "of those elements. Decide the single best next action and call exactly one "
    "tool.\n"
    "- To act on a numbered element, click the CENTER of its box.\n"
    "- To fill a form field: first click the field to focus it, then call "
    "send_keys to type into it.\n"
    "- Do exactly one action per turn, then wait for the next screenshot.\n"
    "- If a click misses or nothing changes, look again and adjust.\n"
    "- If the target elements are not visible, scroll to reveal them.\n"
    "When the requested fields are filled, call task_complete with a short "
    "summary. Do not call task_complete until the task is actually done."
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

            try:
                result = dispatch[tool](**decision["args"])
                if isinstance(result, (bytes, bytearray)):
                    result = "screenshot captured"
            except KeyError:
                result = f"ERROR: unknown tool {tool}"
            except Exception as err:  # noqa: BLE001 - feed it back, don't crash
                result = f"ERROR: {err}"
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
        "elements; to act on one, click the CENTER of its box.\n"
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
