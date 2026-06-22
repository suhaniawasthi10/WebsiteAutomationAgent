# Architecture

This document explains *why* the agent is built the way it is. For setup and how to run
it, see [`README.md`](README.md).

## What it is

A perceive-then-act agent that drives a real Chromium browser purely from vision. An LLM
(Gemini 2.5 Flash) is given a screenshot of the current page and a set of coordinate-based
tools, and it decides — one action at a time — where to click, what to type, and when the
task is done.

## Why a vision agent and not a selector script

The required tools are coordinate-based: `take_screenshot`, `click_on_screen(x, y)`,
`double_click`, `send_keys`, `scroll`. That points squarely at a perceive-act loop:

```
screenshot -> model (vision + function calling) -> tool call with coords -> execute -> repeat
```

A plain `page.fill("#name", ...)` script would pass the functional task but the agent
wouldn't be *deciding* anything. Here the model reads the page off the screenshot and
figures out where to act. We deliberately **do not hardcode the field labels** — the agent
finds the form on its own, so it still works if the page's labels change. (On the current
shadcn page the demo form is a "Bug Report" with a title input and a description textarea,
not literally "Name"/"Description" — the agent handles that without changes.)

## The perceive-act loop (`agent.py`)

1. A **system instruction** sets the role: you control a browser via tools, you get a
   screenshot each turn with numbered interactive elements, do exactly one action per
   turn, click a field before typing into it, call `task_complete` when finished.
2. Each turn: screenshot the viewport → draw the set-of-marks overlay → send the
   annotated image + the element list to Gemini with the running history → get back one
   function call.
3. Execute the tool, capture the result (success/error string), feed it back to the model
   as a `function_response` on the next turn.
4. Repeat until the model calls `task_complete` or we hit `MAX_STEPS`.

`MAX_STEPS` (default 15) is the infinite-loop guard — without it a confused model can click
forever — and it also keeps us comfortably under the daily free-tier request cap (one run
is ~10–15 requests).

## The detail that makes or breaks it: coordinate mapping (`browser.py`)

The model returns coordinates in **image-pixel** space; `page.mouse.click(x, y)` expects
**CSS pixels** in the viewport. They line up 1:1 only because we:

- create the browser context with `device_scale_factor=1` and a fixed `viewport`,
- take **viewport** screenshots, never `full_page` (a full-page shot includes scrolled-off
  content, so its coordinates wouldn't match what the mouse can reach),
- **never resize** the image before sending it (resizing would force us to scale the
  returned coordinates back, an easy source of bugs).

We keep the viewport modest (1280×800). Vision models lose coordinate accuracy on
large/high-DPI images, so a smaller viewport clicks more reliably than 1920×1080.

## Element detection: set-of-marks (`annotate.py`)

**Baseline:** the model looks at the raw screenshot and returns `click_on_screen(x, y)`.
Simple, but Gemini Flash is decent — not pinpoint — at raw coordinates, so it sometimes
misses by a few pixels.

**Intelligence upgrade (what we ship):** before screenshotting we query the DOM for
interactive elements (`input, textarea, button, [role=textbox], a`), keep the ones visible
in the viewport, and draw a **numbered red box** over each. The model is handed both the
annotated image and a list of `{id, tag, label, bbox}`, where the label is derived from
aria-label / placeholder / associated `<label>` / inner text. The model picks a target and
clicks the **center of its box**. This takes the pixel-precision burden off the model and
is the strongest answer to "how does your agent identify elements intelligently?" It
matters more on a smaller free model like Flash than it would on a larger one.

We still expose `click_on_screen(x, y)` as required — set-of-marks just feeds the model
better information so the coordinates it returns are good.

## Tool design (`tools.py`, `browser.py`)

Every tool is declared to Gemini via `types.FunctionDeclaration` and implemented ourselves
with Playwright (no `browser-use` framework — we wanted to understand what's under the
hood):

| Tool | Playwright call |
|---|---|
| `navigate_to_url(url)` | `page.goto(url, wait_until="domcontentloaded")` + retry |
| `take_screenshot()` | `page.screenshot()` (viewport only) → PNG bytes |
| `click_on_screen(x, y)` | `page.mouse.click(x, y)` (coords validated) |
| `double_click(x, y)` | `page.mouse.dblclick(x, y)` |
| `send_keys(text)` | `page.keyboard.type(text, delay=30)` |
| `scroll(direction, amount)` | `page.mouse.wheel(0, ±amount)` |

`send_keys` takes **no coordinates** on purpose: the flow mirrors how a person fills a
form — `click_on_screen` the field to focus it, then `send_keys` to type. Two internal
tools round it out: `close_browser` (lifecycle, driven by the agent) and
`task_complete(summary)`, which is how the model signals it's finished and ends the loop.

## The brain, behind an interface (`model.py`)

Gemini lives behind a thin `Brain.act(contents) -> {tool, args, text, content}` wrapper.
The rest of the code doesn't know it's Gemini, so it could be swapped for another provider.
Automatic function calling is disabled — we handle exactly one call per turn — and the
temperature is low (0.2) for steadier coordinate picking.

## Free-tier / token hygiene

The free tier is ~10 requests/min, and every turn re-sends the whole `contents`. Stacking
15 screenshots would blow the per-minute token cap and slow everything down, so we keep
**only the latest screenshot** in context (`_trim_images`) — the model doesn't need stale
ones. A small `STEP_DELAY` between steps also helps stay under the rate limit.

## Error handling

- **Navigation:** `navigate_to_url` retries on timeout with backoff and fails loudly after
  exhausting attempts.
- **Rate limits (429 / `RESOURCE_EXHAUSTED`):** the Gemini call retries with exponential
  backoff. This is real on the free tier, so it's built in from the start.
- **Bad/out-of-bounds tool calls:** coordinates are validated against the viewport; an
  invalid call is rejected with an error string fed back to the model instead of crashing.
- **Clicks that land on nothing:** the next screenshot simply shows no change; the loop
  lets the model see that and retry. A no-progress counter stops the run after several
  consecutive unchanged screens.
- Every step is wrapped in try/except so one bad step never kills the run, and the browser
  is always closed via a `finally`.

## Logging & run artifacts (`logger.py`)

Each run gets a `runs/<timestamp>/` folder with `run.log` (step number, the model's stated
reasoning, the chosen tool + args, the result) and one annotated screenshot per step. The
log is also echoed to the console so the run can be followed live, and the saved
screenshots let you scroll back through exactly what the agent saw and did.

## What we'd change to scale this

Cache the accessibility tree between turns, run parallel tabs, and add explicit
self-correction when a click misses (e.g. re-query the element and retry at its center).
