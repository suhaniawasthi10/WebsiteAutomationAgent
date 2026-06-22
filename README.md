# Website Automation Agent

A mini [browser-use](https://github.com/browser-use/browser-use) clone. An LLM looks at
screenshots of a live web page and drives a **real** browser by issuing tool calls
(click, type, scroll) one step at a time until the task is done.

**Target task:** open the shadcn *React Hook Form* docs page, find the form fields on
its own, and fill them in — without any hardcoded selectors.

The whole stack is free: **Playwright** (browser), **Pillow** (annotation overlay), and
**Gemini 2.5 Flash** (vision + function calling) on Google's free tier — no credit card.

---

## How it works (in one diagram)

```
screenshot -> annotate (set-of-marks) -> Gemini (vision + function call)
   ^                                                      |
   |                                                      v
   +------------------ execute tool (Playwright) <--------+
```

Each turn the agent screenshots the viewport, draws numbered boxes over the interactive
elements, sends that to Gemini, gets back **one** tool call, executes it, and repeats
until the model calls `task_complete` (or it hits `MAX_STEPS`). See
[`ARCHITECTURE.md`](ARCHITECTURE.md) for the design decisions.

---

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
```

`requirements.txt`: `playwright`, `google-genai`, `python-dotenv`, `pillow`.

> Note: the SDK is **`google-genai`** (`from google import genai`), *not* the deprecated
> `google-generativeai`.

### Get a free Gemini API key (≈1 minute, no card)

1. Go to **[Google AI Studio → API keys](https://aistudio.google.com/apikey)**.
2. Sign in with a Google account and click **Create API key**.
3. Copy the key.

### Configure

```bash
cp .env.example .env
```

Then open `.env` and paste your key into `GEMINI_API_KEY`. The keys:

| Key | Default | Meaning |
|---|---|---|
| `GEMINI_API_KEY` | — | your AI Studio key (required) |
| `MODEL` | `gemini-2.5-flash` | the model id |
| `HEADLESS` | `false` | `false` shows the browser live (good for a demo) |
| `VIEWPORT_WIDTH` | `1280` | viewport width in CSS pixels |
| `VIEWPORT_HEIGHT` | `800` | viewport height in CSS pixels |
| `MAX_STEPS` | `15` | infinite-loop guard |
| `STEP_DELAY` | `2` | seconds between steps (keeps you under ~10 req/min) |
| `TARGET_URL` | shadcn react-hook-form docs | page to automate |

---

## Run

```bash
python main.py
```

With `HEADLESS=false` a Chromium window opens, navigates to the target page, and you
watch the agent scroll to the form, click each field, type a sample value, and finish.

### What to expect

- Console output narrates every step (reasoning, the chosen tool + args, the result).
- A new folder `runs/<timestamp>/` is created with `run.log` and one annotated
  screenshot per step (`step_00.png`, `step_01.png`, …) — proof of exactly what the
  agent saw and did. `runs/` is gitignored.
- The run ends when the model calls `task_complete`, or after `MAX_STEPS` steps.

### If something misbehaves

- **Clicks miss:** lower the viewport (e.g. `1024x768`) — vision models are more
  accurate at lower resolution — and rely on the set-of-marks boxes.
- **`429` / rate limit:** raise `STEP_DELAY`. The Gemini call also retries with
  exponential backoff automatically.
- **It loops:** the system instruction is tightened to do one action per turn and call
  `task_complete` when done; `MAX_STEPS` is the hard stop either way.

---

## Project layout

```
src/
  config.py    # env + typed settings
  logger.py    # per-run folder: screenshots + run.log
  browser.py   # Playwright wrapper = the tool implementations
  annotate.py  # set-of-marks overlay (the intelligence layer)
  tools.py     # Gemini function declarations + dispatch map
  model.py     # Gemini client wrapper (the brain)
  agent.py     # the perceive-act loop
main.py        # entry point, wires everything together
```
