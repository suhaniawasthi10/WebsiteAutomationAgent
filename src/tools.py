"""Gemini function declarations + a name->callable dispatch map.

These declarations are the only tools the model is allowed to call during the
loop. Lifecycle (open/close browser) and the per-turn screenshot are driven by
the agent, not the model. `task_complete` is how the model ends the run; it is
handled by the loop, not dispatched to the browser.
"""

from google.genai import types

navigate = types.FunctionDeclaration(
    name="navigate_to_url",
    description="Load a URL in the browser.",
    parameters={
        "type": "object",
        "properties": {"url": {"type": "string"}},
        "required": ["url"],
    },
)

click = types.FunctionDeclaration(
    name="click_on_screen",
    description=(
        "Click at viewport pixel coordinates. To target a numbered element, "
        "click the center of its box."
    ),
    parameters={
        "type": "object",
        "properties": {
            "x": {"type": "integer"},
            "y": {"type": "integer"},
        },
        "required": ["x", "y"],
    },
)

double_click = types.FunctionDeclaration(
    name="double_click",
    description="Double-click at viewport pixel coordinates.",
    parameters={
        "type": "object",
        "properties": {
            "x": {"type": "integer"},
            "y": {"type": "integer"},
        },
        "required": ["x", "y"],
    },
)

send_keys = types.FunctionDeclaration(
    name="send_keys",
    description="Type text into the currently focused element. Click a field first.",
    parameters={
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
)

scroll = types.FunctionDeclaration(
    name="scroll",
    description="Scroll the page to reveal off-screen elements.",
    parameters={
        "type": "object",
        "properties": {
            "direction": {"type": "string", "enum": ["up", "down"]},
            "amount": {"type": "integer"},
        },
        "required": ["direction", "amount"],
    },
)

take_screenshot = types.FunctionDeclaration(
    name="take_screenshot",
    description="Capture the current viewport. A fresh screenshot is provided each turn.",
    parameters={"type": "object", "properties": {}},
)

task_complete = types.FunctionDeclaration(
    name="task_complete",
    description="Call this when the task is done. Provide a short summary of what you did.",
    parameters={
        "type": "object",
        "properties": {"summary": {"type": "string"}},
        "required": ["summary"],
    },
)

GEMINI_TOOLS = types.Tool(
    function_declarations=[
        navigate,
        click,
        double_click,
        send_keys,
        scroll,
        take_screenshot,
        task_complete,
    ]
)

# Every declared tool name, for coverage checks.
TOOL_NAMES = [d.name for d in GEMINI_TOOLS.function_declarations]

# Tools the agent loop handles itself rather than dispatching to the browser.
LOOP_HANDLED = {"task_complete"}


def build_dispatch(browser):
    """Map each model-callable tool name to the browser method that runs it."""
    return {
        "navigate_to_url": browser.navigate_to_url,
        "click_on_screen": browser.click_on_screen,
        "double_click": browser.double_click,
        "send_keys": browser.send_keys,
        "scroll": browser.scroll,
        "take_screenshot": browser.take_screenshot,
    }
