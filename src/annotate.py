"""Set-of-marks overlay — the intelligence layer (see context.md).

Instead of asking the model to pixel-hunt on a raw screenshot, we find the
interactive elements via the DOM, draw numbered boxes over them, and hand the
model both the annotated image and a list of {id, tag, label, bbox}. The model
picks an id; we click the box center. This matters more on Gemini Flash, which
is decent but not pinpoint at raw coordinates.
"""

import io

from PIL import Image, ImageDraw, ImageFont

SELECTOR = "input, textarea, button, [role=textbox], a"

# JS run on each element to derive a human-readable label from the best source
# available (aria-label, placeholder, associated/wrapping <label>, then text).
_LABEL_JS = r"""
el => {
  const clean = s => (s || '').trim().replace(/\s+/g, ' ');
  let label = el.getAttribute('aria-label') || el.getAttribute('placeholder') || '';
  if (!label && el.id) {
    const lab = document.querySelector('label[for="' + el.id + '"]');
    if (lab) label = lab.textContent;
  }
  if (!label) {
    const wrap = el.closest('label');
    if (wrap) label = wrap.textContent;
  }
  if (!label) label = el.value || el.innerText || el.textContent || '';
  label = clean(label);
  return label.length > 40 ? label.slice(0, 40) + '…' : label;
}
"""


def get_interactive_elements(page):
    """Return interactive elements visible in the current viewport.

    bounding_box() is viewport-relative (accounts for scroll), so its coords map
    straight onto the screenshot and onto page.mouse.click.
    """
    size = page.viewport_size or {"width": 0, "height": 0}
    vw, vh = size["width"], size["height"]

    elements = []
    idx = 0
    for handle in page.query_selector_all(SELECTOR):
        try:
            if not handle.is_visible():
                continue
            box = handle.bounding_box()
        except Exception:
            continue
        if not box:
            continue
        x, y, w, h = box["x"], box["y"], box["width"], box["height"]
        if w <= 1 or h <= 1:
            continue
        # keep only boxes that actually overlap the viewport
        if x + w < 0 or y + h < 0 or x > vw or y > vh:
            continue
        try:
            tag = handle.evaluate("el => el.tagName.toLowerCase()")
            label = handle.evaluate(_LABEL_JS)
        except Exception:
            continue
        elements.append(
            {"id": idx, "tag": tag, "label": label, "bbox": [x, y, w, h]}
        )
        idx += 1
    return elements


def draw_marks(png_bytes, elements):
    """Draw a numbered red box over each element; return annotated PNG bytes."""
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _load_font(15)

    for el in elements:
        x, y, w, h = el["bbox"]
        x0, y0, x1, y1 = int(x), int(y), int(x + w), int(y + h)
        draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=2)

        tag = str(el["id"])
        tw, th = _text_dims(draw, tag, font)
        bx0 = x0
        by0 = max(0, y0 - th - 4)
        draw.rectangle([bx0, by0, bx0 + tw + 6, by0 + th + 4], fill=(255, 0, 0))
        draw.text((bx0 + 3, by0 + 2), tag, fill=(255, 255, 255), font=font)

    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


def element_center(el):
    """Resolve an element id/box to the click point (its center)."""
    x, y, w, h = el["bbox"]
    return int(x + w / 2), int(y + h / 2)


def format_elements(elements):
    """One line per element for the text the model sees alongside the image."""
    if not elements:
        return "(no interactive elements detected)"
    return "\n".join(
        f'[{el["id"]}] {el["tag"]} "{el["label"]}"' for el in elements
    )


def _load_font(size):
    for path in (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _text_dims(draw, text, font):
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top
