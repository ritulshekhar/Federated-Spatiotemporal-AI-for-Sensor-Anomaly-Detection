"""Assemble the dashboard page by inlining its data bundle into the template.

The published artifact is a single HTML file — an artifact cannot fetch a
sibling JSON — so the bundle is injected at build time rather than loaded at
run time. Keeping the template and the data separate means the page can be
regenerated from a new run without editing any markup.

    python -m dashboard.build_data     # read outputs/ -> dashboard/data.json
    python -m dashboard.build_page     # template + data -> dashboard/index.html
"""

from __future__ import annotations

import json
from pathlib import Path

from src.utils.logging_setup import get_logger

log = get_logger(__name__)

HERE = Path(__file__).parent
TEMPLATE = HERE / "template.html"
DATA = HERE / "data.json"
OUT = HERE / "index.html"
TOKEN = "__FEDAIRGUARD_DATA__"


def build() -> str:
    template = TEMPLATE.read_text(encoding="utf-8")
    if TOKEN not in template:
        raise ValueError(f"{TEMPLATE.name} has no {TOKEN} placeholder")
    bundle = json.loads(DATA.read_text(encoding="utf-8"))

    # Compact, and with the sequence that would close the script block escaped.
    payload = json.dumps(bundle, separators=(",", ":")).replace("</", "<\\/")
    return template.replace(TOKEN, payload)


def main() -> None:
    html = build()
    OUT.write_text(html, encoding="utf-8")
    log.info("wrote %s (%.1f KB)", OUT, OUT.stat().st_size / 1024)


if __name__ == "__main__":
    main()
