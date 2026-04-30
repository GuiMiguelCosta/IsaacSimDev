from __future__ import annotations

from pathlib import Path


def clean_repeated_absolute_path(raw_path: str) -> str:
    """Collapse accidental repeated absolute paths from editable path fields.

    Input is raw UI/config text; the output is stripped path text. This exists
    because pasting into an Omni string field can leave values such as
    /path/to/folder/path/to/folder, which should resolve to /path/to/folder
    when the repeated prefix is the real existing path.
    """

    path_text = str(raw_path or "").strip()
    if not path_text:
        return ""

    expanded_path_text = str(Path(path_text).expanduser())
    if Path(expanded_path_text).exists() or not Path(expanded_path_text).is_absolute():
        return expanded_path_text

    for split_index in range(1, len(expanded_path_text)):
        candidate = expanded_path_text[:split_index]
        if not expanded_path_text.startswith(candidate, split_index):
            continue
        if not Path(candidate).exists():
            continue

        remainder = expanded_path_text[split_index:]
        while remainder.startswith(candidate):
            remainder = remainder[len(candidate) :]
        if not remainder:
            return candidate

    return expanded_path_text
