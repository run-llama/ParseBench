"""Read-time normalizer for the renamed FormNode types (form_section ->
section, form_item -> field, form_table -> table).

The scoring vocabulary is the NEW names. On-disk artifacts written before the rename
(GT files, historical predictions, raw output from pre-rename producers) still
carry the old names, so every consumer normalizes trees at its JSON load boundary via
normalize_formnode_types(). Files are never rewritten on disk — normalization happens
on read, keeping old artifacts diffable.
"""

from typing import Any

_TYPE_RENAMES = {
    "form_section": "section",
    "form_item": "field",
    "form_table": "table",
}


def normalize_formnode_types(value: Any) -> Any:
    """Recursively map pre-rename FormNode `type` values to the new names, in place.

    Safe on mixed/new-name trees (unknown types untouched); returns `value` for chaining.
    """
    if isinstance(value, list):
        for entry in value:
            normalize_formnode_types(entry)
        return value
    if isinstance(value, dict):
        t = value.get("type")
        if isinstance(t, str) and t in _TYPE_RENAMES:
            value["type"] = _TYPE_RENAMES[t]
        for child in value.values():
            normalize_formnode_types(child)
    return value
