"""ByteVerity parse provider.

A VLM proposer (configurable; GPT-6 Luna for the submitted rows) plus deterministic PDF facts, an open-source layout
detector (docling-layout-heron, Apache-2.0) and 17 sealed decision oracles shipped as complete decision tables.
"""

from . import provider  # noqa: F401  (registers the `byteverity` provider)
