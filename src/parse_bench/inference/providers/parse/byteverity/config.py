"""Runtime configuration for the ByteVerity provider.

BYTEVERITY_TABLES      directory of sealed decision tables (default: the bundled rulebook/tables)
BYTEVERITY_VISION_PY   python with torch+transformers for the layout detector, if not importable in-process
"""

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
