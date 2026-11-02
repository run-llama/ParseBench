"""Extract providers imported for registry side effects.

Only the minimal ParseBench EXTRACT integration providers are registered here.
Imports are best-effort so base ParseBench imports do not require optional
provider SDKs such as ``extend-ai``.
"""

import importlib
import logging

logger = logging.getLogger(__name__)

_PROVIDER_MODULES = [
    "extend",
    "llamaextract_v2_api",
]


def load_providers() -> None:
    """Import each provider module so it registers itself; skip ones whose SDK is missing.

    Called by ``registry.load_builtin_providers`` on first use rather than on package
    import, so helpers under this package (such as label adapters) stay cheap to import.
    """
    for module in _PROVIDER_MODULES:
        try:
            importlib.import_module(f"parse_bench.inference.providers.extract.{module}")
        except ImportError:
            logger.debug("Skipping extract provider %s (missing dependency)", module)
