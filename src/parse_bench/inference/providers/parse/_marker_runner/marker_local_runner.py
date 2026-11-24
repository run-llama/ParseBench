"""Convert one file with a local Marker install and print {"markdown", "json"} as one JSON object.

Run with the Python of the environment Marker is installed in (``pip install marker-pdf``):

    python marker_local_runner.py FILE [--mode fast]

One conversion yields both outputs (``marker_single`` writes a single format per run). Marker's
defaults are kept: in particular image extraction stays on, since Marker's layout step only
detects pictures when it is (marker.builders.layout).
"""

import json
import sys


def main() -> None:
    args = sys.argv[1:]
    if not args:
        sys.exit("usage: marker_local_runner.py FILE [--mode MODE]")
    path = args[0]
    config: dict = {}
    if "--mode" in args:
        config["mode"] = args[args.index("--mode") + 1]

    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.renderers.json import JSONRenderer
    from marker.renderers.markdown import MarkdownRenderer

    converter = PdfConverter(artifact_dict=create_model_dict(), config=config)
    document = converter.build_document(path)
    markdown = converter.resolve_dependencies(MarkdownRenderer)(document).markdown
    rendered = converter.resolve_dependencies(JSONRenderer)(document)
    json.dump({"markdown": markdown, "json": _strip_images(rendered.model_dump(mode="json"))}, sys.stdout)


def _strip_images(node):
    """Drop the base64 image payloads; the layout only needs block types and boxes."""
    if isinstance(node, dict):
        return {k: _strip_images(v) for k, v in node.items() if k != "images"}
    if isinstance(node, list):
        return [_strip_images(v) for v in node]
    return node


if __name__ == "__main__":
    main()
