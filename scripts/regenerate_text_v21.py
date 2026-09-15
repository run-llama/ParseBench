"""Regenerate and self-check v2.1 content rules without touching structural rules.

Run with the local ParseBench source installed: python scripts/regenerate_text_v21.py
DATASET [--write]. Without --write, fail if sidecars differ. Always force benchmark
reevaluation after migration. PDFs must be real files, not symlinks to old sidecars.
"""

import argparse
import json
from pathlib import Path

from parse_bench.evaluation.metrics.parse.rules_base import create_test_rule
from parse_bench.evaluation.metrics.parse.text_v21 import RULE_TYPES, reference_rules


def regenerate(data: dict, markdown: str) -> dict:
    rules = reference_rules(markdown) + [r for r in data["test_rules"] if r["type"] not in RULE_TYPES]
    non_batched = [r for r in rules if r["type"] != "required_content"]
    for rule in rules:
        if rule["type"] != "required_content":
            continue
        for name, kind in [("words", "word"), ("sentences", "sentence")]:
            # Explicit curated assertions remain explicit. Only references to
            # generated bags move when the replacement changes rule ordering.
            if name + "_from" not in rule:
                continue
            index = next((i for i, r in enumerate(non_batched) if r["type"] == f"missing_{kind}_percent"), None)
            if index is None:
                rule.pop(name + "_from")
                rule[name] = []
            else:
                rule[name + "_from"] = {"rule_index": index, "field": f"bag_of_{kind}"}
    for rule in rules:
        if rule["type"] in RULE_TYPES:
            result = create_test_rule(rule).run(markdown)
            if result[2] != 1:
                raise ValueError(f"Reference failed its own {rule['type']}: {result}")
    return {**data, "test_rules": rules}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    changed, count = [], 0
    for pdf in sorted(args.dataset.rglob("*.pdf")):
        md, sidecar = pdf.with_suffix(".md"), pdf.with_suffix(".test.json")
        old = json.loads(sidecar.read_text())
        new = regenerate(old, md.read_text())
        if new != old:
            changed.append(str(sidecar.relative_to(args.dataset)))
            if args.write:
                sidecar.write_text(json.dumps(new, ensure_ascii=False, indent=2) + "\n")
        count += 1
    print(json.dumps({"documents": count, "changed": len(changed), "written": args.write}))
    if changed and not args.write:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
