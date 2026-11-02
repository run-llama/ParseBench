# Leaderboard accountability

Every row of [`leaderboard.csv`](../leaderboard.csv) is tied to the outputs it was scored from, so anyone can
re-score it and the numbers stay current as the harness improves.

- [`results.yaml`](results.yaml) maps each row (`provider` = the csv `Provider`) to a pipeline folder in
  [`llamaindex/ParseBench-results`](https://huggingface.co/datasets/llamaindex/ParseBench-results)
  (`results/<pipeline>/<category>.jsonl`) at a pinned `results_revision`.
- **Verified** rows (`Verified=yes` in the csv) are re-scored from those outputs with the current ParseBench
  harness against the ground truth at `data_revision`. Pages the provider produced nothing usable for (no record,
  empty output, no layout) score 0, exactly as in `parse-bench run`. If the evaluator leaves any page of a dimension
  out of its average (a harness error such as an unknown label, or layout data missing from the record), that
  dimension is not re-scored: it keeps its old value and is listed in the re-score report.
- **Unverified** rows (`Verified=no`, marked † in the README) have no published outputs, or published outputs that miss
  more than 1% of a category's pages; their numbers are as submitted, and the map's `note` says what is missing.
  They cannot be added or changed until complete outputs are published.

## Adding or updating a row

1. Run the pipeline and publish its outputs to the results repo as a pull request (see the
   [results repo README](https://huggingface.co/datasets/llamaindex/ParseBench-results)):
   `pack.py` → `validate.py` → `hf upload ... --create-pr`.
2. In your ParseBench PR, add or update the row in `leaderboard.csv` and its entry in `results.yaml`:
   ```yaml
   - provider: My Parser            # exactly the csv Provider
     status: verified
     pipeline: my_parser            # folder under results/
     results_revision: refs/pr/12   # the results-repo PR; a maintainer swaps in the merge commit
   ```
   Then run `python scripts/leaderboard/check.py --fix` to set the `Verified` column.
3. The **Leaderboard verification** workflow checks that every row is mapped, that the folder exists, and
   re-scores the added or changed rows (about 20 minutes per row). It fails if a submitted value differs from the
   re-score by more than 0.5 points, and posts the comparison on the PR.

The verification scores with the **base branch's** harness and reads only `leaderboard.csv` and `results.yaml`
from the PR, so it is safe for fork PRs. If your PR also changes scoring code, land that first.

## Scheduled re-score

The **Leaderboard re-score** workflow runs daily. If `src/parse_bench` or `scripts/leaderboard` (or the ground truth
revision) changed since `scored_with` in `results.yaml`, it re-scores every verified row (one job per pipeline, 20
at a time) and opens or updates the `leaderboard/rescore` pull request with the new numbers. Changes of 2 points or more are flagged ⚠️ and labelled
`large-score-change`. Run it by hand with *Run workflow* (tick **force** to re-score without a change).
Pull requests opened with the default `GITHUB_TOKEN` do not trigger other workflows, so CI does not run on the
re-score PR on its own; push an empty commit or re-run CI from the PR if you want it.

## Running locally

```bash
python scripts/leaderboard/check.py                     # map / csv consistency
python scripts/leaderboard/check.py --hf                # also check the HF folders
python scripts/leaderboard/rescore.py run --pipelines llamaparse_agentic --out rescore.json   # or omit --pipelines
python scripts/leaderboard/apply.py apply rescore.json --report report.md   # write new scores
python scripts/leaderboard/apply.py compare rescore.json                    # check without writing
```

`rescore.py run` downloads the ground truth at `data_revision` once into `.cache/leaderboard-gt/`.

## Repository setup

No secrets are needed once `llamaindex/ParseBench-results` is public. While it is private, add an `HF_TOKEN`
secret with read access; both workflows pass it through when present. Create the `leaderboard-rescore` and
`large-score-change` labels, and allow GitHub Actions to create pull requests (*Settings → Actions → General*).
