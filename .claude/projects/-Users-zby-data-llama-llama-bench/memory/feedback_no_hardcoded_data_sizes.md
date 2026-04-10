---
name: No hardcoded dataset sizes
description: Never hardcode test case counts, PDF counts, or dataset sizes in docs — the dataset is evolving and current data/ is just for early testing
type: feedback
---

Do not mention specific dataset sizes (number of test cases, PDFs, categories counts, rule type counts) in README, docs, or user-facing text. The dataset in the repo is temporary test data. The real dataset will be hosted on HuggingFace and will grow over time. Point users to `llama-bench info` to see current stats after downloading.
