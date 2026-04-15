# Table Metric Visualizations

Manim Community Edition animations explaining table similarity metrics: TEDS, GriTS, and TableRecordMatch (from ParseBench).

## Videos

- **`teds_vs_grits.py`** — TEDS vs GriTS comparison. Covers tree vs grid representations, edit operations, formulas, and the row-column asymmetry problem.
- **`table_metrics.py`** — All three metrics. Adds TableRecordMatch and GTRM, showing how column reordering and header errors are handled differently.

## Reproduce

```bash
# Install ManimCE (requires Python 3.9+, ffmpeg, cairo, pango)
pip install manim

# Render high quality (1080p60)
manim -qh teds_vs_grits.py TEDSvsGRITS
manim -qh table_metrics.py TableMetrics

# Render low quality for quick preview
manim -ql teds_vs_grits.py TEDSvsGRITS
manim -ql table_metrics.py TableMetrics
```

Output appears in `media/videos/`.

## System dependencies (macOS)

```bash
brew install cairo pango ffmpeg
```
