"""Table zoom-pass prompt (shared by the live provider and offline replays)."""

PROMPT = """You are acting purely as a vision transcription model. Do NOT run any commands.
The attached image is a crop showing ONE table from a document page.
Transcribe it as exactly ONE HTML <table>:
- header row(s) inside <thead> using <th>; body rows inside <tbody> using <td>
- use rowspan / colspan for merged cells so every row spans the same number of columns
- copy every cell's text exactly; keep empty cells as empty <td></td>; do not drop rows or columns
- no commentary, no markdown fences — output ONLY the <table>...</table>"""
