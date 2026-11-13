"""Field-level form tree helpers shared by the form key-value scorer.

A form tree (pages of sections, fields and tables) is flattened into fields, aligned with
the reference fields, and scored per axis (value, label, checkbox, kind, id) as an F1.
"""
