from manim import *
import numpy as np

# ── Color Palette ──────────────────────────────────────────────
BG = "#1a1a2e"
TEDS_COL = "#4ade80"       # green
GRITS_COL = "#22d3ee"      # cyan
TRM_COL = "#c084fc"        # purple
GTRM_COL = "#f472b6"       # pink
HIGHLIGHT = "#facc15"      # yellow
ERR = "#f87171"            # red
GOOD = "#4ade80"           # green
CELL_FILL = "#334155"
CELL_STROKE = "#94a3b8"
HEADER_FILL = "#1e3a5f"
HEADER_FILL_2 = "#3b1f5e"


class TableMetrics(Scene):
    """TEDS vs GriTS vs TableRecordMatch — full explainer."""

    def setup(self):
        self.camera.background_color = BG

    def construct(self):
        self.intro()
        self.recap_teds_grits()
        self.problem_reordering()
        self.problem_headers()
        self.introduce_trm()
        self.trm_handles_both()
        self.gtrm_combined()
        self.summary()
        self.outro()

    # ── helpers ─────────────────────────────────────────────────
    def make_table(self, data, cw=1.0, ch=0.42, font_size=16,
                   header=True, header_color=HEADER_FILL,
                   highlight_cells=None, dim_cells=None):
        rows, cols = len(data), len(data[0])
        rects = VGroup()
        texts = VGroup()
        for i in range(rows):
            for j in range(cols):
                fill = header_color if (header and i == 0) else CELL_FILL
                opacity = 0.7 if (header and i == 0) else 0.4
                sc = CELL_STROKE
                sw = 1.5
                if highlight_cells and (i, j) in highlight_cells:
                    sc = ERR
                    sw = 3
                    fill = ERR
                    opacity = 0.25
                if dim_cells and (i, j) in dim_cells:
                    opacity = 0.1
                rect = Rectangle(width=cw, height=ch,
                                 fill_color=fill, fill_opacity=opacity,
                                 stroke_color=sc, stroke_width=sw)
                rect.move_to(np.array([j * cw, -i * ch, 0]))
                t = Text(str(data[i][j]), font_size=font_size, color=WHITE)
                t.move_to(rect.get_center())
                rects.add(rect)
                texts.add(t)
        table = VGroup(rects, texts)
        table.move_to(ORIGIN)
        return table, rects, texts

    def section_title(self, text, color=WHITE, size=44):
        t = Text(text, font_size=size, weight=BOLD, color=color)
        t.to_edge(UP, buff=0.45)
        return t

    def badge(self, label, color, width=2.8):
        box = RoundedRectangle(width=width, height=0.55, corner_radius=0.1,
                               color=color, fill_opacity=0.2, stroke_width=2)
        t = Text(label, font_size=18, weight=BOLD, color=color)
        t.move_to(box.get_center())
        return VGroup(box, t)

    def penalty_indicator(self, level, position):
        """level: 'high', 'low', 'none'."""
        colors = {"high": ERR, "low": HIGHLIGHT, "none": GOOD}
        labels = {"high": "Heavy penalty", "low": "Mild penalty",
                  "none": "No penalty"}
        col = colors[level]
        t = Text(labels[level], font_size=16, weight=BOLD, color=col)
        t.move_to(position)
        return t

    # ── Scene 1 : Intro ───────────────────────────────────────
    def intro(self):
        title = Text("Measuring Table Similarity", font_size=52, weight=BOLD)
        sub = Text("TEDS  ·  GriTS  ·  TableRecordMatch",
                    font_size=26, color=GREY_B)
        sub2 = Text("ParseBench (Zhang et al., 2025)", font_size=20,
                     color=GREY_C)
        g = VGroup(title, sub, sub2).arrange(DOWN, buff=0.4)
        self.play(Write(title), run_time=1.2)
        self.play(FadeIn(sub, shift=UP * 0.3))
        self.play(FadeIn(sub2, shift=UP * 0.2))
        self.wait(2)
        self.play(FadeOut(g))

    # ── Scene 2 : Quick recap ─────────────────────────────────
    def recap_teds_grits(self):
        title = self.section_title("Quick Recap")
        self.play(Write(title), run_time=0.6)

        # TEDS card
        teds_card = RoundedRectangle(width=5.2, height=2.8, corner_radius=0.15,
                                     color=TEDS_COL, fill_opacity=0.08,
                                     stroke_width=2)
        teds_name = Text("TEDS", font_size=32, weight=BOLD, color=TEDS_COL)
        teds_items = VGroup(
            Text("HTML tree representation", font_size=16, color=GREY_A),
            Text("Tree edit distance", font_size=16, color=GREY_A),
            Text("Rows ≠ columns (asymmetric)", font_size=16, color=ERR),
        ).arrange(DOWN, buff=0.18, aligned_edge=LEFT)
        teds_content = VGroup(teds_name, teds_items).arrange(DOWN, buff=0.3)
        teds_content.move_to(teds_card)

        # GriTS card
        grits_card = RoundedRectangle(width=5.2, height=2.8, corner_radius=0.15,
                                      color=GRITS_COL, fill_opacity=0.08,
                                      stroke_width=2)
        grits_name = Text("GriTS", font_size=32, weight=BOLD, color=GRITS_COL)
        grits_items = VGroup(
            Text("2D grid / matrix representation", font_size=16, color=GREY_A),
            Text("Optimal substructure matching", font_size=16, color=GREY_A),
            Text("Rows = columns (symmetric)", font_size=16, color=GOOD),
        ).arrange(DOWN, buff=0.18, aligned_edge=LEFT)
        grits_content = VGroup(grits_name, grits_items).arrange(DOWN, buff=0.3)
        grits_content.move_to(grits_card)

        cards = VGroup(
            VGroup(teds_card, teds_content),
            VGroup(grits_card, grits_content),
        ).arrange(RIGHT, buff=0.6).next_to(title, DOWN, buff=0.5)

        self.play(
            FadeIn(teds_card), FadeIn(teds_content),
            FadeIn(grits_card), FadeIn(grits_content),
            run_time=1.2,
        )
        self.wait(1.5)

        # Shared problem
        problem = Text(
            "Both focus on structural similarity...\n"
            "but what about semantic correctness?",
            font_size=22, color=HIGHLIGHT, line_spacing=1.3,
        )
        problem.to_edge(DOWN, buff=0.5)
        self.play(Write(problem), run_time=1.2)
        self.wait(2)

        self.play(FadeOut(VGroup(
            title, teds_card, teds_content, grits_card, grits_content, problem
        )))

    # ── Scene 3 : Problem — Column Reordering ─────────────────
    def problem_reordering(self):
        title = self.section_title("Problem 1: Column Reordering", color=ERR)
        self.play(Write(title), run_time=0.8)

        # Ground truth
        gt_data = [["Name", "Age", "City"],
                   ["Alice", "30",  "NYC"],
                   ["Bob",   "25",  "LA"]]
        gt, _, _ = self.make_table(gt_data, cw=1.1)
        gt_lab = Text("Ground Truth", font_size=20, color=WHITE, weight=BOLD)
        gt.shift(LEFT * 3.5 + UP * 0.3)
        gt_lab.next_to(gt, UP, buff=0.2)

        # Prediction: columns reordered
        pred_data = [["City", "Name", "Age"],
                     ["NYC",  "Alice", "30"],
                     ["LA",   "Bob",   "25"]]
        pred, _, _ = self.make_table(pred_data, cw=1.1)
        pred_lab = Text("Prediction (cols swapped)", font_size=20,
                        color=HIGHLIGHT, weight=BOLD)
        pred.shift(RIGHT * 3.5 + UP * 0.3)
        pred_lab.next_to(pred, UP, buff=0.2)

        self.play(FadeIn(gt), Write(gt_lab))
        self.play(FadeIn(pred), Write(pred_lab))
        self.wait(0.5)

        # Semantically identical note
        sem = Text("Semantically identical — same data, different column order",
                    font_size=18, color=GREY_B)
        sem.next_to(VGroup(gt, pred), DOWN, buff=0.4)
        self.play(Write(sem))
        self.wait(1)

        # But TEDS/GriTS penalize it
        penalty_box = RoundedRectangle(width=11, height=1.6, corner_radius=0.12,
                                       color=ERR, fill_opacity=0.08,
                                       stroke_width=2)
        penalty_box.to_edge(DOWN, buff=0.4)

        teds_p = VGroup(
            Text("TEDS:", font_size=18, weight=BOLD, color=TEDS_COL),
            Text("Heavy penalty", font_size=18, weight=BOLD, color=ERR),
            Text("(cells at wrong tree positions)", font_size=14, color=GREY_B),
        ).arrange(RIGHT, buff=0.15)

        grits_p = VGroup(
            Text("GriTS:", font_size=18, weight=BOLD, color=GRITS_COL),
            Text("Heavy penalty", font_size=18, weight=BOLD, color=ERR),
            Text("(column order changes grid alignment)", font_size=14,
                 color=GREY_B),
        ).arrange(RIGHT, buff=0.15)

        penalties = VGroup(teds_p, grits_p).arrange(DOWN, buff=0.2)
        penalties.move_to(penalty_box)

        self.play(FadeIn(penalty_box))
        self.play(Write(teds_p), run_time=0.8)
        self.play(Write(grits_p), run_time=0.8)
        self.wait(2)

        self.play(FadeOut(VGroup(
            title, gt, gt_lab, pred, pred_lab, sem, penalty_box, penalties
        )))

    # ── Scene 4 : Problem — Header Dropping ───────────────────
    def problem_headers(self):
        title = self.section_title("Problem 2: Header Errors", color=ERR)
        self.play(Write(title), run_time=0.8)

        # Ground truth
        gt_data = [["Name", "Age", "City"],
                   ["Alice", "30",  "NYC"],
                   ["Bob",   "25",  "LA"]]
        gt, _, _ = self.make_table(gt_data, cw=1.1)
        gt_lab = Text("Ground Truth", font_size=20, color=WHITE, weight=BOLD)

        # Prediction: headers transposed
        pred_data = [["Age", "Name", "City"],
                     ["Alice", "30",  "NYC"],
                     ["Bob",   "25",  "LA"]]
        pred, pred_r, _ = self.make_table(
            pred_data, cw=1.1,
            highlight_cells={(0, 0), (0, 1)},
        )
        pred_lab = Text("Prediction (headers swapped)", font_size=20,
                        color=ERR, weight=BOLD)

        gt.shift(LEFT * 3.5 + UP * 0.3)
        pred.shift(RIGHT * 3.5 + UP * 0.3)
        gt_lab.next_to(gt, UP, buff=0.2)
        pred_lab.next_to(pred, UP, buff=0.2)

        self.play(FadeIn(gt), Write(gt_lab))
        self.play(FadeIn(pred), Write(pred_lab))
        self.wait(0.5)

        # This is catastrophic
        cat = Text(
            'Catastrophic — "Alice" is now keyed as "Age",  "30" as "Name"',
            font_size=17, color=ERR,
        )
        cat.next_to(VGroup(gt, pred), DOWN, buff=0.35)
        self.play(Write(cat))
        self.wait(1)

        # But TEDS/GriTS treat it as mild
        penalty_box = RoundedRectangle(width=11, height=1.6, corner_radius=0.12,
                                       color=HIGHLIGHT, fill_opacity=0.08,
                                       stroke_width=2)
        penalty_box.to_edge(DOWN, buff=0.4)

        teds_p = VGroup(
            Text("TEDS:", font_size=18, weight=BOLD, color=TEDS_COL),
            Text("Mild penalty", font_size=18, weight=BOLD, color=HIGHLIGHT),
            Text("(only 2 header cells differ)", font_size=14, color=GREY_B),
        ).arrange(RIGHT, buff=0.15)

        grits_p = VGroup(
            Text("GriTS:", font_size=18, weight=BOLD, color=GRITS_COL),
            Text("Mild penalty", font_size=18, weight=BOLD, color=HIGHLIGHT),
            Text("(structure is mostly correct)", font_size=14, color=GREY_B),
        ).arrange(RIGHT, buff=0.15)

        penalties = VGroup(teds_p, grits_p).arrange(DOWN, buff=0.2)
        penalties.move_to(penalty_box)

        self.play(FadeIn(penalty_box))
        self.play(Write(teds_p), run_time=0.8)
        self.play(Write(grits_p), run_time=0.8)
        self.wait(1.5)

        # The core issue
        issue = Text(
            "Structure-focused metrics miss semantic correctness!",
            font_size=20, weight=BOLD, color=ERR,
        )
        issue.next_to(penalty_box, UP, buff=0.15)
        self.play(Write(issue))
        self.wait(2)

        self.play(FadeOut(VGroup(
            title, gt, gt_lab, pred, pred_lab, cat,
            penalty_box, penalties, issue,
        )))

    # ── Scene 5 : Introduce TableRecordMatch ──────────────────
    def introduce_trm(self):
        title = self.section_title("TableRecordMatch", color=TRM_COL)
        sub = Text("from ParseBench (Zhang et al., 2025)",
                    font_size=20, color=GREY_B)
        sub.next_to(title, DOWN, buff=0.12)
        self.play(Write(title), FadeIn(sub, shift=UP * 0.2))

        # Core idea
        idea = Text("Treat a table as a bag of records",
                     font_size=26, color=HIGHLIGHT, weight=BOLD)
        idea.next_to(sub, DOWN, buff=0.5)
        self.play(Write(idea))
        self.wait(1)

        # Show a table transforming into records
        gt_data = [["Name", "Age", "City"],
                   ["Alice", "30",  "NYC"],
                   ["Bob",   "25",  "LA"]]
        tbl, _, _ = self.make_table(gt_data, cw=1.1, header_color=HEADER_FILL_2)
        tbl.shift(LEFT * 4.0 + DOWN * 1.0)

        self.play(FadeIn(tbl))
        self.wait(0.5)

        # Arrow
        arr = Arrow(tbl.get_right() + RIGHT * 0.2,
                    RIGHT * 0.2 + DOWN * 0.5,
                    color=TRM_COL, stroke_width=2.5, buff=0.1)
        self.play(GrowArrow(arr), run_time=0.5)

        # Records as key-value cards
        def record_card(pairs, idx):
            items = VGroup()
            for k, v in pairs:
                key_t = Text(f"{k}:", font_size=14, weight=BOLD, color=TRM_COL)
                val_t = Text(f" {v}", font_size=14, color=WHITE)
                row = VGroup(key_t, val_t).arrange(RIGHT, buff=0.05)
                items.add(row)
            items.arrange(DOWN, buff=0.1, aligned_edge=LEFT)
            box = RoundedRectangle(
                width=items.get_width() + 0.4,
                height=items.get_height() + 0.3,
                corner_radius=0.1, color=TRM_COL,
                fill_opacity=0.12, stroke_width=1.5,
            )
            label = Text(f"Record {idx}", font_size=12, color=GREY_B)
            label.next_to(box, UP, buff=0.08)
            items.move_to(box)
            return VGroup(box, items, label)

        r1 = record_card([("Name", "Alice"), ("Age", "30"),
                          ("City", "NYC")], 1)
        r2 = record_card([("Name", "Bob"), ("Age", "25"),
                          ("City", "LA")], 2)
        records = VGroup(r1, r2).arrange(DOWN, buff=0.35)
        records.shift(RIGHT * 2.5 + DOWN * 1.0)

        self.play(FadeIn(r1, shift=RIGHT * 0.3))
        self.play(FadeIn(r2, shift=RIGHT * 0.3))
        self.wait(1.5)

        # Key insight callouts
        insight1 = Text("Each row → record keyed by column headers",
                        font_size=17, color=GREY_A)
        insight2 = Text("Order doesn't matter — it's a bag!",
                        font_size=17, color=TRM_COL)
        insights = VGroup(insight1, insight2).arrange(DOWN, buff=0.15)
        insights.to_edge(DOWN, buff=0.4)
        self.play(Write(insight1))
        self.play(Write(insight2))
        self.wait(2)

        self.play(FadeOut(VGroup(tbl, arr, records, idea, insights)))

        # ── Formulas ──
        step2 = Text("Match & score records", font_size=24, color=HIGHLIGHT)
        step2.next_to(sub, DOWN, buff=0.4)
        self.play(Write(step2))

        # Equation 1
        eq1_label = Text("Overall score:", font_size=18, color=GREY_B)
        eq1 = MathTex(
            r"\text{TRM}(G, P) = "
            r"\frac{\displaystyle\sum_{(g,p) \in M}"
            r"\text{RecordSim}(g, p)}{\max(|G|,\;|P|)}",
            font_size=30,
        )
        eq1[0][:3].set_color(TRM_COL)

        # Equation 2
        eq2_label = Text("Per-record similarity:", font_size=18, color=GREY_B)
        eq2 = MathTex(
            r"\text{RecordSim}(g, p) = "
            r"\frac{\displaystyle\sum_{k \in K(g) \cap K(p)}"
            r"\mathbf{1}[g[k] = p[k]]}{|K(g) \cup K(p)|}",
            font_size=28,
        )

        formulas = VGroup(
            VGroup(eq1_label, eq1).arrange(DOWN, buff=0.15),
            VGroup(eq2_label, eq2).arrange(DOWN, buff=0.15),
        ).arrange(DOWN, buff=0.5)
        formulas.next_to(step2, DOWN, buff=0.4)

        self.play(Write(eq1_label), Write(eq1), run_time=1.5)
        self.wait(1)
        self.play(Write(eq2_label), Write(eq2), run_time=1.5)
        self.wait(1)

        # Annotation
        ann = VGroup(
            Text("M = optimal matching between records", font_size=15,
                 color=GREY_B),
            Text("K(r) = set of column keys for record r", font_size=15,
                 color=GREY_B),
            Text("𝟏[·] = 1 if values match, 0 otherwise", font_size=15,
                 color=GREY_B),
        ).arrange(DOWN, buff=0.1, aligned_edge=LEFT)
        ann.to_edge(DOWN, buff=0.35)
        self.play(FadeIn(ann))
        self.wait(3)

        self.play(FadeOut(VGroup(title, sub, step2, formulas, ann)))

    # ── Scene 6 : TRM handles both problems ───────────────────
    def trm_handles_both(self):
        title = self.section_title("TRM Fixes Both Problems", color=TRM_COL)
        self.play(Write(title), run_time=0.7)

        # ── Problem 1 fix: column reordering ──
        p1_label = Text("Column Reordering", font_size=24, weight=BOLD,
                        color=HIGHLIGHT)
        p1_label.next_to(title, DOWN, buff=0.4)
        self.play(Write(p1_label))

        # Show two tables side by side, small
        gt_data = [["Name", "Age", "City"],
                   ["Alice", "30",  "NYC"]]
        pred_data = [["City", "Name", "Age"],
                     ["NYC",  "Alice", "30"]]

        gt, _, _ = self.make_table(gt_data, cw=0.9, ch=0.38, font_size=14,
                                    header_color=HEADER_FILL_2)
        pred, _, _ = self.make_table(pred_data, cw=0.9, ch=0.38, font_size=14,
                                      header_color=HEADER_FILL_2)
        gt.shift(LEFT * 4.0 + UP * 0.1)
        pred.shift(LEFT * 0.5 + UP * 0.1)
        gl = Text("GT", font_size=14, color=GREY_B)
        pl = Text("Pred", font_size=14, color=GREY_B)
        gl.next_to(gt, UP, buff=0.1)
        pl.next_to(pred, UP, buff=0.1)

        self.play(FadeIn(gt), FadeIn(pred), FadeIn(gl), FadeIn(pl))

        # Show records are the same
        rec_text = Text(
            'Both → {Name: "Alice", Age: "30", City: "NYC"}',
            font_size=16, color=TRM_COL,
        )
        rec_text.shift(UP * 0.1 + RIGHT * 3.2)

        eq_sign = Text("Same bag of records!", font_size=18, weight=BOLD,
                        color=GOOD)
        eq_sign.next_to(rec_text, DOWN, buff=0.2)

        trm_score = Text("TRM = 1.0  (perfect)", font_size=20, weight=BOLD,
                          color=TRM_COL)
        trm_score.next_to(eq_sign, DOWN, buff=0.2)

        self.play(Write(rec_text))
        self.play(Write(eq_sign))
        self.play(Write(trm_score))
        self.wait(2)

        self.play(FadeOut(VGroup(
            gt, pred, gl, pl, rec_text, eq_sign, trm_score, p1_label
        )))

        # ── Problem 2 fix: header transposition ──
        p2_label = Text("Header Transposition", font_size=24, weight=BOLD,
                        color=ERR)
        p2_label.next_to(title, DOWN, buff=0.4)
        self.play(Write(p2_label))

        gt_data2 = [["Name", "Age"],
                    ["Alice", "30"],
                    ["Bob",   "25"]]
        pred_data2 = [["Age", "Name"],
                      ["Alice", "30"],
                      ["Bob",   "25"]]

        gt2, _, _ = self.make_table(gt_data2, cw=1.0, ch=0.38, font_size=14,
                                     header_color=HEADER_FILL_2)
        pred2, _, _ = self.make_table(
            pred_data2, cw=1.0, ch=0.38, font_size=14,
            header_color=HEADER_FILL_2,
            highlight_cells={(0, 0), (0, 1)},
        )
        gt2.shift(LEFT * 4.5 + UP * 0.0)
        pred2.shift(LEFT * 1.5 + UP * 0.0)
        gl2 = Text("GT", font_size=14, color=GREY_B)
        pl2 = Text("Pred", font_size=14, color=GREY_B)
        gl2.next_to(gt2, UP, buff=0.1)
        pl2.next_to(pred2, UP, buff=0.1)

        self.play(FadeIn(gt2), FadeIn(pred2), FadeIn(gl2), FadeIn(pl2))

        # Show the records
        gt_rec = VGroup(
            Text('GT record:', font_size=15, weight=BOLD, color=GREY_A),
            Text('{Name: "Alice", Age: "30"}', font_size=14, color=GOOD),
        ).arrange(DOWN, buff=0.1, aligned_edge=LEFT)

        pred_rec = VGroup(
            Text('Pred record:', font_size=15, weight=BOLD, color=GREY_A),
            Text('{Age: "Alice", Name: "30"}', font_size=14, color=ERR),
        ).arrange(DOWN, buff=0.1, aligned_edge=LEFT)

        recs = VGroup(gt_rec, pred_rec).arrange(DOWN, buff=0.3,
                                                  aligned_edge=LEFT)
        recs.shift(RIGHT * 2.5 + UP * 0.1)

        self.play(FadeIn(gt_rec))
        self.play(FadeIn(pred_rec))
        self.wait(0.5)

        # Show the comparison
        comp = VGroup(
            Text('Name: "Alice" vs "30"  →  ✗', font_size=15, color=ERR),
            Text('Age:  "30" vs "Alice"  →  ✗', font_size=15, color=ERR),
        ).arrange(DOWN, buff=0.12, aligned_edge=LEFT)
        comp.next_to(recs, DOWN, buff=0.35)

        self.play(Write(comp[0]))
        self.play(Write(comp[1]))

        trm_bad = Text("RecordSim = 0 / 2 = 0.0", font_size=20,
                        weight=BOLD, color=ERR)
        trm_bad.next_to(comp, DOWN, buff=0.25)
        self.play(Write(trm_bad))

        trm_verdict = Text("TRM ≈ 0.0  — correctly catastrophic!",
                            font_size=20, weight=BOLD, color=TRM_COL)
        trm_verdict.to_edge(DOWN, buff=0.4)
        self.play(Write(trm_verdict))
        self.wait(3)

        self.play(FadeOut(VGroup(
            title, p2_label, gt2, pred2, gl2, pl2,
            recs, comp, trm_bad, trm_verdict,
        )))

    # ── Scene 7 : GTRM combined metric ───────────────────────
    def gtrm_combined(self):
        title = self.section_title("GTRM: Best of Both Worlds", color=GTRM_COL)
        self.play(Write(title), run_time=0.8)

        # Motivation
        motiv = Text(
            "GriTS captures structure.  TRM captures semantics.\nCombine them.",
            font_size=22, color=GREY_A, line_spacing=1.3,
        )
        motiv.next_to(title, DOWN, buff=0.5)
        self.play(Write(motiv))
        self.wait(1.5)

        # Formula
        eq = MathTex(
            r"\text{GTRM}", r"=", r"\frac{\text{GriTS} + \text{TRM}}{2}",
            font_size=44,
        )
        eq[0].set_color(GTRM_COL)
        eq[2][0:5].set_color(GRITS_COL)
        eq[2][6:9].set_color(TRM_COL)
        eq.next_to(motiv, DOWN, buff=0.6)
        self.play(Write(eq), run_time=1.5)
        self.wait(1)

        # Two components visualized
        grits_box = RoundedRectangle(width=4.5, height=2.2, corner_radius=0.15,
                                     color=GRITS_COL, fill_opacity=0.1,
                                     stroke_width=2)
        grits_label = Text("GriTS", font_size=26, weight=BOLD, color=GRITS_COL)
        grits_desc = VGroup(
            Text("Grid structure matching", font_size=15, color=GREY_A),
            Text("Row/column symmetry", font_size=15, color=GREY_A),
            Text("Spatial accuracy (IoU)", font_size=15, color=GREY_A),
        ).arrange(DOWN, buff=0.12, aligned_edge=LEFT)
        grits_inner = VGroup(grits_label, grits_desc).arrange(DOWN, buff=0.2)
        grits_inner.move_to(grits_box)

        trm_box = RoundedRectangle(width=4.5, height=2.2, corner_radius=0.15,
                                   color=TRM_COL, fill_opacity=0.1,
                                   stroke_width=2)
        trm_label = Text("TRM", font_size=26, weight=BOLD, color=TRM_COL)
        trm_desc = VGroup(
            Text("Bag-of-records matching", font_size=15, color=GREY_A),
            Text("Order insensitive", font_size=15, color=GREY_A),
            Text("Header-aware (key-value)", font_size=15, color=GREY_A),
        ).arrange(DOWN, buff=0.12, aligned_edge=LEFT)
        trm_inner = VGroup(trm_label, trm_desc).arrange(DOWN, buff=0.2)
        trm_inner.move_to(trm_box)

        boxes = VGroup(
            VGroup(grits_box, grits_inner),
            VGroup(trm_box, trm_inner),
        ).arrange(RIGHT, buff=0.5)
        boxes.to_edge(DOWN, buff=0.5)

        plus = Text("+", font_size=36, color=WHITE)
        plus.move_to(boxes.get_center())

        self.play(
            FadeIn(grits_box), FadeIn(grits_inner),
            FadeIn(trm_box), FadeIn(trm_inner),
            Write(plus),
            run_time=1.2,
        )
        self.wait(3)

        self.play(FadeOut(VGroup(title, motiv, eq, boxes, plus)))

    # ── Scene 8 : Summary ─────────────────────────────────────
    def summary(self):
        title = self.section_title("Comparison")
        self.play(Write(title), run_time=0.6)

        # Build a comparison matrix
        # Headers
        metrics = ["TEDS", "GriTS", "TRM"]
        metric_colors = [TEDS_COL, GRITS_COL, TRM_COL]
        properties = [
            "Representation",
            "Row/col symmetry",
            "Column reorder",
            "Header errors",
            "Provides P & R",
        ]
        values = [
            ["HTML tree",     "2D grid",    "Bag of records"],
            ["No",            "Yes",        "N/A (orderless)"],
            ["Heavy penalty", "Heavy penalty", "No penalty"],
            ["Mild penalty",  "Mild penalty",  "Heavy penalty"],
            ["No",            "Yes",        "Yes"],
        ]
        value_colors = [
            [GREY_A, GREY_A, GREY_A],
            [ERR,    GOOD,   GOOD],
            [ERR,    ERR,    GOOD],
            [ERR,    ERR,    GOOD],
            [ERR,    GOOD,   GOOD],
        ]

        # Build the table manually
        cw_prop = 2.6
        cw_val = 2.8
        ch = 0.5
        start_y = 1.8
        start_x = -5.5

        all_cells = VGroup()

        # Metric headers
        for j, (m, mc) in enumerate(zip(metrics, metric_colors)):
            t = Text(m, font_size=20, weight=BOLD, color=mc)
            t.move_to(np.array([start_x + cw_prop + j * cw_val + cw_val / 2,
                                start_y + ch / 2, 0]))
            all_cells.add(t)

        # Property column header
        prop_h = Text("Property", font_size=18, weight=BOLD, color=GREY_B)
        prop_h.move_to(np.array([start_x + cw_prop / 2,
                                  start_y + ch / 2, 0]))
        all_cells.add(prop_h)

        # Separator line
        sep = Line(
            np.array([start_x, start_y, 0]),
            np.array([start_x + cw_prop + 3 * cw_val, start_y, 0]),
            color=GREY_D, stroke_width=1.5,
        )
        all_cells.add(sep)

        # Rows
        rows_group = VGroup()
        for i, (prop, vals, cols) in enumerate(
                zip(properties, values, value_colors)):
            y = start_y - (i + 1) * ch
            pt = Text(prop, font_size=16, color=GREY_A)
            pt.move_to(np.array([start_x + cw_prop / 2, y, 0]))
            row_items = VGroup(pt)
            for j, (v, c) in enumerate(zip(vals, cols)):
                vt = Text(v, font_size=15, color=c)
                vt.move_to(np.array([start_x + cw_prop + j * cw_val
                                      + cw_val / 2, y, 0]))
                row_items.add(vt)
            rows_group.add(row_items)

        # Animate
        self.play(FadeIn(all_cells), run_time=0.8)
        self.play(
            LaggedStart(*[FadeIn(r, shift=RIGHT * 0.2) for r in rows_group],
                         lag_ratio=0.15),
            run_time=2.5,
        )
        self.wait(2)

        # Final takeaway
        takeaway_box = RoundedRectangle(
            width=10, height=1.2, corner_radius=0.12,
            color=GTRM_COL, fill_opacity=0.12, stroke_width=2,
        )
        takeaway_box.to_edge(DOWN, buff=0.35)

        takeaway = VGroup(
            Text("GTRM = (GriTS + TRM) / 2", font_size=22,
                 weight=BOLD, color=GTRM_COL),
            Text("Structural correctness + Semantic correctness",
                 font_size=18, color=GREY_A),
        ).arrange(DOWN, buff=0.12)
        takeaway.move_to(takeaway_box)

        self.play(FadeIn(takeaway_box), Write(takeaway))
        self.wait(3)

        self.play(FadeOut(VGroup(title, all_cells, rows_group,
                                  takeaway_box, takeaway)))

    # ── Outro ──────────────────────────────────────────────────
    def outro(self):
        t1 = Text("TEDS", font_size=36, weight=BOLD, color=TEDS_COL)
        dot1 = Text("·", font_size=36, color=GREY_B)
        t2 = Text("GriTS", font_size=36, weight=BOLD, color=GRITS_COL)
        dot2 = Text("·", font_size=36, color=GREY_B)
        t3 = Text("TRM", font_size=36, weight=BOLD, color=TRM_COL)
        row = VGroup(t1, dot1, t2, dot2, t3).arrange(RIGHT, buff=0.3)

        sub = Text("ParseBench — Zhang et al., 2025",
                    font_size=20, color=GREY_B)
        g = VGroup(row, sub).arrange(DOWN, buff=0.4)
        self.play(Write(row), run_time=1.2)
        self.play(FadeIn(sub))
        self.wait(2.5)
        self.play(FadeOut(g))
