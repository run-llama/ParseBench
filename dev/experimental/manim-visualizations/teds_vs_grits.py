from manim import *
import numpy as np

# ── Color Palette ──────────────────────────────────────────────
BG = "#1a1a2e"
TEDS_COLOR = "#4ade80"     # green
GRITS_COLOR = "#22d3ee"    # cyan/teal
HIGHLIGHT = "#facc15"      # yellow
ERR = "#f87171"            # red
ACCENT = "#fbbf24"         # gold
CELL_FILL = "#334155"
CELL_STROKE = "#94a3b8"
HEADER_FILL = "#1e3a5f"


class TEDSvsGRITS(Scene):
    """Full explainer: TEDS vs GriTS for table similarity."""

    def setup(self):
        self.camera.background_color = BG

    def construct(self):
        self.intro()
        self.problem_setup()
        self.teds_section()
        self.grits_section()
        self.asymmetry_demo()
        self.summary()

    # ── helpers ─────────────────────────────────────────────────
    def make_table(self, data, cw=1.0, ch=0.45, font_size=18,
                   header=True, highlight_cells=None):
        """Return (table_vgroup, cell_rects_flat, cell_texts_flat)."""
        rows, cols = len(data), len(data[0])
        rects = VGroup()
        texts = VGroup()
        for i in range(rows):
            for j in range(cols):
                fill = HEADER_FILL if (header and i == 0) else CELL_FILL
                opacity = 0.7 if (header and i == 0) else 0.45
                rect = Rectangle(
                    width=cw, height=ch,
                    fill_color=fill, fill_opacity=opacity,
                    stroke_color=CELL_STROKE, stroke_width=1.5,
                )
                rect.move_to(np.array([j * cw, -i * ch, 0]))
                if highlight_cells and (i, j) in highlight_cells:
                    rect.set_stroke(ERR, width=3)
                    rect.set_fill(ERR, opacity=0.25)
                t = Text(str(data[i][j]), font_size=font_size, color=WHITE)
                t.move_to(rect.get_center())
                rects.add(rect)
                texts.add(t)
        table = VGroup(rects, texts)
        table.move_to(ORIGIN)
        return table, rects, texts

    def section_title(self, text, color=WHITE):
        t = Text(text, font_size=44, weight=BOLD, color=color)
        t.to_edge(UP, buff=0.45)
        return t

    def tree_node(self, label, color, radius=0.28, font_size=13):
        c = Circle(radius=radius, color=color, fill_opacity=0.55,
                   stroke_width=2)
        t = Text(label, font_size=font_size, color=WHITE)
        t.move_to(c.get_center())
        return VGroup(c, t)

    # ── Scene 1 : Intro ───────────────────────────────────────
    def intro(self):
        title = Text("TEDS  vs  GriTS", font_size=60, weight=BOLD)
        title[0:4].set_color(TEDS_COLOR)
        title[8:13].set_color(GRITS_COLOR)
        sub = Text("Two Ways to Measure Table Similarity",
                    font_size=28, color=GREY_B)
        g = VGroup(title, sub).arrange(DOWN, buff=0.5)
        self.play(Write(title), run_time=1.2)
        self.play(FadeIn(sub, shift=UP * 0.3))
        self.wait(2)
        self.play(FadeOut(g))

    # ── Scene 2 : Problem Setup ───────────────────────────────
    def problem_setup(self):
        header = self.section_title("The Problem")
        self.play(Write(header), run_time=0.7)

        gt_data = [["Name", "Age", "City"],
                   ["Alice", "30",  "NYC"],
                   ["Bob",   "25",  "LA"]]
        pred_data = [["Name", "Age City", ""],
                     ["Alice", "30",      "NYC"],
                     ["Bob",   "25",      "LA"]]

        gt, _, _ = self.make_table(gt_data, cw=1.15)
        pr, _, _ = self.make_table(pred_data, cw=1.15)

        gt_lab = Text("Ground Truth", font_size=22, color=TEDS_COLOR)
        pr_lab = Text("Prediction", font_size=22, color=ERR)

        gt.shift(LEFT * 3.2 + DOWN * 0.4)
        pr.shift(RIGHT * 3.2 + DOWN * 0.4)
        gt_lab.next_to(gt, UP, buff=0.25)
        pr_lab.next_to(pr, UP, buff=0.25)

        self.play(FadeIn(gt), Write(gt_lab))
        self.play(FadeIn(pr), Write(pr_lab))

        arrow = Arrow(gt.get_right() + RIGHT * 0.15,
                      pr.get_left() + LEFT * 0.15,
                      color=HIGHLIGHT, stroke_width=2.5, buff=0.1)
        q = Text("How similar?", font_size=30, color=HIGHLIGHT, weight=BOLD)
        q.next_to(arrow, UP, buff=0.2)

        self.play(GrowArrow(arrow), Write(q))
        self.wait(2)
        self.play(FadeOut(VGroup(gt, pr, gt_lab, pr_lab, arrow, q, header)))

    # ── Scene 3 : TEDS ────────────────────────────────────────
    def teds_section(self):
        title = self.section_title("TEDS", color=TEDS_COLOR)
        sub = Text("Tree Edit Distance-based Similarity",
                    font_size=24, color=GREY_B)
        sub.next_to(title, DOWN, buff=0.15)
        self.play(Write(title), FadeIn(sub, shift=UP * 0.2))

        # ─ step 1: table → tree ─
        step = Text("Represent table as an HTML tree",
                     font_size=22, color=HIGHLIGHT)
        step.next_to(sub, DOWN, buff=0.35)
        self.play(Write(step))

        data = [["A", "B", "C"], ["D", "E", "F"]]
        tbl, _, _ = self.make_table(data, cw=0.75, ch=0.4, font_size=16)
        tbl.shift(LEFT * 4.2 + DOWN * 1.0)
        self.play(FadeIn(tbl))

        # build tree
        root = self.tree_node("<table>", BLUE_C)
        tr1 = self.tree_node("<tr>", TEAL_C)
        tr2 = self.tree_node("<tr>", TEAL_C)
        tds_labels = ["A", "B", "C", "D", "E", "F"]
        tds = [self.tree_node(f"{l}", GREEN_C, radius=0.24, font_size=14)
               for l in tds_labels]

        # positions
        tc = RIGHT * 1.8 + DOWN * 0.2
        root.move_to(tc + UP * 1.8)
        tr1.move_to(tc + UP * 0.4 + LEFT * 1.6)
        tr2.move_to(tc + UP * 0.4 + RIGHT * 1.6)
        for k, td in enumerate(tds[:3]):
            td.move_to(tc + DOWN * 1.1 + LEFT * (2.4 - k * 1.2))
        for k, td in enumerate(tds[3:]):
            td.move_to(tc + DOWN * 1.1 + RIGHT * (0.0 + k * 1.2))

        edges = VGroup(
            Line(root[0].get_bottom(), tr1[0].get_top(), stroke_width=2, color=GREY_B),
            Line(root[0].get_bottom(), tr2[0].get_top(), stroke_width=2, color=GREY_B),
            *[Line(tr1[0].get_bottom(), tds[i][0].get_top(), stroke_width=2, color=GREY_B)
              for i in range(3)],
            *[Line(tr2[0].get_bottom(), tds[i][0].get_top(), stroke_width=2, color=GREY_B)
              for i in range(3, 6)],
        )

        conv_arrow = Arrow(tbl.get_right() + RIGHT * 0.15,
                           tc + LEFT * 3.0, color=HIGHLIGHT,
                           stroke_width=2, buff=0.1)
        self.play(GrowArrow(conv_arrow), run_time=0.5)

        self.play(FadeIn(root))
        self.play(Create(edges[0]), Create(edges[1]),
                  FadeIn(tr1), FadeIn(tr2), run_time=0.8)
        self.play(
            LaggedStart(*[AnimationGroup(Create(edges[i + 2]), FadeIn(tds[i]))
                          for i in range(6)], lag_ratio=0.15),
            run_time=1.5)
        self.wait(0.5)

        # highlight asymmetry hint
        hint_box = SurroundingRectangle(VGroup(tr1, tr2), color=HIGHLIGHT,
                                        buff=0.15, stroke_width=2)
        hint = Text("Rows are intermediate nodes (extra cost to delete)",
                     font_size=18, color=HIGHLIGHT)
        hint.to_edge(DOWN, buff=0.5)
        self.play(Create(hint_box), Write(hint))
        self.wait(2)

        tree_all = VGroup(root, tr1, tr2, *tds, edges, conv_arrow, hint_box)
        self.play(FadeOut(VGroup(tbl, tree_all, step, hint)))

        # ─ step 2: edit operations + formula ─
        step2 = Text("Compare trees via edit distance",
                      font_size=22, color=HIGHLIGHT)
        step2.next_to(sub, DOWN, buff=0.35)
        self.play(Write(step2))

        ops = [("Delete node", "cost = 1", ERR),
               ("Insert node", "cost = 1", TEDS_COLOR),
               ("Rename node", "cost ∈ [0,1]", HIGHLIGHT)]
        boxes = VGroup()
        for name, cost, col in ops:
            box = RoundedRectangle(width=3.4, height=1.1, corner_radius=0.12,
                                   color=col, fill_opacity=0.12, stroke_width=2)
            n = Text(name, font_size=22, weight=BOLD, color=col)
            c = Text(cost, font_size=16, color=GREY_B)
            VGroup(n, c).arrange(DOWN, buff=0.12).move_to(box)
            boxes.add(VGroup(box, n, c))
        boxes.arrange(RIGHT, buff=0.35).next_to(step2, DOWN, buff=0.45)

        self.play(LaggedStart(*[FadeIn(b, shift=UP * 0.3) for b in boxes],
                               lag_ratio=0.25), run_time=1.3)
        self.wait(1)

        formula = MathTex(
            r"\text{TEDS}", r"(T_a, T_b)", r"= 1 -",
            r"\frac{\text{EditDist}(T_a, T_b)}{\max(|T_a|,\;|T_b|)}",
            font_size=34,
        )
        formula[0].set_color(TEDS_COLOR)
        formula.next_to(boxes, DOWN, buff=0.55)

        score_note = Text("1 = identical  ·  0 = completely different",
                          font_size=18, color=GREY_B)
        score_note.next_to(formula, DOWN, buff=0.3)

        self.play(Write(formula), run_time=1.5)
        self.play(FadeIn(score_note))
        self.wait(2.5)

        self.play(FadeOut(VGroup(title, sub, step2, boxes, formula, score_note)))

    # ── Scene 4 : GriTS ───────────────────────────────────────
    def grits_section(self):
        title = self.section_title("GriTS", color=GRITS_COLOR)
        sub = Text("Grid Table Similarity",
                    font_size=24, color=GREY_B)
        sub.next_to(title, DOWN, buff=0.15)
        self.play(Write(title), FadeIn(sub, shift=UP * 0.2))

        step = Text("Represent table as a flat 2D matrix",
                     font_size=22, color=HIGHLIGHT)
        step.next_to(sub, DOWN, buff=0.35)
        self.play(Write(step))

        data = [["A", "B", "C"], ["D", "E", "F"]]
        tbl, rects, _ = self.make_table(data, cw=0.75, ch=0.4, font_size=16)
        tbl.shift(LEFT * 4.2 + DOWN * 0.8)
        self.play(FadeIn(tbl))

        conv_arrow = Arrow(tbl.get_right() + RIGHT * 0.15,
                           LEFT * 0.5 + DOWN * 0.8, color=HIGHLIGHT,
                           stroke_width=2, buff=0.1)
        self.play(GrowArrow(conv_arrow), run_time=0.5)

        matrix = MathTex(
            r"M = \begin{bmatrix} A & B & C \\ D & E & F \end{bmatrix}",
            font_size=42,
        )
        matrix.shift(RIGHT * 2.0 + DOWN * 0.8)
        self.play(Write(matrix), run_time=1.2)

        sym = Text("Rows and columns are treated symmetrically!",
                    font_size=20, color=GRITS_COLOR)
        sym.next_to(matrix, DOWN, buff=0.5)
        self.play(Write(sym))
        self.wait(1.5)

        self.play(FadeOut(VGroup(tbl, conv_arrow, matrix, sym, step)))

        # formula
        step2 = Text("Find best cell alignment, then compute F1 score",
                      font_size=22, color=HIGHLIGHT)
        step2.next_to(sub, DOWN, buff=0.35)
        self.play(Write(step2))

        # Three variants
        variants = VGroup()
        for name, desc, col in [
            ("GriTS-Top", "topology (span IoU)", GRITS_COLOR),
            ("GriTS-Loc", "location (bbox IoU)", BLUE_C),
            ("GriTS-Con", "content (text sim.)", GREEN_C),
        ]:
            box = RoundedRectangle(width=3.4, height=1.0, corner_radius=0.12,
                                   color=col, fill_opacity=0.12, stroke_width=2)
            n = Text(name, font_size=20, weight=BOLD, color=col)
            d = Text(desc, font_size=15, color=GREY_B)
            VGroup(n, d).arrange(DOWN, buff=0.1).move_to(box)
            variants.add(VGroup(box, n, d))
        variants.arrange(RIGHT, buff=0.35).next_to(step2, DOWN, buff=0.45)
        self.play(LaggedStart(*[FadeIn(v, shift=UP * 0.3) for v in variants],
                               lag_ratio=0.25), run_time=1.3)
        self.wait(1)

        formula = MathTex(
            r"\text{GriTS}", r"(A, B)", r"=",
            r"\frac{2 \displaystyle\sum_{ij} f(\tilde{A}_{ij},\tilde{B}_{ij})}"
            r"{|A| + |B|}",
            font_size=32,
        )
        formula[0].set_color(GRITS_COLOR)
        formula.next_to(variants, DOWN, buff=0.5)

        f_note = Text("f  = IoU or text similarity  ·  provides precision & recall",
                       font_size=17, color=GREY_B)
        f_note.next_to(formula, DOWN, buff=0.25)

        self.play(Write(formula), run_time=1.5)
        self.play(FadeIn(f_note))
        self.wait(2.5)

        self.play(FadeOut(VGroup(title, sub, step2, variants, formula, f_note)))

    # ── Scene 5 : Asymmetry Demo (centerpiece) ────────────────
    def asymmetry_demo(self):
        title = self.section_title("The Key Difference", color=HIGHLIGHT)
        self.play(Write(title))

        sub = Text("Delete one row  vs  one column from a 4×4 table",
                    font_size=22, color=GREY_B)
        sub.next_to(title, DOWN, buff=0.2)
        self.play(FadeIn(sub))

        cw, ch, fs = 0.65, 0.38, 14

        # Ground truth 4×4
        gt_data = [[f"{i*4+j+1}" for j in range(4)] for i in range(4)]
        gt, gt_r, gt_t = self.make_table(gt_data, cw=cw, ch=ch,
                                          font_size=fs, header=False)
        gt_lab = Text("Ground Truth  (4×4)", font_size=18, color=WHITE,
                       weight=BOLD)
        gt.move_to(UP * 0.7)
        gt_lab.next_to(gt, UP, buff=0.2)

        self.play(FadeIn(gt), Write(gt_lab))
        self.wait(1)

        # Shrink GT up
        self.play(gt.animate.scale(0.75).move_to(UP * 2.6),
                  gt_lab.animate.scale(0.75).move_to(UP * 3.35))

        # ─ Prediction A: missing last row (3×4) ─
        pa_data = [[f"{i*4+j+1}" for j in range(4)] for i in range(3)]
        pa, pa_r, _ = self.make_table(pa_data, cw=cw, ch=ch,
                                       font_size=fs, header=False)
        pa_lab = Text("Missing Row  (3×4)", font_size=16, color=ERR)

        # ─ Prediction B: missing last col (4×3) ─
        pb_data = [[f"{i*4+j+1}" for j in range(3)] for i in range(4)]
        pb, pb_r, _ = self.make_table(pb_data, cw=cw, ch=ch,
                                       font_size=fs, header=False)
        pb_lab = Text("Missing Column  (4×3)", font_size=16, color=ERR)

        pa.move_to(LEFT * 3.5 + DOWN * 0.0)
        pb.move_to(RIGHT * 3.5 + DOWN * 0.0)
        pa_lab.next_to(pa, UP, buff=0.2)
        pb_lab.next_to(pb, UP, buff=0.2)

        # draw red X through missing data
        pa_cross = Cross(stroke_color=ERR, stroke_width=3).scale(0.25)
        pa_cross.next_to(pa, DOWN, buff=0.08)
        pa_miss = Text("−4 cells", font_size=14, color=ERR)
        pa_miss.next_to(pa_cross, DOWN, buff=0.08)

        pb_cross = Cross(stroke_color=ERR, stroke_width=3).scale(0.25)
        pb_cross.next_to(pb, RIGHT, buff=0.08)
        pb_miss = Text("−4 cells", font_size=14, color=ERR)
        pb_miss.next_to(pb_cross, RIGHT, buff=0.08)

        self.play(FadeIn(pa), Write(pa_lab), FadeIn(pb), Write(pb_lab))
        self.play(FadeIn(pa_cross), Write(pa_miss),
                  FadeIn(pb_cross), Write(pb_miss))
        self.wait(1.5)

        # ─ TEDS calculation ─
        teds_header = Text("TEDS", font_size=28, weight=BOLD, color=TEDS_COLOR)
        teds_header.move_to(DOWN * 1.6)
        self.play(Write(teds_header))

        # Row delete cost explanation
        # GT tree: 1 <table> + 4 <tr> + 16 <td> = 21 nodes
        # Delete row: 1 <tr> + 4 <td> = 5 edits
        # Delete col: 4 <td> = 4 edits
        teds_row_calc = VGroup(
            Text("Delete: 1 <tr> + 4 <td>", font_size=14, color=GREY_B),
            Text("= 5 edits", font_size=14, color=ERR),
        ).arrange(RIGHT, buff=0.15)
        teds_row_score = MathTex(
            r"\text{TEDS} = 1 - \tfrac{5}{21} \approx",
            font_size=28,
        )
        teds_row_val = Text("0.762", font_size=28, weight=BOLD, color=TEDS_COLOR)
        teds_row = VGroup(teds_row_calc,
                          VGroup(teds_row_score, teds_row_val).arrange(RIGHT, buff=0.1)
                          ).arrange(DOWN, buff=0.15)
        teds_row.move_to(LEFT * 3.5 + DOWN * 2.3)

        teds_col_calc = VGroup(
            Text("Delete: 4 <td>", font_size=14, color=GREY_B),
            Text("= 4 edits", font_size=14, color=ERR),
        ).arrange(RIGHT, buff=0.15)
        teds_col_score = MathTex(
            r"\text{TEDS} = 1 - \tfrac{4}{21} \approx",
            font_size=28,
        )
        teds_col_val = Text("0.810", font_size=28, weight=BOLD, color=TEDS_COLOR)
        teds_col = VGroup(teds_col_calc,
                          VGroup(teds_col_score, teds_col_val).arrange(RIGHT, buff=0.1)
                          ).arrange(DOWN, buff=0.15)
        teds_col.move_to(RIGHT * 3.5 + DOWN * 2.3)

        self.play(Write(teds_row_calc), Write(teds_col_calc))
        self.play(Write(teds_row_score), Write(teds_row_val),
                  Write(teds_col_score), Write(teds_col_val))
        self.wait(1)

        # Highlight the difference
        neq = MathTex(r"\neq", font_size=40, color=ERR)
        neq.move_to(DOWN * 2.35)
        self.play(Write(neq))
        self.wait(1.5)

        problem_note = Text(
            "Same severity, different scores! Row deletion penalized more.",
            font_size=18, color=ERR,
        )
        problem_note.to_edge(DOWN, buff=0.3)
        self.play(Write(problem_note))
        self.wait(2)

        # ─ now show GriTS ─
        self.play(FadeOut(VGroup(
            teds_header, teds_row, teds_col, neq, problem_note
        )))

        grits_header = Text("GriTS", font_size=28, weight=BOLD, color=GRITS_COLOR)
        grits_header.move_to(DOWN * 1.6)
        self.play(Write(grits_header))

        # GriTS: GT=16 cells, pred=12 cells both times
        # GriTS = 2*12/(16+12) = 24/28 ≈ 0.857
        grits_row_score = MathTex(
            r"\text{GriTS} = \tfrac{2 \times 12}{16 + 12} \approx",
            font_size=28,
        )
        grits_row_val = Text("0.857", font_size=28, weight=BOLD, color=GRITS_COLOR)
        grits_row = VGroup(grits_row_score, grits_row_val).arrange(RIGHT, buff=0.1)
        grits_row.move_to(LEFT * 3.5 + DOWN * 2.3)

        grits_col_score = MathTex(
            r"\text{GriTS} = \tfrac{2 \times 12}{16 + 12} \approx",
            font_size=28,
        )
        grits_col_val = Text("0.857", font_size=28, weight=BOLD, color=GRITS_COLOR)
        grits_col = VGroup(grits_col_score, grits_col_val).arrange(RIGHT, buff=0.1)
        grits_col.move_to(RIGHT * 3.5 + DOWN * 2.3)

        self.play(Write(grits_row_score), Write(grits_row_val),
                  Write(grits_col_score), Write(grits_col_val))
        self.wait(0.5)

        eq = MathTex(r"=", font_size=40, color=GRITS_COLOR)
        eq.move_to(DOWN * 2.35)
        self.play(Write(eq))

        ok_note = Text(
            "Same severity, same score. Rows and columns treated equally.",
            font_size=18, color=GRITS_COLOR,
        )
        ok_note.to_edge(DOWN, buff=0.3)
        self.play(Write(ok_note))
        self.wait(3)

        self.play(FadeOut(VGroup(
            title, sub, gt, gt_lab,
            pa, pa_lab, pb, pb_lab,
            pa_cross, pa_miss, pb_cross, pb_miss,
            grits_header, grits_row, grits_col, eq, ok_note,
        )))

    # ── Scene 6 : Summary ─────────────────────────────────────
    def summary(self):
        title = self.section_title("Summary")
        self.play(Write(title))

        # Two-column comparison
        col_w = 5.5
        teds_box = RoundedRectangle(
            width=col_w, height=4.5, corner_radius=0.15,
            color=TEDS_COLOR, fill_opacity=0.08, stroke_width=2)
        grits_box = RoundedRectangle(
            width=col_w, height=4.5, corner_radius=0.15,
            color=GRITS_COLOR, fill_opacity=0.08, stroke_width=2)

        teds_box.shift(LEFT * 3.2 + DOWN * 0.6)
        grits_box.shift(RIGHT * 3.2 + DOWN * 0.6)

        teds_title = Text("TEDS", font_size=30, weight=BOLD, color=TEDS_COLOR)
        grits_title = Text("GriTS", font_size=30, weight=BOLD, color=GRITS_COLOR)
        teds_title.move_to(teds_box.get_top() + DOWN * 0.35)
        grits_title.move_to(grits_box.get_top() + DOWN * 0.35)

        def bullet_list(items, box, color):
            texts = VGroup()
            for item in items:
                t = Text(item, font_size=16, color=GREY_A)
                texts.add(t)
            texts.arrange(DOWN, buff=0.22, aligned_edge=LEFT)
            texts.next_to(box.get_top() + DOWN * 0.8, DOWN, buff=0.1)
            texts.shift(LEFT * 0.3)
            return texts

        teds_items = [
            "  HTML tree representation",
            "  Tree edit distance algorithm",
            "  Rows ≠ columns (asymmetric)",
            "  Row errors penalized more",
            "  Well-established (ICDAR)",
            "  Single similarity score",
        ]
        teds_bullets = bullet_list(teds_items, teds_box, TEDS_COLOR)
        # Color the problematic items
        teds_bullets[2].set_color(ERR)
        teds_bullets[3].set_color(ERR)

        grits_items = [
            "  2D grid / matrix representation",
            "  Optimal substructure matching",
            "  Rows = columns (symmetric)",
            "  Equal treatment of all errors",
            "  Newer (ICDAR 2023)",
            "  Precision + Recall + F1",
        ]
        grits_bullets = bullet_list(grits_items, grits_box, GRITS_COLOR)
        grits_bullets[2].set_color(GRITS_COLOR)
        grits_bullets[3].set_color(GRITS_COLOR)

        self.play(FadeIn(teds_box), FadeIn(grits_box),
                  Write(teds_title), Write(grits_title))
        self.play(
            LaggedStart(*[FadeIn(t, shift=RIGHT * 0.2) for t in teds_bullets],
                         lag_ratio=0.15),
            LaggedStart(*[FadeIn(t, shift=LEFT * 0.2) for t in grits_bullets],
                         lag_ratio=0.15),
            run_time=2.5,
        )
        self.wait(3)

        # Final takeaway
        takeaway = Text(
            "GriTS fixes TEDS's row-column asymmetry\n"
            "and adds precision / recall breakdown.",
            font_size=22, color=HIGHLIGHT, line_spacing=1.4,
        )
        takeaway.to_edge(DOWN, buff=0.35)
        self.play(Write(takeaway), run_time=1.5)
        self.wait(3)

        self.play(FadeOut(VGroup(
            title, teds_box, grits_box, teds_title, grits_title,
            teds_bullets, grits_bullets, takeaway,
        )))

        # End card
        thanks = Text("TEDS vs GriTS", font_size=48, weight=BOLD)
        thanks[0:4].set_color(TEDS_COLOR)
        thanks[7:12].set_color(GRITS_COLOR)
        self.play(Write(thanks))
        self.wait(2)
        self.play(FadeOut(thanks))
