"""We format the manuscript tables in Word while retaining the underlying CSVs."""

import csv
from pathlib import Path

from docx import Document
from docx.shared import Inches, Pt

TABLES = Path(__file__).resolve().parent.parent / "results" / "tables"


def write_table(destination, title, header, rows):
    doc = Document()
    section = doc.sections[0]
    section.left_margin = section.right_margin = Inches(0.65)
    for style_name in ("Normal", "Title"):
        style = doc.styles[style_name]
        style.font.name = "Times New Roman"
        style.font.size = Pt(10 if style_name == "Normal" else 14)
    doc.add_paragraph(title, style="Title")
    table = doc.add_table(rows=1, cols=len(header))
    table.style = "Table Grid"
    for index, row in enumerate([header, *rows]):
        cells = table.rows[0].cells if index == 0 else table.add_row().cells
        if len(row) == 1:
            cells = [cells[0].merge(cells[-1])]
        for cell, value in zip(cells, row):
            run = cell.paragraphs[0].add_run(value)
            run.font.name = "Times New Roman"
            run.bold = index == 0 or len(row) == 1
    doc.save(destination)


def read_model_rows(filename):
    with (TABLES.parent / "main_analysis" / filename).open(encoding="utf-8") as stream:
        return {
            (row["sample"], row["mediator"].split()[0]): row
            for row in csv.DictReader(stream)
            if row["model"] == "single"
        }


def estimate(row, key, digits=2):
    value_key = "indirect_ab" if not key else f"{key}beta"
    value = float(row[value_key])
    lo, hi = (float(row[f"{key}ci_{bound}"]) for bound in ("lo", "hi"))
    return f"{value:.{digits}f} [{lo:.2f}, {hi:.2f}]".replace("-", "−")


def supplementary_rows():
    # We use the primary results directly so S1 follows manuscript rounding.
    paths = read_model_rows("table_aim2_paths.csv")
    indirect = read_model_rows("table_aim2_indirect.csv")
    samples = sorted({sample for sample, _ in paths}, key=lambda sample: not sample.startswith("Full"))
    rows = [
        ["Aim 1: Total effect (screen time → EAH)"],
        ["Total effect (c)", *(estimate(paths[sample, "FF"], "c_") for sample in samples)],
        ["Aim 2: Attentional bias to food cues (food − toy) as mediator"],
    ]
    for code, label in (("FF", "First-fixation bias"), ("CGD", "Cumulative-gaze bias")):
        rows.append([label])
        for key, title in (("a_", "a: screen time → bias"), ("b_", "b: bias → EAH"),
                           ("cp_", "c′: direct effect")):
            rows.append([title, *(estimate(paths[sample, code], key) for sample in samples)])
        rows.append(["Indirect effect (a×b)", *(estimate(indirect[sample, code], "", 3) for sample in samples)])
        rows.append(["% mediated", *(f'{float(indirect[sample, code]["prop_mediated_pct"]):.0f}%' for sample in samples)])
    header = ["Path / effect", *(sample.replace("n=", "n = ").replace("Consumers only", "Consumers") for sample in samples)]
    return header, rows


if __name__ == "__main__":
    with (TABLES / "table1_detailed.csv").open(encoding="utf-8", newline="") as stream:
        header, *rows = csv.reader(stream)
    write_table(TABLES / "table1_detailed.docx", "Table 1. Sample characteristics", header, rows)
    write_table(TABLES / "table_s1_mediation.docx", "Supplementary Table S1. Mediation estimates",
                *supplementary_rows())
