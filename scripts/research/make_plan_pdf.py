"""Build the FTMO deployment plan as a PDF.

A document meant to be printed and worked through, not read once. The numbers
come from docs/findings/propfirm-sizing.md and the studies behind it; this
script only lays them out.

    python scripts/research/make_plan_pdf.py
    python scripts/research/make_plan_pdf.py -o somewhere/else.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate, Frame, KeepTogether, PageBreak, PageTemplate, Paragraph,
    Spacer, Table, TableStyle,
)

# --- palette ---------------------------------------------------------------
INK = colors.HexColor("#171A1F")
SLATE = colors.HexColor("#5F6976")
FAINT = colors.HexColor("#8A94A2")
RULE = colors.HexColor("#D6DAE0")
RULE_HARD = colors.HexColor("#B4BCC6")
GOOD = colors.HexColor("#2F6F62")
GOOD_BG = colors.HexColor("#E9F1EF")
WARN = colors.HexColor("#A0742A")
WARN_BG = colors.HexColor("#F7F1E4")
BAD = colors.HexColor("#9B4A33")
BAD_BG = colors.HexColor("#F5EBE8")

MARGIN = 20 * mm
TITLE = "FTMO Two-Step: Deployment Plan"
SUBTITLE = "USTEC risk-managed long, sized for a funded-account challenge"


def styles():
    ss = getSampleStyleSheet()
    def add(name, **kw):
        ss.add(ParagraphStyle(name=name, **kw))
    add("H1", fontName="Helvetica-Bold", fontSize=19, leading=23,
        textColor=INK, spaceBefore=0, spaceAfter=3)
    add("Deck", fontName="Helvetica", fontSize=10.5, leading=15,
        textColor=SLATE, spaceAfter=14)
    add("H2", fontName="Helvetica-Bold", fontSize=12.5, leading=16,
        textColor=INK, spaceBefore=13, spaceAfter=4)
    add("Eyebrow", fontName="Helvetica-Bold", fontSize=7.5, leading=10,
        textColor=FAINT, spaceBefore=14, spaceAfter=2)
    add("Body", fontName="Helvetica", fontSize=9.0, leading=12.7,
        textColor=INK, alignment=TA_LEFT, spaceAfter=6)
    add("Small", fontName="Helvetica", fontSize=8.4, leading=12,
        textColor=SLATE, spaceAfter=5)
    add("Mono", fontName="Courier", fontSize=8.6, leading=12.6, textColor=INK)
    add("Cell", fontName="Helvetica", fontSize=8.4, leading=11.1, textColor=INK)
    add("CellS", fontName="Helvetica", fontSize=8.6, leading=11.6, textColor=SLATE)
    add("CellB", fontName="Helvetica-Bold", fontSize=8.6, leading=11.6, textColor=INK)
    add("Callout", fontName="Helvetica", fontSize=9.2, leading=13.4, textColor=INK,
        leftIndent=8, spaceAfter=4, spaceBefore=4)
    return ss


S = styles()
P = lambda t, s="Body": Paragraph(t, S[s])


def rule(color=RULE, thickness=0.6, space=6):
    t = Table([[""]], colWidths=[170 * mm], rowHeights=[thickness])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), color),
                           ("TOPPADDING", (0, 0), (-1, -1), 0),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))
    return [Spacer(1, space), t, Spacer(1, space)]


def data_table(rows, widths, header=True, highlight=None, align_right=None):
    """A table with the project's house look: hairlines, no boxes."""
    align_right = align_right or []
    body = [[Paragraph(c, S["CellB" if (header and r == 0) else "Cell"])
             if not isinstance(c, Paragraph) else c
             for c in row] for r, row in enumerate(rows)]
    t = Table(body, colWidths=widths, hAlign="LEFT")
    style = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3.8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.8),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, RULE),
    ]
    if header:
        style += [("LINEBELOW", (0, 0), (-1, 0), 0.9, RULE_HARD),
                  ("TEXTCOLOR", (0, 0), (-1, 0), FAINT)]
    for col in align_right:
        style.append(("ALIGN", (col, 0), (col, -1), "RIGHT"))
    if highlight:
        for r, bg in highlight:
            style.append(("BACKGROUND", (0, r), (-1, r), bg))
    t.setStyle(TableStyle(style))
    return t


def callout(title, body, tone="warn"):
    bg, fg = {"warn": (WARN_BG, WARN), "bad": (BAD_BG, BAD),
              "good": (GOOD_BG, GOOD)}[tone]
    inner = [Paragraph(f'<font color="{fg.hexval()}"><b>{title}</b></font>', S["Callout"]),
             Paragraph(body, S["Callout"])]
    t = Table([[inner]], colWidths=[170 * mm], hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("LINEBEFORE", (0, 0), (0, -1), 2.2, fg),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    return t


def checklist(items):
    rows = []
    for txt in items:
        box = Table([[""]], colWidths=[3.6 * mm], rowHeights=[3.6 * mm])
        box.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.8, RULE_HARD)]))
        rows.append([box, Paragraph(txt, S["Cell"])])
    t = Table(rows, colWidths=[8 * mm, 162 * mm], hAlign="LEFT")
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, RULE),
    ]))
    return t


def header_footer(canvas, doc):
    canvas.saveState()
    w, h = A4
    canvas.setFont("Helvetica", 7.2)
    canvas.setFillColor(FAINT)
    if doc.page > 1:
        canvas.drawString(MARGIN, h - MARGIN + 6 * mm, TITLE.upper())
    canvas.drawRightString(w - MARGIN, MARGIN - 8 * mm, f"{doc.page}")
    canvas.drawString(MARGIN, MARGIN - 8 * mm,
                      "QuantProject  |  not investment advice  |  "
                      "figures from docs/findings/propfirm-sizing.md")
    canvas.restoreState()


def build(out: Path) -> None:
    doc = BaseDocTemplate(str(out), pagesize=A4,
                          leftMargin=MARGIN, rightMargin=MARGIN,
                          topMargin=MARGIN, bottomMargin=MARGIN + 4 * mm,
                          title=TITLE, author="QuantProject")
    frame = Frame(MARGIN, MARGIN + 4 * mm, A4[0] - 2 * MARGIN,
                  A4[1] - 2 * MARGIN - 4 * mm, id="main",
                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    doc.addPageTemplates([PageTemplate(id="all", frames=[frame],
                                       onPage=header_footer)])
    st = []

    # ---------------- page 1: the decision --------------------------------
    st += [P(TITLE, "H1"), P(SUBTITLE, "Deck")]
    st += [callout(
        "Read this first",
        "A three-month median to funded is not achievable with this strategy. "
        "The arithmetic is on the last page. The fastest defensible plan below "
        "has a median of <b>7 to 23 months</b>, and it gets there by accepting "
        "roughly three paid attempts rather than one. Nothing here is investment "
        "advice, and the strategy's edge is not statistically significant &mdash; "
        "see <i>Risks</i>.", "bad")]

    st += [P("PICK ONE", "Eyebrow"), P("The two configurations", "H2")]
    st += [P(
        "Both run the same strategy. They differ only in size and in whether a "
        "daily circuit breaker is armed. <b>Fast is recommended if speed matters "
        "to you</b>: it reaches funded sooner on every split, and the extra fees "
        "are about &euro;1,100 across the whole journey.", "Body")]

    rows = [
        ["", "STEADY", "FAST  (recommended for speed)"],
        ["Position scale", "0.50x cushion", "3.50x cushion"],
        ["Daily circuit breaker", "off (optional)", "on, at &minus;3.0%"],
        ["Pass rate per attempt", "99.2 &ndash; 100%", "31 &ndash; 60%"],
        ["Median time to funded", "25 &ndash; 43 months", "7 &ndash; 23 months"],
        ["Expected paid attempts", "~1", "~3"],
        ["Expected fees (100k)", "&euro;539", "~&euro;1,633"],
        ["Worst failure mode", "none observed", "total loss, 11 &ndash; 24%"],
    ]
    st += [data_table(rows, [46 * mm, 52 * mm, 72 * mm],
                      highlight=[(0, colors.white)])]
    st += [Spacer(1, 5)]
    st += [P("Ranges span the three test periods (2020&ndash;23, 2024&ndash;25H1, "
             "2025H2&ndash;26). Take the pessimistic end as the plan and the "
             "optimistic end as luck.", "Small")]

    st += [P("STEP 1", "Eyebrow"), P("Expert settings", "H2")]
    st += [P("Attach <font face='Courier'>USTEC_RiskManagedLong.mq5</font> to a "
             "USTEC chart, any timeframe. It reads the bars it needs itself.", "Body")]
    mono = (
        "InpPropMode       = true       // cushion sizing on\n"
        "InpPropScale      = 3.50       // 0.50 for the Steady plan\n"
        "InpDailyStopPct   = 3.0        // 0.0 for Steady; 3.0 above 1.0x\n"
        "InpMaxWeight      = 1.0        // down from the shipped 2.0 - matters\n"
        "InpPropCushionCap = 4.0        // 3.0 for Steady\n"
        "InpPropInitial    = 0          // 0 latches the balance at attach\n"
        "InpPropMaxLossPct = 10.0       // FTMO total loss limit\n"
        "InpTargetVolPct   = 15.0       // unchanged from the study\n"
        "InpSwapCheckBps   = 2.5        // refuses to trade above this swap"
    )
    box = Table([[Paragraph(mono.replace("\n", "<br/>").replace(" ", "&nbsp;"),
                            S["Mono"])]], colWidths=[170 * mm], hAlign="LEFT")
    box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F4F5F7")),
        ("BOX", (0, 0), (-1, -1), 0.5, RULE),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
    ]))
    st += [box, Spacer(1, 6)]
    st += [P("<b>Why the leverage cap comes down.</b> A challenge watches "
             "floating equity, not closing balances. At the shipped 2.0x cap the "
             "worst single-day floating excursion was &minus;5.07%, &minus;5.19% "
             "and &minus;7.61% on the three periods &mdash; each of those breaches "
             "the 5% daily floor on its own. At 1.0x they are &minus;3.67%, "
             "&minus;4.49% and &minus;5.33%.", "Body")]

    # ---------------- page 2: before you pay -------------------------------
    st += [PageBreak()]
    st += [P("STEP 2", "Eyebrow"), P("Before you pay for anything", "H2")]
    st += [P("Every item here is cheaper to check now than to discover during a "
             "paid challenge.", "Body")]
    st += [checklist([
        "Run the expert on an FTMO <b>free trial</b> or demo for one full month.",
        "Confirm the log's <i>effective weight</i> (strategy target &times; cushion "
        "multiple) matches <font face='Courier'>scripts/research/propfirm_eval.py</font>. "
        "A sizing rule is easy to get subtly wrong.",
        "Confirm the daily breaker actually fires: force it once on demo by "
        "setting <font face='Courier'>InpDailyStopPct</font> very low for a day.",
        "Read your account's real USTEC <b>swap</b>. Break-even is about 5 bps a "
        "night; above roughly 3 the case weakens sharply.",
        "Check the contract size and minimum lot support your account size. At a "
        "0.01 minimum lot and ~23,000 index level the smallest position is about "
        "$230 of notional.",
        "Confirm FTMO's current terms match the assumptions: 10% then 5% targets, "
        "5% daily, 10% static total, 4-day minimum, <b>no time limit</b>.",
        "Decide your stop-trying rule now, in writing, and put a number on it.",
    ])]

    st += [P("STEP 3", "Eyebrow"), P("What a normal run looks like", "H2")]
    rows = [
        ["Phase", "What happens", "Your job"],
        ["Days 1&ndash;4", "Minimum trading days served. Position sized by the "
         "cushion, which starts at full.", "Nothing. Do not intervene."],
        ["Weeks 1&ndash;8", "Equity drifts. Cushion sizing shrinks the position "
         "after losses and grows it after gains.",
         "Check the log daily. Confirm one decision per session."],
        ["A losing streak", "The position gets smaller by design. This is the "
         "rule working, not failing.",
         "Do not increase size to recover. That is the one action that "
         "converts a slow run into a failed one."],
        ["Breaker fires", "Flat for the rest of the day at about &minus;3.2%. "
         "Re-enters tomorrow.", "Nothing. Log it."],
        ["Target hit", "Phase passes. Verification starts from a fresh balance "
         "with its own floors.", "Re-latch: restart the expert so "
         "<font face='Courier'>InpPropInitial</font> takes the new balance."],
    ]
    st += [data_table(rows, [30 * mm, 78 * mm, 62 * mm])]

    st += [callout(
        "The single most important operational rule",
        "If the account is down and the position looks small, that is the cushion "
        "rule doing exactly what it is for. Increasing size to make the target "
        "back is how this plan fails. The sizing is the strategy.", "warn")]

    # ---------------- page 3: risks and the arithmetic ---------------------
    # No forced break here: letting 'When to stop' flow keeps the
    # closing pages full rather than leaving two of them half empty.
    st += [Spacer(1, 10)]
    st += [P("STEP 4", "Eyebrow"), P("When to stop", "H2")]
    st += [P("Decide these before starting. Written in advance they are "
             "discipline; written afterwards they are excuses.", "Body")]
    rows = [
        ["Trigger", "What it means", "Action"],
        ["4 failed attempts", "Past the 90th percentile of the three attempts "
         "expected. Something differs from the backtest.",
         "Stop. Re-run the study on current data before paying again."],
        ["Two breaches of the daily floor in one attempt",
         "Either the breaker is misconfigured or volatility is far above the "
         "sample.", "Stop that attempt's sizing. Halve the scale."],
        ["Live weight differs from the script",
         "An implementation bug, not a market outcome.",
         "Stop immediately and reconcile."],
        ["Swap above ~3 bps a night", "Financing is eating the edge.",
         "Do not start. The expert refuses on its own."],
        ["18 months, no funded account", "The timeline assumption has failed.",
         "Stop and reassess rather than continuing on sunk cost."],
    ]
    st += [data_table(rows, [42 * mm, 68 * mm, 60 * mm])]

    st += [P("HONEST", "Eyebrow"), P("Risks, in the order they matter", "H2")]
    st += [P(
        "<b>1. The edge may not be real.</b> The strategy's mean session return "
        "has a t-statistic of 1.87, 1.10 and 1.22 across the three periods: "
        "positive everywhere, significant nowhere. Re-running the whole "
        "simulation with the drift removed and everything else kept, cushion "
        "sizing <b>almost never fails &mdash; it just never finishes</b>: 12&ndash;22% "
        "of runs complete within twelve years, the rest grind sideways above the "
        "floor. So a high pass probability here is <b>not</b> evidence the "
        "strategy works. It is what the sizing rule does to any series with "
        "roughly this risk.", "Body")]
    st += [P(
        "<b>2. The worst day in the sample is not the worst day there is.</b> "
        "The daily floor is the one rule cushion sizing cannot protect, which is "
        "why the breaker exists. It still assumes you can get out.", "Body")]
    st += [P(
        "<b>3. Swap is not in any backtest number.</b> A quote feed carries no "
        "financing. Break-even is about 5 bps a night and it comes straight off "
        "a timeline already measured in months.", "Body")]
    st += [P(
        "<b>4. The rules can change.</b> These are FTMO's 2025 terms. A firm that "
        "reintroduces a 30-day limit inverts the entire conclusion.", "Body")]
    st += [P(
        "<b>5. A funded account is a contract, not capital.</b> Nothing in this "
        "plan addresses that, and it is not a small consideration.", "Body")]

    # Only the heading and its table need to travel together; the closing
    # paragraphs can flow, which keeps the document to three pages instead of
    # leaving two of them half empty.
    appendix_head = rule() + [
        P("APPENDIX", "Eyebrow"),
        P("Why three months is not on the menu", "H2"),
        P("Passing +10% then +5% is 15.5% compounded. For that to be the "
          "<i>median</i> inside 63 trading days the strategy must run near 62% a "
          "year. The daily floor then caps volatility: a 5% daily loss has to be "
          "rare, so 5% must sit several standard deviations out.", "Body"),
        data_table([
            ["5% sits at", "Annual sigma", "Sharpe needed", "P(a 5% day) over 63 days"],
            ["2.0 sigma", "39.7%", "1.56", "76.5%"],
            ["2.5 sigma", "31.7%", "1.95", "32.5%"],
            ["3.0 sigma", "26.5%", "2.34", "8.2%"],
            ["3.5 sigma", "22.7%", "2.73", "1.5%"],
        ], [30 * mm, 34 * mm, 34 * mm, 50 * mm], align_right=[1, 2, 3]),
    ]
    appendix_tail = [
        Spacer(1, 4),
        P("<b>This strategy runs at a Sharpe of 0.95.</b> Scaling cannot close "
          "the gap, because it multiplies drift and volatility together and "
          "leaves the ratio exactly where it started. Size trades time against "
          "probability <i>along</i> a frontier the Sharpe fixes; it cannot move "
          "the frontier. Reaching three months needs a Sharpe near 2.0 &mdash; "
          "realistically four to five uncorrelated strategies rather than one, "
          "since Sharpe scales with the square root of the number of independent "
          "return streams.", "Body"),
        P("Buying several challenges in parallel does not substitute. The same "
          "strategy on N accounts produces N perfectly correlated outcomes: they "
          "all pass together or all fail together.", "Small"),
    ]
    st += [KeepTogether(appendix_head + appendix_tail)]

    doc.build(st)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out", type=Path,
                    default=Path("reports/strategies/FTMO_DEPLOYMENT_PLAN.pdf"))
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    build(args.out)
    print(f"wrote {args.out}  ({args.out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
