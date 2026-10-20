#!/usr/bin/env python3
"""
Merconez | SpikeNav | PolarFire FPGA Design Contest 2026 (Track 2)
System block diagram generator (6 pages, landscape US Letter).

USAGE
    pip install reportlab
    python make_block_diagram.py            # writes the PDF named in CONFIG["output"]
    python make_block_diagram.py --grid     # same, plus a coordinate grid to help you position things

HOW THIS FILE IS ORGANISED
    1. CONFIG        - team name, project name, file name, footer. Edit these first.
    2. STYLE         - colours and fonts.
    3. PRIMITIVES    - box(), region(), arrow(), table(), para(), connector() helpers.
    4. PAGE 1..6     - one function per page. Every element is placed with explicit
                       coordinates (x, y, width, height) in points.

COORDINATES
    Page size is 792 x 612 points (11 x 8.5 in, 72 points per inch).
    (0, 0) is the BOTTOM-LEFT corner. x grows to the right, y grows UPWARDS.
    box(c, x, y, w, h, ...) draws a box whose bottom-left corner is at (x, y).
    Run with --grid to overlay labelled gridlines every 20 pt.

SAFETY NETS
    box() and table() print a WARNING to the terminal if any line of text is wider
    than its box or column, so you can shorten the text or widen the box.
"""

import math
import sys

from reportlab.lib.colors import HexColor, white
from reportlab.lib.pagesizes import landscape, letter
from reportlab.pdfgen import canvas

# =====================================================================================
# 1. CONFIG
# =====================================================================================
CONFIG = {
    "team": "Merconez",
    "project": "SpikeNav",
    "tagline": "Real-time Spiking Place Recognition for GPS-denied Robots",
    "track": "Track 2: PolarFire SoC Icicle Kit (MPFS250T)",
    "output": "Merconez_Track2_SystemBlockDiagram.pdf",
    "footer": "Merconez | SpikeNav | Track 2 | [M] measured on DGX Spark / Jetson Orin Nano  [D] derived  [E] estimate",
    # Public repository with all code, baselines, results and reports (linked on every page).
    "repo": "https://github.com/Bejaine/Event-Driven-Spiking-Place-Recognition-for-GPS-Denied-Navigation-on-FPGA",
    "repo_short": "github.com/Bejaine/Event-Driven-Spiking-Place-Recognition-for-GPS-Denied-Navigation-on-FPGA",
}

# =====================================================================================
# 2. STYLE
# =====================================================================================
W, H = landscape(letter)  # 792 x 612
C = {
    "ink": HexColor("#1F2933"), "mute": HexColor("#52606D"), "line": HexColor("#3E4C59"),
    "fab": HexColor("#DCEBFA"), "fabE": HexColor("#2F6FB3"),     # FPGA fabric
    "mss": HexColor("#DDF3E4"), "mssE": HexColor("#2E8B57"),     # RISC-V software
    "ext": HexColor("#EEF0F2"), "extE": HexColor("#7B8794"),     # external / board
    "mem": HexColor("#FDF0D5"), "memE": HexColor("#C08A1E"),     # memory
    "chip": HexColor("#FAFBFC"), "chipE": HexColor("#1F2933"),   # neutral panels
    "jet": HexColor("#EDE7F6"), "jetE": HexColor("#6B46C1"),     # Jetson Orin Nano baseline
    "hdr": HexColor("#102A43"), "stripe": HexColor("#F2F5F8"),
}
FONT, FONT_B, FONT_I = "Helvetica", "Helvetica-Bold", "Helvetica-Oblique"
PAGES = 6
SHOW_GRID = "--grid" in sys.argv


def warn(msg):
    print("WARNING:", msg)


# =====================================================================================
# 3. PRIMITIVES
# =====================================================================================
def header(c, title, subtitle, page):
    """Dark title band at the top of every page, plus footer."""
    c.setFillColor(C["hdr"]); c.rect(0, H - 52, W, 52, stroke=0, fill=1)
    if c.stringWidth(title, FONT_B, 16) > W - 56:
        warn(f"page {page} title too long: '{title}'")
    c.setFillColor(white); c.setFont(FONT_B, 16); c.drawString(28, H - 26, title)
    c.setFont(FONT, 9.5); c.drawString(28, H - 42, subtitle)
    c.setFont(FONT, 8); c.drawRightString(W - 28, H - 42, f"Page {page} of {PAGES}")
    c.setFillColor(C["mute"]); c.setFont(FONT, 7); c.drawString(28, 14, CONFIG["footer"])
    repo_link(c, W - 28, 14, size=7, right=True)
    if c.stringWidth(CONFIG["footer"], FONT, 7) + c.stringWidth(CONFIG["repo_short"], FONT_B, 7) > W - 70:
        warn("footer and repository link overlap")


def repo_link(c, x, y, size=7, right=False, prefix=""):
    """Clickable repository URL (blue, bold). right=True aligns its right end at x."""
    txt = prefix + CONFIG["repo_short"]
    tw = c.stringWidth(txt, FONT_B, size)
    x0 = x - tw if right else x
    c.setFillColor(C["fabE"]); c.setFont(FONT_B, size); c.drawString(x0, y, txt)
    c.linkURL(CONFIG["repo"], (x0, y - 2, x0 + tw, y + size), relative=0)


def grid(c):
    """Debug overlay: light gridlines every 20 pt, labelled every 100 pt."""
    c.setLineWidth(0.25)
    for x in range(0, int(W) + 1, 20):
        c.setStrokeColor(HexColor("#FF6B6B") if x % 100 == 0 else HexColor("#FFC9C9"))
        c.line(x, 0, x, H)
        if x % 100 == 0:
            c.setFillColor(HexColor("#C92A2A")); c.setFont(FONT, 6); c.drawString(x + 1, 3, str(x))
    for y in range(0, int(H) + 1, 20):
        c.setStrokeColor(HexColor("#FF6B6B") if y % 100 == 0 else HexColor("#FFC9C9"))
        c.line(0, y, W, y)
        if y % 100 == 0:
            c.setFillColor(HexColor("#C92A2A")); c.setFont(FONT, 6); c.drawString(2, y + 1, str(y))


def box(c, x, y, w, h, title, lines=(), fill="fab", edge="fabE", dashed=False,
        tsize=8.6, lsize=7.2, align="center"):
    """Rounded box. title may contain '\\n'. lines = list of smaller grey text lines."""
    c.setFillColor(C[fill]); c.setStrokeColor(C[edge]); c.setLineWidth(1.1)
    if dashed:
        c.setDash(4, 3)
    c.roundRect(x, y, w, h, 5, stroke=1, fill=1)
    c.setDash()
    ty = y + h - (12 if title else 10)
    c.setFillColor(C["ink"]); c.setFont(FONT_B, tsize)
    for t in title.split("\n") if title else []:
        if c.stringWidth(t, FONT_B, tsize) > w - 6:
            warn(f"title '{t}' too wide for box at ({x},{y})")
        c.drawCentredString(x + w / 2, ty, t); ty -= tsize + 2
    c.setFont(FONT, lsize); c.setFillColor(C["mute"])
    for ln in lines:
        if c.stringWidth(ln, FONT, lsize) > w - 6:
            warn(f"line '{ln}' too wide for box at ({x},{y})")
        if align == "center":
            c.drawCentredString(x + w / 2, ty, ln)
        else:
            c.drawString(x + 6, ty, ln)
        ty -= lsize + 2.2
    if ty < y - 1:
        warn(f"text overflows bottom of box '{title}' at ({x},{y})")


def region(c, x, y, w, h, label, fill=None, edge=None, label_pos="top"):
    """Large labelled panel that groups boxes."""
    fill = fill or C["chip"]; edge = edge or C["chipE"]
    c.setFillColor(fill); c.setStrokeColor(edge); c.setLineWidth(1.2)
    c.roundRect(x, y, w, h, 7, stroke=1, fill=1)
    c.setFillColor(edge); c.setFont(FONT_B, 8.5)
    c.drawString(x + 8, (y + h - 12) if label_pos == "top" else (y + 5), label)


def arrow(c, pts, label=None, lpos=None, dashed=False, color="line", head=True, lsize=6.6):
    """Poly-line through pts [(x,y),...] with an arrow head at the last point."""
    c.setStrokeColor(C[color]); c.setFillColor(C[color]); c.setLineWidth(1.2)
    if dashed:
        c.setDash(4, 3)
    p = c.beginPath(); p.moveTo(*pts[0])
    for q in pts[1:]:
        p.lineTo(*q)
    c.drawPath(p, stroke=1, fill=0); c.setDash()
    if head:
        (x1, y1), (x2, y2) = pts[-2], pts[-1]
        a = math.atan2(y2 - y1, x2 - x1); s = 6
        hp = c.beginPath(); hp.moveTo(x2, y2)
        hp.lineTo(x2 - s * math.cos(a - 0.4), y2 - s * math.sin(a - 0.4))
        hp.lineTo(x2 - s * math.cos(a + 0.4), y2 - s * math.sin(a + 0.4)); hp.close()
        c.drawPath(hp, stroke=0, fill=1)
    if label:
        c.setFont(FONT_I, lsize); c.setFillColor(C["ink"])
        for i, t in enumerate(label.split("\n")):
            c.drawCentredString(lpos[0], lpos[1] - i * (lsize + 1.5), t)


def connector(c, x, y, letter_):
    """Small circled letter: an arrow ending at one connector continues from the other."""
    c.setFillColor(white); c.setStrokeColor(C["line"]); c.setLineWidth(1.1)
    c.circle(x, y, 5.5, stroke=1, fill=1)
    c.setFillColor(C["ink"]); c.setFont(FONT_B, 7); c.drawCentredString(x, y - 2.4, letter_)


def para(c, x, y, lines, size=7.6, lead=10.2, color="ink"):
    """Left-aligned text block. A line starting with '**' is drawn bold. Returns next y."""
    for ln in lines:
        bold = ln.startswith("**")
        c.setFont(FONT_B if bold else FONT, size); c.setFillColor(C[color])
        c.drawString(x, y, ln[2:] if bold else ln)
        y -= lead
    return y


def table(c, x, y, colw, rows, size=7.2, rowh=12.5, lead=9.0, head_fill="#102A43"):
    """rows[0] is the header. Cells may contain '\\n' for extra lines. Returns bottom y."""
    yy = y
    for r, row in enumerate(rows):
        nl = max(len(cell.split("\n")) for cell in row)
        h = rowh + (nl - 1) * lead
        top = yy + rowh - 3.5
        if r == 0:
            c.setFillColor(HexColor(head_fill)); c.rect(x, top - h, sum(colw), h, fill=1, stroke=0)
        elif r % 2 == 0:
            c.setFillColor(C["stripe"]); c.rect(x, top - h, sum(colw), h, fill=1, stroke=0)
        font = FONT_B if r == 0 else FONT
        c.setFillColor(white if r == 0 else C["ink"]); c.setFont(font, size)
        xx = x
        for i, cell in enumerate(row):
            for j, ln in enumerate(cell.split("\n")):
                if c.stringWidth(ln, font, size) > colw[i] - 6:
                    warn(f"table cell '{ln}' too wide for column {i}")
                c.drawString(xx + 4, yy - j * lead, ln)
            xx += colw[i]
        yy -= h
    return yy


def legend(c, x, y):
    c.setFont(FONT_B, 7.5); c.setFillColor(C["ink"]); c.drawString(x, y + 12, "Legend")
    items = [("fab", "fabE", "FPGA fabric (RTL / SmartHLS)"), ("mss", "mssE", "RISC-V software (Linux)"),
             ("mem", "memE", "Off-chip memory"), ("ext", "extE", "Board / external")]
    xx = x
    for f, e, t in items:
        c.setFillColor(C[f]); c.setStrokeColor(C[e]); c.rect(xx, y - 1, 14, 9, fill=1, stroke=1)
        c.setFillColor(C["ink"]); c.setFont(FONT, 7); c.drawString(xx + 18, y, t)
        xx += 18 + c.stringWidth(t, FONT, 7) + 12
    c.setDash(4, 3); c.setStrokeColor(C["line"]); c.rect(xx, y - 1, 14, 9, fill=0, stroke=1); c.setDash()
    c.drawString(xx + 18, y, "Dashed = extension")
    xx += 18 + c.stringWidth("Dashed = extension", FONT, 7) + 12
    connector(c, xx + 5, y + 3, "A"); c.setFillColor(C["ink"]); c.setFont(FONT, 7)
    c.drawString(xx + 14, y, "= linked connectors")


# =====================================================================================
# 4. PAGES
# =====================================================================================
# Column grid shared by the chip panels on page 1 (x-start, x-end).
C1, C2, C3, C4 = (186, 286), (294, 394), (402, 502), (510, 612)


def cw(a, b=None):
    """Width spanned from column a to column b (inclusive)."""
    return (b or a)[1] - a[0]


def log_bar_chart(c, x, y, rows, label_w=96, axis_w=128, lo=0.1, hi=1000.0, row_h=15):
    """Horizontal bars on a log axis. rows: (label, value_lo, value_hi, fill, edge, value_text).
    (x, y) is the top-left of the first row. Returns the y below the axis."""
    import math as _m
    x0 = x + label_w
    span = _m.log10(hi) - _m.log10(lo)
    def px(v):
        return x0 + (_m.log10(v) - _m.log10(lo)) / span * axis_w
    yy = y
    for label, vlo, vhi, fill, edge, vtxt in rows:
        c.setFont(FONT, 6.8); c.setFillColor(C["ink"])
        c.drawRightString(x0 - 4, yy - 9, label)
        c.setFillColor(C[fill]); c.setStrokeColor(C[edge]); c.setLineWidth(0.8)
        c.rect(x0, yy - 11, px(vlo) - x0, 9, fill=1, stroke=1)
        if vhi > vlo:                                    # range: draw lighter extension
            c.setDash(2, 2); c.rect(px(vlo), yy - 11, px(vhi) - px(vlo), 9, fill=0, stroke=1); c.setDash()
        c.setFont(FONT_B, 6.6); c.setFillColor(C["ink"])
        c.drawString(px(vhi) + 3, yy - 9, vtxt)
        yy -= row_h
    c.setStrokeColor(C["line"]); c.setLineWidth(0.6); c.line(x0, yy + 2, x0 + axis_w, yy + 2)
    c.setFont(FONT, 6.2); c.setFillColor(C["mute"])
    v = lo
    while v <= hi * 1.001:
        c.line(px(v), yy + 2, px(v), yy - 1)
        c.drawCentredString(px(v), yy - 8, ("%g" % v) + (" ms" if v == hi else ""))
        v *= 10
    return yy - 12


# ------------------------------------------------------------------------- PAGE 1
def page1(c):
    header(c, f"{CONFIG['project']}: {CONFIG['tagline']}",
           f"System Block Diagram  |  {CONFIG['track']}  |  Team: {CONFIG['team']}", 1)

    # ---- left column: sensors / inputs
    region(c, 20, 150, 128, 395, "SENSORS / INPUTS", C["ext"], C["extE"])
    box(c, 28, 468, 112, 60, "Odometry sensors", ["IMU (I2C/SPI) and/or", "wheel encoders;", "mikroBUS / RPi header"], "chip", "extE")
    box(c, 28, 398, 112, 60, "USB camera", ["UVC, 640x480, fixed", "forward mount; USB", "2.0 OTG, Linux V4L2"], "chip", "extE")
    box(c, 28, 300, 112, 82, "Dataset replay", ["Nordland (validated", "pre-proposal), Oxford", "RobotCar, SFU, St Lucia", "(same frame path)"], "chip", "extE")
    box(c, 28, 160, 112, 120, "Demo platform", ["handheld rig or small", "rover on a campus", "route; 12 V battery", "pack for the Icicle;", "teach run logs ground", "truth (GPS / markers)"], "chip", "extE")

    # ---- chip outline (label drawn at the bottom so the top edge stays free for wiring)
    c.setFillColor(C["chip"]); c.setStrokeColor(C["chipE"]); c.setLineWidth(1.4)
    c.roundRect(160, 72, 470, 473, 7, stroke=1, fill=1)
    c.setFillColor(C["chipE"]); c.setFont(FONT_B, 8.5)
    c.drawString(168, 76, "PolarFire SoC MPFS250T (one chip: RISC-V microprocessor subsystem + FPGA fabric)")

    # ---- RISC-V microprocessor subsystem (MSS)
    region(c, 170, 352, 450, 172, "MICROPROCESSOR SUBSYSTEM: 4x U54 RV64GC (Linux) + E51 monitor", C["mss"], C["mssE"])
    top, bot, bh = 440, 362, 64
    box(c, C1[0], top, cw(C1), bh, "Frame capture", ["V4L2 grab, grey-", "scale; frame to", "LPDDR4 for DMA"], "chip", "mssE")
    box(c, C2[0], top, cw(C2), bh, "Sequence matcher", ["SeqSLAM conv. over", "score columns,", "Lseq = 2 to 10"], "chip", "mssE")
    box(c, C3[0], top, cw(C3), bh, "Route localiser", ["1-D particle filter:", "odometry predicts,", "VPR corrects"], "chip", "mssE")
    box(c, C4[0], top, cw(C4), bh, "Odometry driver", ["IMU / encoders,", "100-200 Hz, bias", "calibration"], "chip", "mssE")
    box(c, C1[0], bot, cw(C1), bh, "Place scoring", ["assignments, hyper-", "active mask, module", "+ ensemble fusion"], "chip", "mssE")
    box(c, C2[0], bot, cw(C2), bh, "Teach-and-repeat\nmap", ["place -> pose,", "module index"], "chip", "mssE")
    box(c, C3[0], bot, cw(C3), bh, "Module scheduler", ["belief -> module list", "(all modules for", "global relocalisation)"], "chip", "mssE")
    box(c, C4[0], bot, cw(C4), bh, "Telemetry + energy", ["pose, R@1, latency,", "PAC1934 power ->", "energy per query"], "chip", "mssE")
    arrow(c, [(394, 472), (402, 472)])                               # seq matcher -> localiser
    arrow(c, [(510, 472), (502, 472)])                               # odometry -> localiser
    arrow(c, [(260, 426), (260, 433), (330, 433), (330, 440)])       # scoring -> seq matcher
    arrow(c, [(372, 426), (372, 433), (420, 433), (420, 440)])       # map -> localiser
    arrow(c, [(452, 440), (452, 426)])                               # localiser -> scheduler
    arrow(c, [(490, 440), (490, 434), (548, 434), (548, 426)])       # localiser -> telemetry
    arrow(c, [(452, 362), (452, 352)], head=False)                   # scheduler -> connector A
    connector(c, 452, 346, "A")

    # ---- FPGA fabric
    region(c, 170, 88, 450, 252, "FPGA FABRIC: SNN inference accelerator (RTL / SmartHLS)", C["fab"], C["fabE"])
    r1, r1h = 240, 74
    box(c, C4[0], r1, cw(C4), r1h, "AXI4 frame reader", ["DMA from LPDDR4", "via FIC (AXI4)", "pixel stream"])
    box(c, C3[0], r1, cw(C3), r1h, "Pre-processing", ["resize to 28x28,", "7x7 patch norm,", "scale 0-255"])
    box(c, C2[0], r1, cw(C2), r1h, "Spike generator", ["16 xorshift units,", "rate = pixel / 4 Hz,", "~2.4 events/step [M]"])
    box(c, C1[0], r1, cw(C1), r1h, "AER broadcast", ["(input id, step)", "events, same spike", "train to all engines"])
    arrow(c, [(510, 277), (502, 277)]); arrow(c, [(402, 277), (394, 277)]); arrow(c, [(294, 277), (286, 277)])

    r2, r2h = 128, 92
    box(c, C1[0], r2, cw(C1), r2h, "Spike-count\nreadout", ["400 x 8-bit counts", "per module run,", "written to LPDDR4"])
    box(c, C2[0], r2, cw(C2, C3), r2h, "SNN module engines  x4  (parallel)", [
        "each runs one VPRSNN module: 784 inputs,",
        "400 excitatory + 400 inhibitory neurons;",
        "8-bit weights, hashed delays, integer state",
        "50 lanes x 2 math blocks: 1.19 ms/run [D]",
        "R@1 within 1-2 pts of float model [M]"], tsize=8.6, lsize=7.0)
    box(c, C4[0], r2, cw(C4), r2h, "Weight streamer", ["module list in;", "DMA weights, theta,", "labels per module", "(316 KB) from", "LPDDR4; AXI4-Lite"])
    box(c, C2[0], 94, cw(C2, C3), 28, "", [
        "LSRAM weight banks + delay rings: 544 of 812 [D];",
        "resident = 4 modules; streaming = 2 run, 2 prefetch"], lsize=6.8)
    arrow(c, [(250, 240), (250, 231), (398, 231), (398, 220)], "spike events", (330, 223))
    arrow(c, [(294, 174), (286, 174)])                               # engines -> readout
    arrow(c, [(510, 174), (502, 174)])                               # streamer -> engines
    connector(c, 560, 230, "A")
    arrow(c, [(560, 224.5), (560, 220)])                             # connector A -> streamer
    arrow(c, [(186, 174), (177, 174), (177, 394), (186, 394)])       # readout -> place scoring
    c.setFont(FONT_I, 6.6); c.setFillColor(C["ink"])
    c.drawString(182, 343, "spike counts -> place scoring")

    # ---- sensors -> chip
    arrow(c, [(140, 506), (152, 506), (152, 534), (561, 534), (561, 504)], "odometry samples (I2C / SPI / GPIO)", (330, 537))
    arrow(c, [(140, 428), (156, 428), (156, 470), (186, 470)])
    arrow(c, [(140, 340), (150, 340), (150, 428)], head=False)

    # ---- right column: board I/O + memory
    region(c, 642, 72, 130, 473, "BOARD I/O + MEMORY", C["ext"], C["extE"])
    box(c, 650, 452, 114, 72, "Gigabit Ethernet", ["ground-station dash-", "board: live map, route", "error, energy plots"], "chip", "extE")
    box(c, 650, 378, 114, 62, "UART: MAVLink", ["VISION_POSITION_", "ESTIMATE -> PX4 /", "ArduPilot (ext.)"], "chip", "extE", dashed=True)
    box(c, 650, 296, 114, 70, "PAC1934 monitor", ["on-board, per-rail", "voltage + current", "over I2C"], "chip", "extE")
    box(c, 650, 160, 114, 128, "2 GB LPDDR4", ["frame buffers;", "module library:", "8-bit weights, theta,", "labels, masks for", "every module of", "every ensemble", "member; results"], "mem", "memE")
    box(c, 650, 82, 114, 68, "Toolchain", ["Libero SoC, SmartHLS,", "SoftConsole, Yocto;", "built: C golden model,", "cycle model, baselines"], "chip", "extE")
    arrow(c, [(612, 414), (640, 414), (640, 488), (650, 488)])       # telemetry -> ethernet
    arrow(c, [(612, 398), (650, 398)], dashed=True)                  # telemetry -> MAVLink
    arrow(c, [(650, 331), (640, 331), (640, 378), (612, 378)])       # PAC1934 -> telemetry
    arrow(c, [(650, 277), (612, 277)], "frames", (631, 281))         # LPDDR4 -> frame reader
    arrow(c, [(650, 174), (612, 174)], "weights", (631, 178))        # LPDDR4 -> streamer

    legend(c, 160, 44)


# ------------------------------------------------------------------------- PAGE 2
def page2(c):
    header(c, "Pre-proposal Evidence: Measured Before Writing a Line of RTL",
           "Run by Merconez on a DGX Spark (GB10), reproduced bit-exactly on a Jetson Orin Nano; authors' MIT-licensed VPRSNN code + Nordland", 2)

    # ---- fidelity chain
    region(c, 20, 455, 752, 90, "VERIFIED FIDELITY CHAIN (solid = done before the proposal, dashed = planned)", C["chip"], C["chipE"])
    chain = [
        ("Brian2 reference", ["VPRSNN code, 3 seeds;", "R@1 0.954 [M]"], False),
        ("Float model", ["spike-exact vs Brian2:", "728 / 728 spikes [M]"], False),
        ("Fixed-point model", ["integer spec; 12", "approximations tested"], False),
        ("C golden model", ["32,000 / 32,000 counts;", "same on Jetson CPU+GPU [M]"], False),
        ("RTL testbench", ["vs C golden model,", "spike by spike"], True),
        ("Icicle board", ["target: identical", "counts to C model"], True),
    ]
    bx, bw, gap = 32, 104, 20
    for i, (t, lines, planned) in enumerate(chain):
        x = bx + i * (bw + gap)
        box(c, x, 465, bw, 56, t, lines, "chip" if planned else "mss", "fabE" if planned else "mssE", dashed=planned, lsize=6.9)
        if i < len(chain) - 1:
            arrow(c, [(x + bw, 493), (x + bw + gap, 493)])

    # ---- full-scale accuracy
    region(c, 20, 290, 470, 155, "FULL-SCALE ACCURACY: Nordland, 2,700 test places, authors' released weights [M]", C["chip"], C["chipE"])
    table(c, 28, 412, [158, 62, 62, 62, 62], [
        ["Configuration (Poisson seeds P0 / P1)", "R@1", "R@5", "SeqSLAM L4", "SeqSLAM L10"],
        ["float model (Brian2-exact)", "0.549 / 0.553", "0.692 / 0.688", "0.886 / 0.872", "0.976 / 0.964"],
        ["SpikeNav hardware configuration", "0.544 / 0.543", "0.692 / 0.691", "0.856 / 0.861", "0.966 / 0.969"],
        ["  + masks from our own engine", "0.532", "0.684", "0.860", "0.967"],
        ["no synaptic delays", "0.448", "0.667", "0.821", "0.957"],
        ["hardware configuration, dt = 0.5 ms", "0.338", "0.594", "0.753", "0.928"],
    ], size=6.9, rowh=12)
    para(c, 30, 316, ["Hardware configuration = 8-bit weights, hashed 1 ms delays, hard reset, shared xorshift spike train,",
                      "integer state. Cost vs float: about 1-2 R@1 points; with sequence matching (L10) within 1 point."],
         size=6.8, lead=8.8, color="mute")

    # ---- time per module run chart
    region(c, 502, 290, 270, 155, "TIME PER MODULE RUN, ALL PLATFORMS (log scale)", C["chip"], C["chipE"])
    log_bar_chart(c, 510, 428, [
        ("Brian2, Jetson A78AE [M]", 2086, 2086, "jet", "jetE", "2.09 s"),
        ("C, Icicle U54 [E]", 480, 970, "ext", "extE", "0.5-1 s"),
        ("Brian2, DGX X925 [M]", 257, 257, "mss", "mssE", "257 ms"),
        ("C golden, Jetson A78AE [M]", 51.2, 51.2, "jet", "jetE", "51.2 ms"),
        ("C golden, DGX A725 [M]", 32.7, 32.7, "mss", "mssE", "32.7 ms"),
        ("CUDA, Jetson GPU [M]", 26.1, 26.1, "jet", "jetE", "26.1 ms"),
        ("C golden, DGX X925 [M]", 18.2, 18.2, "mss", "mssE", "18.2 ms"),
        ("SpikeNav fabric L=50 [D]", 1.19, 1.19, "fab", "fabE", "1.19 ms"),
        ("SpikeNav fabric L=100 [D]", 0.62, 0.62, "fab", "fabE", "0.62 ms"),
    ], label_w=100, axis_w=118, row_h=13, hi=10000.0)

    # ---- ablation
    region(c, 20, 40, 470, 240, "HARDWARE-APPROXIMATION ABLATION: 100 places, 3 model x 3 Poisson seeds [M]", C["chip"], C["chipE"])
    ye = table(c, 28, 252, [188, 72, 70, 74, 50], [
        ["Configuration", "R@1", "dR@1 vs Brian2", "Ensemble-3 R@1", "Result"],
        ["Brian2 reference", "0.954 ± 0.007", "-", "0.998", "ref"],
        ["(a) float model", "0.963 ± 0.003", "+0.9", "0.993", "pass"],
        ["(b) 8-bit weights", "0.947 ± 0.015", "-0.8", "0.997", "pass"],
        ["(c') 4-bit weights, per-module scale", "0.958 ± 0.010", "+0.4", "0.993", "pass"],
        ["(d) hashed delays, 0.1 ms", "0.957 ± 0.009", "+0.3", "1.000", "pass"],
        ["(e') hashed delays, 1 ms binned", "0.963 ± 0.015", "+0.9", "0.993", "pass"],
        ["(f) no delays", "0.944 ± 0.005", "-1.0", "0.997", "pass *"],
        ["(g) dt = 0.2 ms", "0.960 ± 0.007", "+0.6", "0.997", "pass"],
        ["(g) dt = 0.5 ms", "0.884 ± 0.017", "-7.0", "0.987", "FAIL"],
        ["(h) hard reset per module run", "0.952 ± 0.007", "-0.2", "0.990", "pass"],
        ["(i) one spike train shared by modules", "0.947 ± 0.012", "-0.8", "0.997", "pass"],
        ["(j) xorshift spike generator", "0.960 ± 0.012", "+0.6", "0.997", "pass"],
        ["(k) integer (fixed-point) state", "0.962 ± 0.017", "+0.8", "1.000", "pass"],
        ["(l) all combined, 8-bit weights", "0.959 ± 0.010", "+0.5", "0.993", "pass"],
        ["(l) all combined, dt = 0.5 ms", "0.823 ± 0.023", "-13.1", "0.970", "FAIL"],
    ], size=6.7, rowh=11.2)
    para(c, 30, ye - 6, ["Pass: |dR@1| <= 2 points and inside the Brian2 seed-to-seed noise band. * Saturated at 100 places:",
                         "at 2,700 places removing delays costs 10 points (table above). ± is over Poisson seeds."],
         size=6.5, lead=8.2, color="mute")

    # ---- decisions
    region(c, 502, 40, 270, 240, "DESIGN DECISIONS DRIVEN BY THE DATA", C["chip"], C["chipE"])
    para(c, 510, 252, [
        "**Keep synaptic delays, but never store them",
        "  no delays: -10 points at 2,700 places; hashed",
        "  regeneration matches random delays.",
        "**Keep the 0.1 ms time step",
        "  0.5 ms: -21 points at full scale, runaway activity.",
        "**8-bit weights (4-bit validated as fallback)",
        "  both inside the noise band; 4-bit halves LSRAM.",
        "**1 ms delay bins",
        "  no measurable loss; 6 instead of 48 LSRAM",
        "  blocks per engine.",
        "**50 lanes x 2 math blocks per engine",
        "  100 lanes would need 800 multipliers (784 exist);",
        "  50 lanes use 400 and finish a run in 1.19 ms.",
        "**Buffers sized from measured activity",
        "  8,452 input spikes per image; 2.41 per step,",
        "  max 13; <= 57 events per neuron per 1 ms slot.",
        "**The fabric is necessary, not optional",
        "  Icicle U54 cores: est. 1.9-3.9 s per 4-module query [E];",
        "  Jetson Orin Nano GPU, our bit-exact CUDA port:",
        "  28 ms per query [M]; fabric target ~1.2 ms [D].",
    ], size=6.9, lead=9.9)


# ------------------------------------------------------------------------- PAGE 3 (baselines)
def page_baselines(c):
    header(c, "Baselines: DGX Spark and Jetson Orin Nano vs the FPGA Target",
           "Same trained modules, queries and spike-train seeds on every platform  |  full reports, scripts and raw data in our repository", 3)

    region(c, 20, 292, 752, 253, "SAME WORKLOAD ON EVERY PLATFORM: one module run = 784 inputs x 800 neurons x 3,500 steps (dt 0.1 ms)",
           C["chip"], C["chipE"])
    rows = [
        ["Platform", "Implementation", "Per module run", "Per 4-module query", "Energy/query: above idle / total", "Evidence (results/)"],
        ["DGX Spark: Cortex-X925, 3.9 GHz", "Brian2 reference simulator", "257 ms [M]", "1.03 s [D]", "no rail sensors on the DGX", "e1_brian2_clean_timing.json"],
        ["DGX Spark: Cortex-X925, 3.9 GHz", "C golden model, 1 core", "18.2 ms [M]", "72.8 ms [D]", "-", "e5_bench.json"],
        ["DGX Spark: Cortex-A725, 2.8 GHz", "C golden model, 1 core", "32.7 ms [M]", "131 ms [D]", "-", "e5_bench.json"],
        ["DGX Spark: GB10 GPU", "PyTorch float engine (tooling)", "169 runs/s [M]", "-", "-", "e5_gpu.json"],
        ["Jetson Orin Nano: A78AE, 1.73 GHz", "Brian2 reference simulator", "2.09 s [M]", "8.3 s [D]", "-", "orin_brian2_timing.json"],
        ["Jetson Orin Nano: 1 A78AE core", "C golden model", "51.2 ms [M]", "205 ms [D]", "201 / 1,232 mJ [D]", "orin_cpu.json"],
        ["Jetson Orin Nano: 4 A78AE cores", "C golden, 4 modules in parallel", "51.2 ms [M]", "51.2 ms [M]", "152 / 456 mJ [D]", "orin_cpu.json"],
        ["Jetson Orin Nano: GPU, 1.02 GHz", "CUDA port (ours, bit-exact)", "26.1 ms [M]", "28.0 ms [M]", "57 / 200 mJ [D]", "orin_cuda.json"],
        ["Jetson Orin Nano: GPU, batched", "CUDA port, 96 runs per launch", "714 runs/s [M]", "-", "26 / 55 mJ [D]", "orin_cuda.json"],
        ["Jetson Orin Nano: GPU", "PyTorch float engine (tooling)", "51.5 runs/s [M]", "-", "999 / 1,396 mJ [D]", "orin_gpu.json"],
        ["Icicle Kit: U54 RV64GC, 600 MHz", "C golden model (290 M instr. [M])", "0.48-0.97 s [E]", "1.9-3.9 s [E]", "to measure (PAC1934)", "e5_bench.json"],
        ["Icicle Kit: FPGA fabric (SpikeNav)", "4 engines x 50 lanes, 125 MHz", "1.19 ms [D]", "~1.2 ms [D]", "to measure (PAC1934)", "e6_cyclemodel.json"],
    ]
    ye = table(c, 28, 517, [150, 138, 74, 90, 136, 148], rows, size=6.8, rowh=12.4)
    para(c, 30, ye - 6, [
        "Per 4-module query = 4 x module run when the modules run one after another [D]. Jetson power: INA3221 VDD_IN (whole module), sampled every 10 ms,",
        "MAXN_SUPER, idle 5.09-5.11 W; energy above idle = (P_load - P_idle) / rate, total = P_load / rate. Batched energy is per 4 module runs. The DGX has no rail sensors.",
    ], size=6.5, lead=8.4, color="mute")
    para(c, 30, ye - 34, [
        "**Takeaway: per 4-module query, the fabric target (~1.2 ms [D]) is ~23x faster than the best Jetson result (GPU, 28 ms [M]) and ~43x faster than four",
        "**Jetson CPU cores (51 ms [M]); the Icicle's U54 cores alone would need seconds [E]. Only the fabric is projected to sustain >= 25 queries/s on a 500-place ensemble.",
    ], size=7.0, lead=9.4)

    # ---- consistency
    region(c, 20, 40, 372, 242, "CROSS-PLATFORM CONSISTENCY: bit-identical everywhere [M]", C["chip"], C["chipE"])
    checks = [
        ("Golden vectors", "4 trained modules x 20 queries: identical spike\ncounts on X925, A725, Jetson A78AE, Jetson GPU\n(CUDA) and an RV64 build under QEMU"),
        ("Brian2", "Jetson reproduces the DGX output spike for\nspike: 8,206 / 8,206 spikes, E1 test module"),
        ("Activity traces", "1,200 module runs x 3,500 steps of input events:\nidentical; all sizing statistics identical"),
        ("Ablation re-run", "hardware + float configs, 3 model seeds: every\nthreshold and R@1 identical to the DGX"),
        ("Test suite", "13 / 13 tests pass on both machines, incl. the\nBrian2 exact-raster test"),
    ]
    yy = 252
    for t, d in checks:
        box(c, 30, yy - 30, 88, 32, t, [], "mss", "mssE", tsize=7.6)
        c.setFont(FONT, 6.8); c.setFillColor(C["ink"])
        for j, ln in enumerate(d.split("\n")):
            c.drawString(126, yy - 8 - j * 8.6, ln)
        yy -= 40
    para(c, 30, 50, ["=> the golden vectors are a portable contract: the RTL and the board must match them bit for bit."],
         size=6.8, lead=8.6)

    # ---- targets
    region(c, 404, 40, 368, 242, "WHY AN FPGA: TARGETS SET BY THE JETSON BASELINE", C["chip"], C["chipE"])
    table(c, 410, 254, [112, 122, 122], [
        ["Metric", "Jetson Orin Nano (best)", "SpikeNav on Icicle (target)"],
        ["Latency, 4-module query", "28.0 ms, GPU [M]", "~1.2 ms [D]: ~23x lower"],
        ["500 places x 3 members\n(60 module runs/query)", "11.9 queries/s, GPU\nsaturated [D]", "26-28 queries/s,\nstreaming [D]"],
        ["Energy, 4-module query", "200 mJ total,\n57 mJ above idle [D]", "< 20 mJ total (O3),\nPAC1934 on board"],
        ["Power while running", "7.1-9.8 W GPU; 5.1 W idle [M]", "< 16.7 W meets O3 [D]"],
        ["Determinism", "bit-exact (our port)", "bit-exact vs golden"],
    ], size=6.7, rowh=11.6, lead=8.4)
    para(c, 410, 160, [
        "**The Jetson baseline is a strong one, not a straw man",
        "Our CUDA kernel (one block per module run, delay ring in",
        "shared memory) is 14x faster than the PyTorch engine and",
        "bit-exact. Its limit is structural: 3,500 sequential steps,",
        "each with block-wide barriers. The fabric pipelines every",
        "step in dedicated lanes, with on-chip weights and delays.",
        "",
        "",
        "**Read the full baseline reports",
        "reports/dgx_spark_baseline_report.md, reports/orin_baseline_report.md",
        "and every number's source JSON (results/) in our repository:",
    ], size=6.8, lead=8.9)
    repo_link(c, 410, 52, size=5.9)


# ------------------------------------------------------------------------- PAGE 4
def page3(c):
    header(c, "SNN Module Engine: Mapping VPRSNN onto the Fabric",
           "Reference model: Ensemble of Modular SNNs (Hussaini, Milford, Fischer, IEEE T-RO 2024; MIT-licensed code)", 4)

    region(c, 20, 288, 470, 257, "ONE MODULE ENGINE (x4). Per 0.1 ms step: accumulate input events, then update 800 neurons", C["fab"], C["fabE"])
    box(c, 30, 440, 92, 80, "AER input", ["input id i at", "step t (shared", "by all engines)"], "chip", "fabE")
    box(c, 136, 440, 120, 80, "Weight bank", ["row i = w[i][0..399]", "8-bit, input-major;", "50 lanes x 8 groups", "per input event"], "chip", "fabE")
    box(c, 270, 440, 120, 80, "Delay generator", ["d[i][j] in 0-10 ms", "from a hash of (i, j):", "regenerated, never", "stored [M: no loss]"], "chip", "fabE")
    box(c, 404, 440, 78, 80, "Delay ring", ["1 ms bins,", "10 slots,", "<= 57 events", "per slot [M]"], "chip", "fabE")
    box(c, 404, 336, 78, 80, "Neuron update", ["math blocks:", "6 mult. (E),", "4 mult. (I) per", "neuron-step"], "chip", "fabE")
    box(c, 270, 336, 120, 80, "Lateral inhibition", ["E spike -> paired I", "(w = 10.4); I spikes", "N_I counted once:", "gi[j] += 17(N_I - s_j)"], "chip", "fabE")
    box(c, 136, 336, 120, 80, "Spike counters", ["400 x 8-bit, one", "per excitatory", "neuron; ~82 spikes", "per run [M]"], "chip", "fabE")
    box(c, 30, 336, 92, 80, "To readout", ["counts -> LPDDR4", "-> place scoring", "on RISC-V"], "chip", "fabE")
    arrow(c, [(122, 480), (136, 480)], "i", (129, 486))
    arrow(c, [(256, 480), (270, 480)]); arrow(c, [(390, 480), (404, 480)])
    arrow(c, [(443, 440), (443, 416)])                               # ring -> update
    arrow(c, [(404, 390), (390, 390)]); arrow(c, [(390, 362), (404, 362)])  # update <-> inhibition
    arrow(c, [(420, 336), (420, 326), (196, 326), (196, 336)])       # update -> counters
    arrow(c, [(136, 376), (122, 376)])
    para(c, 32, 309, ["Inhibition needs no 400 x 399 synapse matrix: all I-to-E weights are the constant 17.0, so one",
                      "shared inhibitory spike count per step replaces ~160k stored synapses (equivalence tested in our suite)."],
         size=7.0, lead=9.2, color="mute")

    region(c, 502, 288, 270, 257, "VPRSNN -> HARDWARE MAPPING", C["chip"], C["chipE"])
    table(c, 508, 518, [58, 98, 104], [
        ["Element", "Reference model", "SpikeNav fabric"],
        ["Input", "28x28 grey, 7x7 patch\nnorm, Poisson px/4 Hz", "SmartHLS pre-proc;\nxorshift spikes [M ok]"],
        ["Synapses", "784 x 400, w in [0,1],\nrandom 0-10 ms delay", "8-bit LSRAM rows;\nhashed 1 ms delays"],
        ["E neurons", "conductance LIF,\ntau 100 ms, theta[j]", "integer state on\nmath blocks [M ok]"],
        ["E<->I", "10.4 one-to-one;\n17.0 all-but-self", "shared I-spike count,\nO(1) per neuron"],
        ["Time", "dt 0.1 ms, 350 ms\nper image", "3,500 steps; 0.5 ms\nrejected (-21 pts)"],
        ["Readout", "assignments, hyper-\nactive mask, sum", "counts in fabric,\nscoring on U54"],
        ["Ensemble,\nsequence", "3-5 members;\nSeqSLAM, Lseq 2-10", "engines in parallel;\nSeqSLAM on U54"],
    ], size=6.6, rowh=11.5, lead=8.2)
    para(c, 510, 318, ["Constants as in the VPRSNN code, including its timer",
                       "quirk (50 ms effective E lockout), reproduced exactly."],
         size=6.5, lead=8.4, color="mute")

    region(c, 20, 40, 400, 238, "PIPELINE TIMING (cycle model driven by measured event traces)", C["chip"], C["chipE"])
    table(c, 28, 250, [262, 122], [
        ["Stage", "Value"],
        ["Input events: 8,452 per image; 2.41 per 0.1 ms step", "measured [M]"],
        ["Module run, 50 lanes x 2 multipliers, 3,500 steps", "148,490 cycles [D]"],
        ["Module run at 125 MHz (100 MHz)", "1.19 ms (1.48 ms) [D]"],
        ["Option: 100 lanes, constant multiplies in logic", "0.62 ms [D]"],
        ["Front end: DMA + 28x28 pre-processing", "< 0.3 ms [E]"],
        ["Weight swap: 316 KB per module at 1 GB/s", "0.32 ms, hidden [D]"],
        ["Scoring + SeqSLAM + particle filter (U54)", "~1-3 ms [E]"],
    ], size=7.0, rowh=12.5)
    para(c, 30, 136, [
        "**Same model in software (per module run)",
        "Brian2 on a 3.9 GHz Cortex-X925 core: 257 ms per image [M].",
        "Our C golden model: 18.2 ms on X925, 32.7 ms on A725 [M].",
        "Jetson Orin Nano: 51.2 ms on one A78AE core, 26.1 ms on its GPU [M].",
        "Icicle U54 at 600 MHz: 290 M RV64 instructions [M] -> 0.48-0.97 s [E].",
        "Neuron updates dominate: input accumulation is the bottleneck",
        "on only 3.7% of steps [D], so lanes, not memory, set the speed.",
    ], size=7.2, lead=9.6)

    region(c, 432, 40, 340, 238, "OPERATING MODES (125 MHz, 50 lanes)", C["chip"], C["chipE"])
    table(c, 440, 248, [78, 124, 122], [
        ["Mode", "What runs", "Expected rate"],
        ["Resident", "4 modules held on-chip\n(100 places, the paper's\nrobot-trial size)", "~1.2 ms per query [D]\n(+ < 0.3 ms front end)"],
        ["Streaming", "all modules of all\nmembers; 2 engines\ncompute, 2 banks load", "500 places x 3 members:\n28 queries/s at >= 1 GB/s;\n26 at 0.5 GB/s [D]"],
        ["Prior-gated", "only modules near the\nlocaliser's belief", "~1.2 ms, independent\nof route length [D]"],
        ["Batch\nbenchmark", "each module loaded once,\nall queries run through it", "full Nordland (3,300 x\n132 runs) ~2-3 min [D]"],
    ], size=6.9, rowh=12, lead=8.8)


# ------------------------------------------------------------------------- PAGE 5
def page4(c):
    header(c, "Resource Budget, Hardware / Software / IP, and Verification Flow",
           "Identification of required hardware, software, interfaces and FPGA IP cores  |  baseline: 4 engines x 50 lanes at 125 MHz", 5)

    region(c, 20, 40, 400, 505, "FPGA RESOURCE BUDGET (MPFS250T)", C["chip"], C["chipE"])
    ye = table(c, 30, 515, [168, 74, 66, 72], [
        ["Block", "Logic (LE)", "LSRAM", "Math blocks"],
        ["4 weight banks (8-bit, x40)", "-", "520 [D]", "-"],
        ["4 delay rings (1 ms bins)", "-", "24 [D]", "-"],
        ["4 engine datapaths (50 x 2 mult.)", "~40-55K [E]", "~8 [E]", "400 [D]"],
        ["Pre-processing + spike generators", "~10-15K [E]", "~10 [E]", "~16 [E]"],
        ["DMA, interconnect, CSRs, readout", "~15-20K [E]", "~20 [E]", "0"],
        ["Total", "~65-90K", "~582", "~416"],
        ["Device capacity", "254K", "812", "784"],
        ["Utilisation", "~26-35%", "~72%", "~53%"],
    ], size=7.2)
    para(c, 30, ye - 10, [
        "**How the numbers were obtained",
        "LSRAM and math-block counts come from our cycle model [D]; logic is estimated [E]",
        "and will be confirmed by Libero synthesis in the first weeks.",
        "Module = 784 x 400 weights + theta + labels + mask = 316,054 bytes [D].",
        "LSRAM is the binding resource, so:",
        "  - resident mode holds 4 modules (100 places) entirely on-chip;",
        "  - larger maps stream modules from the LPDDR4 library through the same banks;",
        "  - 4-bit weights (validated within noise [M]) need 260 instead of 520 blocks.",
        "Delays are hash-regenerated, and inhibitory synapses are not stored at all.",
        "100 lanes per engine (0.62 ms) needs 800 multipliers > 784: possible only with",
        "constant-coefficient decays moved to shift-add logic (an optimisation, not baseline).",
        "",
        "**Module library and training cost",
        "  100 places = 4 modules = 1.26 MB; 500 places x 3 members = 60 modules = 19 MB;",
        "  a 1.5 GB share of the LPDDR4 holds ~4,700 modules.",
        "  Training 500 places x 3 members with the reference code: ~31 core-hours,",
        "  ~3.1 h on the 18-core workstation we used [D].",
        "",
        "**Clocking and interfaces",
        "  fabric 125 MHz (PF_CCC); MSS <-> fabric over FIC AXI4;",
        "  LPDDR4 via the MSS DDR controller; PAC1934 and IMU over I2C.",
    ], size=7.2, lead=9.6)

    region(c, 432, 300, 340, 245, "HARDWARE, SOFTWARE AND IP CORES", C["chip"], C["chipE"])
    table(c, 440, 515, [74, 250], [
        ["Category", "Items"],
        ["Board", "PolarFire SoC Icicle Kit (MPFS250T, 2 GB LPDDR4,\nUSB OTG, GbE, PAC1934, mikroBUS, RPi header)"],
        ["Team-supplied", "USB UVC webcam; IMU breakout; microSD; 12 V\nbattery pack; handheld rig or small rover"],
        ["Microchip IP", "PolarFire SoC MSS, FIC0/FIC1 AXI4,\nCoreAXI4Interconnect, CoreAXI4DMAController,\nPF_TPSRAM / PF_DPSRAM (LSRAM), PF_URAM\n(uSRAM), PF_CCC, Math block (MACC)"],
        ["Our RTL / HLS", "pre-processing, spike generator, AER broadcast,\nmodule engines, weight streamer, readout"],
        ["Built already", "float + fixed-point Python models, bit-exact C\ngolden model, cycle model, 13-test suite; DGX\nSpark + Jetson baselines; bit-exact CUDA port"],
        ["Tools", "Libero SoC + SmartHLS, SoftConsole, Yocto\nLinux (Microchip BSP), simulator testbenches"],
        ["Software", "VPRSNN (MIT) + Brian2 reference; Linux\ndrivers, scheduler, scoring, SeqSLAM, filter"],
    ], size=6.9, rowh=12, lead=8.8)

    region(c, 432, 40, 340, 248, "VERIFICATION FLOW (solid = done, dashed = planned)", C["chip"], C["chipE"])
    steps = [("Reference", "VPRSNN + Brian2: 3 seeds, 100 places;\n2,700 places; identical on Jetson", False),
             ("Float model", "spike-exact vs Brian2 (728/728);\nall 12 approximations ablated", False),
             ("C golden model", "bit-exact vs Python (32,000 / 32,000);\nsame on DGX, Jetson CPU + GPU", False),
             ("RTL simulation", "testbench vs C golden model, spike\nby spike per engine, every step", True),
             ("On board", "fabric vs C on U54: identical counts;\nR@1, latency, PAC1934 energy", True)]
    yy = 256
    for i, (t, d, planned) in enumerate(steps):
        box(c, 446, yy - 30, 96, 26, t, [], "chip" if planned else "mss", "fabE" if planned else "mssE",
            dashed=planned, tsize=8.2)
        c.setFont(FONT, 7.0); c.setFillColor(C["ink"])
        for j, ln in enumerate(d.split("\n")):
            c.drawString(552, yy - 14 - j * 9, ln)
        if i < len(steps) - 1:
            arrow(c, [(494, yy - 30), (494, yy - 40)])
        yy -= 40


# ------------------------------------------------------------------------- PAGE 6
def page5(c):
    header(c, "Objectives, Development Plan, and Risk Mitigation",
           "Design and Development phase: Nov 30, 2026 - Mar 27, 2027 (17 weeks); evidence and baselines already complete", 6)

    region(c, 20, 318, 752, 227, "SCHEDULE AND MILESTONES", C["chip"], C["chipE"])
    gx, gy, gw = 232, 512, 528
    months = [("Oct-Nov (pre-kit)", 0, 0.22), ("Dec", 0.22, 0.40), ("Jan", 0.40, 0.62),
              ("Feb", 0.62, 0.81), ("Mar", 0.81, 1.0)]
    for name, a, b in months:
        c.setFillColor(HexColor("#E4E7EB")); c.rect(gx + a * gw, gy, (b - a) * gw - 2, 14, fill=1, stroke=0)
        c.setFillColor(C["ink"]); c.setFont(FONT_B, 7.4); c.drawCentredString(gx + (a + b) / 2 * gw, gy + 4, name)
    tasks = [
        ("Evidence: golden model, ablations, DGX + Jetson", 0.0, 0.08, "mss", "mssE", "DONE before the proposal [M]", False),
        ("RTL engine + testbench vs C golden model", 0.06, 0.30, "fab", "fabE", "engine bit-exact in simulation", False),
        ("Record campus route; train modules (~3 h)", 0.04, 0.30, "ext", "extE", "teach + repeat runs", False),
        ("Board bring-up: Linux, UVC, AXI DMA, PAC1934", 0.22, 0.36, "mss", "mssE", "M1: frames reach fabric", False),
        ("One engine on board, bit-exact vs C model", 0.30, 0.46, "fab", "fabE", "M2: board = golden model", False),
        ("4 engines, streamer, resident + streaming", 0.42, 0.60, "fab", "fabE", "M3: Nordland R@1 + energy", False),
        ("Scoring, SeqSLAM, localiser, dashboard", 0.48, 0.70, "mss", "mssE", "M4: closed loop on replay", False),
        ("Live teach-and-repeat demo on campus", 0.68, 0.84, "ext", "extE", "M5: GPS-free run", False),
        ("Extensions: 100 lanes, 4-bit, on-chip STDP", 0.70, 0.90, "fab", "fabE", "only after M1-M5", True),
        ("Report, README, video, IP and AI disclosure", 0.84, 1.0, "mss", "mssE", "freeze before Mar 27", False),
    ]
    ty = gy - 18
    for name, a, b, f, e, note, dashed in tasks:
        c.setFont(FONT, 7.2); c.setFillColor(C["ink"]); c.drawRightString(gx - 8, ty + 3, name)
        c.setFillColor(C[f]); c.setStrokeColor(C[e])
        if dashed:
            c.setDash(3, 2)
        c.roundRect(gx + a * gw, ty, (b - a) * gw, 11, 3, fill=1, stroke=1); c.setDash()
        c.setFont(FONT_I, 6.4); c.setFillColor(C["mute"])
        nx = gx + b * gw + 4
        if nx + c.stringWidth(note, FONT_I, 6.4) > W - 26:
            c.setFillColor(C["ink"]); c.drawCentredString(gx + (a + b) / 2 * gw, ty + 3, note)
        else:
            c.drawString(nx, ty + 3, note)
        ty -= 16
    para(c, 34, 336, ["Roles: (1) SNN reference, training, datasets, golden model;  (2) RTL / SmartHLS engines and verification;",
                      "(3) Linux, drivers, scoring, localiser, dashboard and demo platform.  Pre-kit RTL simulation absorbs shipping delays."],
         size=7.2, lead=9.2, color="mute")

    region(c, 20, 40, 372, 268, "OBJECTIVES AND MEASURABLE SUCCESS CRITERIA", C["chip"], C["chipE"])
    table(c, 28, 280, [24, 180, 152], [
        ["#", "Objective", "Success criterion / status"],
        ["O1", "Faithful VPRSNN inference in fabric:\nboard counts bit-identical to the\nC golden model", "R@1 within 2 pts of float;\nalready -1 to -2 pts at\n2,700 places in simulation [M]"],
        ["O2", "Real-time inference, including\nensembles", "~1.2 ms per query, ~23x below\nJetson GPU 28 ms [M]; >= 25 q/s\nfor 500 places x 3 members [D]"],
        ["O3", "Energy efficiency, measured on\nthe board (PAC1934)", "< 20 mJ per query: >= 10x below\nJetson GPU 200 mJ [D] and\nC on the U54 cores"],
        ["O4", "GPS-free teach-and-repeat on a\ncampus route", "bounded route error vs\nodometry-only drift"],
        ["O5", "Open-source engine, golden and\ncycle models, benchmarks", "public repo (DGX + Jetson\nbaselines already online);\ntests + README reproduce"],
    ], size=6.8, rowh=11.5, lead=8.6)
    para(c, 28, 104, ["**Attribution and disclosure",
                      "Builds on VPRSNN (Hussaini, Milford, Fischer, IEEE T-RO 2024, MIT License), based on",
                      "Diehl and Cook (2015); SeqSLAM (Milford and Wyeth, 2012). Hardware architecture, fixed-",
                      "point, golden, cycle and CUDA models: Merconez. AI disclosure: Claude (Anthropic) assisted",
                      "drafting; Claude Code implemented evaluation, baseline and CUDA scripts under team",
                      "direction. All results are reviewed by the team and reproducible from our repository:"],
         size=6.6, lead=8.5)
    repo_link(c, 28, 50, size=5.7)

    region(c, 404, 40, 368, 268, "KEY RISKS AND MITIGATIONS", C["chip"], C["chipE"])
    table(c, 410, 280, [112, 244], [
        ["Risk", "Mitigation / evidence"],
        ["Approximations\nhurt accuracy", "retired pre-proposal: combined configuration\n-1 to -2 R@1 points at 2,700 places [M]"],
        ["Math-block budget", "baseline 50 lanes uses 400 of 784 [D]; 100 lanes\nonly with constant multiplies in logic"],
        ["Time step too coarse", "0.5 ms rejected (-21 points) [M]; fixed at 0.1 ms"],
        ["LSRAM capacity", "~582 of 812 at 8-bit [D]; 4-bit fallback [M ok]"],
        ["DDR bandwidth", "compute-bound from 0.5 GB/s at 50 lanes [D];\nmeasure FIC throughput at M1"],
        ["Timing closure", "pipelined lanes; at 100 MHz a run takes 1.48 ms [D]"],
        ["Viewpoint sensitivity\n(stated in the paper)", "same-path teach-and-repeat, fixed camera\nmount, sequence matching"],
        ["Kit arrives late", "RTL verified against the C golden model pre-kit"],
        ["An edge GPU is\nenough", "measured, not assumed: Jetson Orin Nano GPU\n28 ms, 200 mJ per query [M/D]; target 1.2 ms"],
        ["Scope creep", "extensions only after milestone M5"],
    ], size=6.8, rowh=11.5, lead=8.6)


# =====================================================================================
# 5. BUILD
# =====================================================================================
def build():
    c = canvas.Canvas(CONFIG["output"], pagesize=landscape(letter))
    c.setTitle(f"{CONFIG['project']} - System Block Diagram ({CONFIG['team']}, {CONFIG['track']})")
    c.setAuthor(CONFIG["team"])
    for page in (page1, page2, page_baselines, page3, page4, page5):
        page(c)
        if SHOW_GRID:
            grid(c)
        c.showPage()
    c.save()
    print("wrote", CONFIG["output"])


if __name__ == "__main__":
    build()
