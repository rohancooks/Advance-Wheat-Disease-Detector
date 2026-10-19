"""Build the README figures from the notebook and the paper. Stdlib only.

    python assets/make_figures.py

Training curves are parsed from the notebook's saved training log. Sample
photos are pulled out of the paper PDF. The confusion matrix counts are read off
the notebook's heatmap (they exist only as an image there), with its two
middle labels put back in LabelBinarizer order (see README, "Notes").
"""
import base64
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
NOTEBOOK = os.path.join(ROOT, "Wheat_Disease_Detection_final.ipynb")
PAPER = os.path.join(ROOT, "IEEE-STCR-2025_paper_1984.pdf")

FONT = "system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"

# Class order = LabelBinarizer order (alphabetical), which is what the model outputs.
CLASSES = ["Crown and Root Rot", "Healthy Wheat", "Leaf Rust", "Wheat Loose Smut"]
COUNTS = [1021, 1146, 1223, 930]
CM = [[223, 16, 10, 6],
      [5, 268, 8, 5],
      [4, 17, 278, 7],
      [12, 28, 5, 188]]

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7", border="rgba(11,11,11,0.10)",
                  s1="#2a78d6", s2="#eb6834", s3="#1baf7a",
                  ramp=[(0, "#f1f6fd"), (.15, "#cde2fb"), (.4, "#86b6ef"), (.7, "#2a78d6"), (1, "#104281")]),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781",
                 grid="#2c2c2a", axis="#383835", border="rgba(255,255,255,0.10)",
                 s1="#3987e5", s2="#d95926", s3="#199e70",
                 ramp=[(0, "#202226"), (.15, "#1e3350"), (.4, "#1c5cab"), (.7, "#3987e5"), (1, "#9ec5f4")]),
}


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def svg(w, h, body, t):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
            f'font-family="{FONT}">\n'
            f'<rect x="0.5" y="0.5" width="{w - 1}" height="{h - 1}" rx="12" fill="{t["surface"]}" '
            f'stroke="{t["border"]}"/>\n{body}\n</svg>\n')


def text(x, y, s, fill, size=13, weight=400, anchor="start", extra=""):
    return (f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" font-size="{size}" font-weight="{weight}" '
            f'text-anchor="{anchor}" {extra}>{esc(s)}</text>')


def hex_rgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def ramp(stops, v):
    for (a, ca), (b, cb) in zip(stops, stops[1:]):
        if v <= b:
            f = (v - a) / (b - a)
            ra, rb = hex_rgb(ca), hex_rgb(cb)
            return "#" + "".join(f"{round(x + (y - x) * f):02x}" for x, y in zip(ra, rb))
    return stops[-1][1]


def luminance(h):
    def lin(c):
        c /= 255
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (lin(c) for c in hex_rgb(h))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


# ---------------------------------------------------------------- data
def training_log():
    nb = json.load(open(NOTEBOOK, encoding="utf-8"))
    cell = next(c for c in nb["cells"] if "final_model.fit(" in "".join(c["source"]))
    log = re.sub(r"\x1b\[[0-9;]*m", "", "".join("".join(o.get("text", "")) for o in cell["outputs"]))
    pat = (r"50/50 \S+ (\d+)s \S+ - accuracy: ([\d.]+) - loss: ([\d.]+) - "
           r"val_accuracy: ([\d.]+) - val_loss: ([\d.]+)")
    rows = [tuple(float(x) for x in m) for m in re.findall(pat, log)]
    assert len(rows) == 30, len(rows)
    # (epoch, seconds, acc, loss, val_acc, val_loss). Epochs that took a few seconds
    # ran a single batch (the generator ran out), so their training numbers are one batch.
    return [(i + 1,) + r for i, r in enumerate(rows)]


def paper_photos():
    """The four Fig. 1 photos, keyed by class, as base64 JPEG."""
    pdf = open(PAPER, "rb").read()
    want = {(250, 167): "Crown and Root Rot", (522, 379): "Healthy Wheat",
            (337, 447): "Leaf Rust", (141, 255): "Wheat Loose Smut"}
    out = {}
    for m in re.finditer(rb"/Width (\d+)", pdf):
        head = pdf[max(0, m.start() - 300): m.start() + 300]
        h = re.search(rb"/Height (\d+)", head)
        key = (int(m.group(1)), int(h.group(1))) if h else None
        if key in want and b"/DCTDecode" in head:
            start = pdf.index(b"stream", m.start()) + 6
            start += 2 if pdf[start:start + 2] == b"\r\n" else 1
            end = pdf.index(b"endstream", start)
            out[want[key]] = base64.b64encode(pdf[start:end].rstrip(b"\r\n")).decode()
    assert len(out) == 4, out.keys()
    return out


# ---------------------------------------------------------------- figures
def classes_card():
    """Theme-neutral: text sits on a dark scrim over each photo."""
    photos = paper_photos()
    blurbs = {"Crown and Root Rot": "Rot at the roots and stem base",
              "Healthy Wheat": "Disease-free reference class",
              "Leaf Rust": "Orange-brown pustules on the leaves",
              "Wheat Loose Smut": "Grain heads turn black and powdery"}
    W, H, gap, cw, ch = 900, 250, 12, 216, 250
    parts = ['<defs><linearGradient id="scrim" x1="0" y1="0" x2="0" y2="1">'
             '<stop offset="0.45" stop-color="#000" stop-opacity="0"/>'
             '<stop offset="1" stop-color="#000" stop-opacity="0.82"/></linearGradient>']
    for i in range(4):
        parts.append(f'<clipPath id="c{i}"><rect x="{i * (cw + gap)}" y="0" width="{cw}" height="{ch}" rx="12"/></clipPath>')
    parts.append("</defs>")
    for i, (name, n) in enumerate(zip(CLASSES, COUNTS)):
        x = i * (cw + gap)
        parts.append(f'<g clip-path="url(#c{i})">'
                     f'<image href="data:image/jpeg;base64,{photos[name]}" x="{x}" y="0" width="{cw}" height="{ch}" '
                     f'preserveAspectRatio="xMidYMid slice"/>'
                     f'<rect x="{x}" y="0" width="{cw}" height="{ch}" fill="url(#scrim)"/></g>')
        parts.append(text(x + 14, ch - 52, name, "#ffffff", 16, 700))
        parts.append(text(x + 14, ch - 33, blurbs[name], "#e8e8e3", 11.5))
        parts.append(text(x + 14, ch - 14, f"{n:,} images", "#ffffff", 12, 600))
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
            f'font-family="{FONT}">\n' + "\n".join(parts) + "\n</svg>\n")


def training_chart(t):
    log = training_log()
    W, H = 900, 380
    panels = [("Accuracy", 2, 4, (0.4, 1.0), [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], lambda v: f"{v * 100:.0f}%", max),
              ("Loss", 3, 5, (0.0, 3.0), [0, 0.5, 1, 1.5, 2, 2.5, 3], lambda v: f"{v:g}", min)]
    out = [text(28, 38, "Training curves, 30 epochs", t["ink"], 17, 650),
           text(28, 60, "VGG19 base frozen; only the new classification head learns", t["ink2"], 13)]
    # legend
    lx = W - 360
    out.append(f'<line x1="{lx}" y1="44" x2="{lx + 22}" y2="44" stroke="{t["s1"]}" stroke-width="2" stroke-linecap="round"/>')
    out.append(f'<circle cx="{lx + 11}" cy="44" r="4" fill="{t["s1"]}" stroke="{t["surface"]}" stroke-width="2"/>')
    out.append(text(lx + 30, 48, "Training", t["ink2"], 13))
    out.append(f'<line x1="{lx + 120}" y1="44" x2="{lx + 142}" y2="44" stroke="{t["s2"]}" stroke-width="2" stroke-linecap="round"/>')
    out.append(text(lx + 150, 48, "Validation", t["ink2"], 13))
    pw, top, bottom = 360, 92, H - 66
    for p, (title, ti, vi, (lo, hi), ticks, fmt, best) in enumerate(panels):
        x0 = 72 + p * (pw + 80)
        x1 = x0 + pw
        X = lambda e: x0 + (e - 1) / 29 * pw
        Y = lambda v: bottom - (v - lo) / (hi - lo) * (bottom - top)
        out.append(text(x0, top - 12, title, t["ink"], 14, 600))
        for v in ticks:
            out.append(f'<line x1="{x0}" y1="{Y(v):.1f}" x2="{x1}" y2="{Y(v):.1f}" stroke="{t["grid"] if v != lo else t["axis"]}" stroke-width="1"/>')
            out.append(text(x0 - 8, Y(v) + 4, fmt(v), t["muted"], 11.5, anchor="end", extra='font-variant-numeric="tabular-nums"'))
        for e in (1, 5, 10, 15, 20, 25, 30):
            out.append(text(X(e), bottom + 18, e, t["muted"], 11.5, anchor="middle"))
        out.append(text((x0 + x1) / 2, bottom + 36, "Epoch", t["muted"], 11.5, anchor="middle"))
        full = [r for r in log if r[1] > 20]          # epochs that ran all 50 batches
        tr = " ".join(f"{X(r[0]):.1f},{Y(r[ti]):.1f}" for r in full)
        va = " ".join(f"{X(r[0]):.1f},{Y(r[vi]):.1f}" for r in log)
        out.append(f'<polyline points="{va}" fill="none" stroke="{t["s2"]}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
        out.append(f'<polyline points="{tr}" fill="none" stroke="{t["s1"]}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
        for r in full:
            out.append(f'<circle cx="{X(r[0]):.1f}" cy="{Y(r[ti]):.1f}" r="4" fill="{t["s1"]}" stroke="{t["surface"]}" stroke-width="2"/>')
        b = best(log, key=lambda r: r[vi])
        out.append(f'<circle cx="{X(b[0]):.1f}" cy="{Y(b[vi]):.1f}" r="5" fill="{t["s2"]}" stroke="{t["surface"]}" stroke-width="2"/>')
        lbl = f"best {f'{b[vi]:.2f}' if title == 'Loss' else f'{b[vi] * 100:.1f}%'} (epoch {b[0]:.0f})"
        dy = 18 if title == "Loss" else -12
        out.append(text(X(b[0]), Y(b[vi]) + dy, lbl, t["ink"], 12, 600, anchor="end"))
    out.append(text(28, H - 12, "Training dots = the 15 epochs that ran all 50 batches; the other 15 ran one batch each "
                    "(see Notes). Validation is scored every epoch.", t["muted"], 11.5))
    return svg(W, H, "\n".join(out), t)


def confusion_chart(t):
    W, H = 900, 452
    left, top, cw, ch = 210, 118, 160, 68
    out = [text(28, 38, "Confusion matrix, 1,080 held-out test images", t["ink"], 17, 650),
           text(28, 60, "Cell colour = share of that row (actual class). Diagonal = recall. "
                        "61 of the 123 errors were diseased plants predicted as Healthy.", t["ink2"], 13)]
    short = ["Crown & Root Rot", "Healthy", "Leaf Rust", "Loose Smut"]
    out.append(text(left + 2 * cw, top - 34, "Predicted class", t["muted"], 12, 600, anchor="middle"))
    for j, s in enumerate(short):
        out.append(text(left + j * cw + cw / 2, top - 12, s, t["ink2"], 12.5, 600, anchor="middle"))
    out.append(text(36, top + 2 * ch, "Actual", t["muted"], 12, 600, anchor="middle",
                    extra=f'transform="rotate(-90 36 {top + 2 * ch})"'))
    for i, row in enumerate(CM):
        n = sum(row)
        y = top + i * ch
        out.append(text(left - 14, y + ch / 2 - 2, short[i], t["ink2"], 12.5, 600, anchor="end"))
        out.append(text(left - 14, y + ch / 2 + 14, f"n = {n}", t["muted"], 11, anchor="end"))
        for j, v in enumerate(row):
            share = v / n
            fill = ramp(t["ramp"], share)
            ink = "#ffffff" if luminance(fill) < 0.36 else "#0b0b0b"
            x = left + j * cw
            out.append(f'<rect x="{x + 1}" y="{y + 1}" width="{cw - 2}" height="{ch - 2}" rx="4" fill="{fill}"/>')
            out.append(text(x + cw / 2, y + ch / 2 + 1, v, ink, 18, 650, anchor="middle", extra='font-variant-numeric="tabular-nums"'))
            out.append(text(x + cw / 2, y + ch / 2 + 19, f"{share * 100:.1f}%", ink, 11.5, anchor="middle"))
    acc = sum(CM[i][i] for i in range(4)) / sum(map(sum, CM))
    out.append(text(left, top + 4 * ch + 34, f"Overall accuracy {acc * 100:.1f}%  ·  957 of 1,080 correct",
                    t["ink"], 13, 600))
    return svg(W, H, "\n".join(out), t)


def metrics_chart(t):
    P = [CM[i][i] / sum(CM[r][i] for r in range(4)) for i in range(4)]
    R = [CM[i][i] / sum(CM[i]) for i in range(4)]
    F = [2 * p * r / (p + r) for p, r in zip(P, R)]
    W, H = 900, 330
    x0, x1, top, rh = 200, 860, 112, 46
    lo, hi = 0.75, 1.0
    X = lambda v: x0 + (v - lo) / (hi - lo) * (x1 - x0)
    out = [text(28, 38, "Per-class precision, recall and F1", t["ink"], 17, 650),
           text(28, 60, "Axis starts at 75% to separate the dots", t["ink2"], 13)]
    lx = W - 330
    for k, (lab, col) in enumerate((("Precision", t["s1"]), ("Recall", t["s2"]), ("F1", t["s3"]))):
        out.append(f'<circle cx="{lx + k * 105}" cy="44" r="5" fill="{col}"/>')
        out.append(text(lx + k * 105 + 11, 48, lab, t["ink2"], 13))
    bottom = top + 4 * rh
    for v in (0.75, 0.8, 0.85, 0.9, 0.95, 1.0):
        out.append(f'<line x1="{X(v):.1f}" y1="{top - 18}" x2="{X(v):.1f}" y2="{bottom - 18}" stroke="{t["grid"]}" stroke-width="1"/>')
        out.append(text(X(v), bottom + 2, f"{v * 100:.0f}%", t["muted"], 11.5, anchor="middle"))
    notes = {(1, "P"): "lowest precision", (3, "R"): "lowest recall"}
    for i, name in enumerate(CLASSES):
        y = top + i * rh
        vals = (("P", P[i], t["s1"]), ("R", R[i], t["s2"]), ("F", F[i], t["s3"]))
        out.append(text(x0 - 16, y + 4, name, t["ink2"], 12.5, 600, anchor="end"))
        mn, mx = min(v for _, v, _ in vals), max(v for _, v, _ in vals)
        out.append(f'<line x1="{X(mn):.1f}" y1="{y}" x2="{X(mx):.1f}" y2="{y}" stroke="{t["axis"]}" stroke-width="2" stroke-linecap="round"/>')
        for key, v, col in vals:
            out.append(f'<circle cx="{X(v):.1f}" cy="{y}" r="6" fill="{col}" stroke="{t["surface"]}" stroke-width="2"/>')
            if (i, key) in notes:
                out.append(text(X(v) - 12, y + 4, f"{notes[(i, key)]} {v * 100:.1f}%", t["ink"], 12, 600, anchor="end"))
    return svg(W, H, "\n".join(out), t)


def banner():
    W, H = 1280, 320
    stalks = []
    for k, (x, h, s, d) in enumerate([(985, 230, 1.0, 0), (1040, 270, 1.1, 1.2), (1095, 215, 0.95, 0.6),
                                      (1150, 255, 1.05, 1.8), (1205, 225, 0.9, 0.9)]):
        base = H + 10
        tip = base - h
        g = [f'<path d="M{x},{base} Q{x - 6},{(base + tip) / 2} {x},{tip + 60}" stroke="#d9b45a" stroke-width="3" fill="none"/>']
        for n in range(7):
            cy = tip + 8 + n * 12 * s
            for side in (-1, 1):
                g.append(f'<ellipse cx="{x + side * 7 * s:.1f}" cy="{cy:.1f}" rx="{5 * s:.1f}" ry="{11 * s:.1f}" '
                         f'transform="rotate({side * 28} {x + side * 7 * s:.1f} {cy:.1f})" fill="{"#f2cf6b" if n % 2 else "#e2b54a"}"/>')
        g.append(f'<ellipse cx="{x}" cy="{tip:.1f}" rx="{4 * s:.1f}" ry="{10 * s:.1f}" fill="#f6dc8a"/>')
        for n in range(4):
            g.append(f'<line x1="{x}" y1="{tip - 4}" x2="{x + (n - 1.5) * 6}" y2="{tip - 36}" stroke="#f2d27a" stroke-width="1.2"/>')
        stalks.append(f'<g class="sway" style="animation-delay:-{d}s;transform-origin:{x}px {base}px">{"".join(g)}</g>')
    chips = ["4 disease classes", "4,320 images", "88.6% test accuracy", "IEEE STCR 2025"]
    cx, chip_svg = 64, []
    for c in chips:
        w = 18 + len(c) * 8.2
        chip_svg.append(f'<rect x="{cx}" y="214" width="{w:.0f}" height="34" rx="17" fill="#ffffff" fill-opacity="0.10" stroke="#ffffff" stroke-opacity="0.25"/>'
                        + text(cx + w / 2, 236, c, "#fff6dc", 14.5, 600, anchor="middle"))
        cx += w + 10
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="{FONT}">
<style>
.sway{{animation:sway 6s ease-in-out infinite alternate}}
@keyframes sway{{from{{transform:rotate(-2.2deg)}}to{{transform:rotate(2.2deg)}}}}
@media (prefers-reduced-motion: reduce){{.sway{{animation:none}}}}
</style>
<defs>
<linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
<stop offset="0" stop-color="#0c2418"/><stop offset="0.55" stop-color="#17432a"/><stop offset="1" stop-color="#5a5a1e"/>
</linearGradient>
<radialGradient id="sun" cx="0.86" cy="0.2" r="0.5"><stop offset="0" stop-color="#ffd86b" stop-opacity="0.35"/><stop offset="1" stop-color="#ffd86b" stop-opacity="0"/></radialGradient>
<clipPath id="r"><rect width="{W}" height="{H}" rx="18"/></clipPath>
</defs>
<g clip-path="url(#r)">
<rect width="{W}" height="{H}" fill="url(#bg)"/>
<rect width="{W}" height="{H}" fill="url(#sun)"/>
{"".join(stalks)}
{text(64, 78, "DEEP LEARNING  ·  PRECISION AGRICULTURE", "#d9c58c", 14, 700, extra='letter-spacing="2.5"')}
{text(62, 138, "Advanced Wheat Disease Detector", "#ffffff", 50, 800)}
{text(64, 180, "Classifies wheat crop images into four health classes with a fine-tuned VGG19 CNN", "#e6ecd9", 19)}
{"".join(chip_svg)}
</g>
</svg>
'''


def main():
    write("banner.svg", banner())
    write("classes.svg", classes_card())
    for mode, t in THEMES.items():
        write(f"training-{mode}.svg", training_chart(t))
        write(f"confusion-{mode}.svg", confusion_chart(t))
        write(f"metrics-{mode}.svg", metrics_chart(t))


def write(name, s):
    with open(os.path.join(HERE, name), "w", encoding="utf-8", newline="\n") as f:
        f.write(s)
    print(f"  {name:22s} {len(s) / 1024:6.1f} KB")


if __name__ == "__main__":
    main()
