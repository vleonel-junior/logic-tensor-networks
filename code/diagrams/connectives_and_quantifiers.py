"""Diagrams of how LTN connectives and quantifiers act on tensor shapes.

They follow the graphical convention of the LTN paper (Badreddine et al.,
2022): tensors are boxes, the green arrows inside a box are the axes of its
free variables, and the shaded depth stands for feature dimensions;
predicates are lilac boxes, connectives and quantifiers are green circles.
The shapes in grey are those of the examples in
tutorials/2-grounding_connectives.ipynb (10 individuals in x, 5 in y).

Four diagrams are drawn:
    1. And on two subformulas that share one variable (no broadcasting);
    2. Or with broadcasting along the missing axis;
    3. Not on constants (no axis, a single number);
    4. Forall over a single axis.

Usage, from code/:
    python diagrams/connectives_and_quantifiers.py            # French and English
    python diagrams/connectives_and_quantifiers.py --lang en  # English only
    python diagrams/connectives_and_quantifiers.py --out some/folder

English files get an "-en" suffix. Images are written to diagrams/images/
by default (not versioned).
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from PIL import Image, ImageChops

plt.rcParams["mathtext.fontset"] = "cm"

TEAL = "#5da596"        # variable axes
TEAL_FILL = "#9ac7bd"   # connectives and quantifiers
LILAC = "#e1d5e7"       # predicates
TOP_FACE = "#f2f2f2"    # box depth, top face
SIDE_FACE = "#e6e6e6"   # box depth, left face
GREY = "#6b7280"        # shapes and notes
BROADCAST = "#d8ece7"   # rows copied by broadcasting

G = r"\mathcal{G}"

TEXTS = {
    "fr": {
        "repeated": "la même ligne, répétée\npour chaque individu de $y$",
        "no_axis": "aucune variable libre :\npas d'axe, un seul nombre",
        "axis_gone": "l'axe de $x$ disparaît",
    },
    "en": {
        "repeated": "the same row, repeated\nfor each individual of $y$",
        "no_axis": "no free variable:\nno axis, a single number",
        "axis_gone": "the axis of $x$ disappears",
    },
}


class Diagram:
    """A figure whose data coordinates are inches, with drawing helpers."""

    def __init__(self, width, height):
        self.fig = plt.figure(figsize=(width, height), dpi=150)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(0, height)
        self.ax.set_aspect("equal")
        self.ax.axis("off")

    def tensor(self, x, y, w, h, depth=0.18, xaxis=None, yaxis=None, z=2):
        """Box with its lower-left corner at (x, y).

        depth > 0 draws the feature dimensions: the back face is the front
        face shifted up and to the left, so the visible top and left faces
        are parallelograms. xaxis and yaxis name the variables of the
        horizontal and vertical axes.
        """
        ax = self.ax
        if depth:
            d = depth
            ax.add_patch(Polygon([(x, y + h), (x + w, y + h), (x + w - d, y + h + d), (x - d, y + h + d)],
                                 closed=True, fc=TOP_FACE, ec="black", lw=0.8, zorder=z))
            ax.add_patch(Polygon([(x, y), (x, y + h), (x - d, y + h + d), (x - d, y + d)],
                                 closed=True, fc=SIDE_FACE, ec="black", lw=0.8, zorder=z))
        ax.add_patch(Rectangle((x, y), w, h, fc="white", ec="black", lw=0.9, zorder=z + 1))
        self.axes(x, y, h, xaxis, yaxis, z=z + 2)

    def axes(self, x, y, h, xaxis=None, yaxis=None, z=4):
        """Green axis arrows in the top-left corner of a box."""
        x0, y0 = x + 0.12, y + h - 0.13
        if xaxis:
            self.arrow(x0, y0, x0 + 0.5, y0, color=TEAL, lw=1.4, z=z)
            self.ax.text(x0 + 0.25, y0 - 0.07, f"${xaxis}$", color=TEAL, fontsize=13,
                         ha="center", va="top", zorder=z)
        if yaxis:
            self.arrow(x0, y0, x0, y0 - 0.5, color=TEAL, lw=1.4, z=z)
            self.ax.text(x0 + 0.12, y0 - 0.37, f"${yaxis}$", color=TEAL, fontsize=13,
                         ha="left", va="center", zorder=z)

    def label(self, x, y, text, size=14, color="black", ha="center"):
        self.ax.text(x, y, text, fontsize=size, ha=ha, va="top", color=color, zorder=5)

    def shape(self, x, y, text):
        """Tensor shape, in grey under a label."""
        self.label(x, y + 0.13, text, size=11, color=GREY)

    def predicate(self, cx, cy, name, w=0.75, h=0.75):
        self.ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                         boxstyle="round,pad=0,rounding_size=0.12",
                                         fc=LILAC, ec="black", lw=0.9, zorder=3))
        self.ax.text(cx, cy, name, fontsize=15, ha="center", va="center", zorder=4)

    def operator(self, cx, cy, symbol, r=0.45, size=17):
        self.ax.add_patch(Circle((cx, cy), r, fc=TEAL_FILL, ec="black", lw=0.9, zorder=3))
        self.ax.text(cx, cy, symbol, fontsize=size, ha="center", va="center", zorder=4)

    def arrow(self, x0, y0, x1, y1, color="black", lw=1.2, z=1, ls="-"):
        self.ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=11,
                                          color=color, lw=lw, zorder=z, shrinkA=0, shrinkB=0,
                                          linestyle=ls))

    def wire(self, points, color="black", lw=1.2, ls="-"):
        """Orthogonal polyline through points, ending with an arrow head."""
        xs, ys = zip(*points)
        self.ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, zorder=1, ls=ls, solid_capstyle="butt")
        self.arrow(*points[-2], *points[-1], color=color, lw=lw, ls=ls)

    def save(self, path, margin=30):
        """Save as PNG, cropped to the drawing plus a white margin in pixels."""
        self.fig.savefig(path, dpi=150, facecolor="white", bbox_inches="tight", pad_inches=0.15)
        plt.close(self.fig)
        image = Image.open(path).convert("RGB")
        box = ImageChops.difference(image, Image.new("RGB", image.size, "white")).getbbox()
        image.crop((max(0, box[0] - margin), max(0, box[1] - margin),
                    min(image.width, box[2] + margin), min(image.height, box[3] + margin))).save(path)


def and_same_variable(path, texts):
    """And(Eq(x, c1), Eq(x, c2)): both sides have shape (10,), no broadcasting."""
    d = Diagram(13.5, 4.6)
    d.tensor(1.0, 3.3, 1.6, 0.55, depth=0.15)
    d.label(1.8, 3.2, f"${G}(c_1)$")
    d.tensor(0.6, 1.85, 2.4, 0.55, xaxis="x")
    d.label(1.8, 1.75, f"${G}(x)$")
    d.tensor(1.0, 0.4, 1.6, 0.55, depth=0.15)
    d.label(1.8, 0.3, f"${G}(c_2)$")

    d.predicate(4.6, 3.1, r"$\mathrm{Eq}$")
    d.predicate(4.6, 1.2, r"$\mathrm{Eq}$")
    d.wire([(2.6, 3.575), (3.9, 3.575), (3.9, 3.25), (4.225, 3.25)])
    d.wire([(3.0, 2.125), (3.6, 2.125), (3.6, 2.95), (4.225, 2.95)])
    d.wire([(3.6, 2.125), (3.6, 1.35), (4.225, 1.35)])
    d.wire([(2.6, 0.675), (3.9, 0.675), (3.9, 1.05), (4.225, 1.05)])

    # one value per individual of x
    d.tensor(5.6, 2.85, 2.4, 0.5, depth=0, xaxis="x")
    d.label(6.8, 2.75, f"${G}(\\mathrm{{Eq}}(x, c_1))$")
    d.shape(6.8, 2.28, "(10,)")
    d.tensor(5.6, 0.95, 2.4, 0.5, depth=0, xaxis="x")
    d.label(6.8, 0.85, f"${G}(\\mathrm{{Eq}}(x, c_2))$")
    d.shape(6.8, 0.38, "(10,)")
    d.wire([(4.975, 3.1), (5.6, 3.1)])
    d.wire([(4.975, 1.2), (5.6, 1.2)])

    # element-wise on the shared axis
    d.operator(9.1, 2.15, r"$\wedge$")
    d.wire([(8.0, 3.1), (8.35, 3.1), (8.35, 2.3), (8.65, 2.3)])
    d.wire([(8.0, 1.2), (8.35, 1.2), (8.35, 2.0), (8.65, 2.0)])
    d.tensor(10.2, 1.9, 2.4, 0.5, depth=0, xaxis="x")
    d.wire([(9.55, 2.15), (10.2, 2.15)])
    d.label(11.4, 1.8, f"${G}(\\mathrm{{Eq}}(x, c_1) \\wedge \\mathrm{{Eq}}(x, c_2))$", size=13)
    d.shape(11.4, 1.33, "(10,)")
    d.save(path)


def or_broadcast(path, texts):
    """Or(Eq(x, c1), Eq(x, y)): (10,) is repeated along y to match (10, 5)."""
    d = Diagram(16.5, 5.6)
    d.tensor(1.0, 4.35, 1.4, 0.55, depth=0.15)
    d.label(1.7, 4.25, f"${G}(c_1)$")
    d.tensor(0.6, 2.75, 2.4, 0.55, xaxis="x")
    d.label(1.8, 2.65, f"${G}(x)$")
    d.tensor(0.9, 0.9, 1.8, 0.55, xaxis="y")
    d.label(1.8, 0.8, f"${G}(y)$")

    d.predicate(4.5, 4.1, r"$\mathrm{Eq}$")
    d.predicate(4.5, 1.6, r"$\mathrm{Eq}$")
    d.wire([(2.4, 4.625), (3.8, 4.625), (3.8, 4.25), (4.125, 4.25)])
    d.wire([(3.0, 3.025), (3.5, 3.025), (3.5, 3.95), (4.125, 3.95)])
    d.wire([(3.5, 3.025), (3.5, 1.75), (4.125, 1.75)])
    d.wire([(2.7, 1.175), (3.8, 1.175), (3.8, 1.45), (4.125, 1.45)])

    # Eq(x, c1): a vector along x
    d.tensor(5.4, 3.85, 2.4, 0.5, depth=0, xaxis="x")
    d.wire([(4.875, 4.1), (5.4, 4.1)])
    d.label(6.6, 3.75, f"${G}(\\mathrm{{Eq}}(x, c_1))$")
    d.shape(6.6, 3.28, "(10,)")

    # Eq(x, y): a matrix, x across, y down
    d.tensor(5.4, 0.55, 2.4, 1.9, depth=0, xaxis="x", yaxis="y")
    d.wire([(4.875, 1.6), (5.4, 1.6)])
    d.label(6.6, 0.45, f"${G}(\\mathrm{{Eq}}(x, y))$")
    d.shape(6.6, -0.02, "(10, 5)")

    # broadcasting: the vector becomes the first row, copied for every y
    bx, by, bw, bh = 9.1, 2.9, 2.4, 1.9
    rows = 5
    for k in range(rows):
        d.ax.add_patch(Rectangle((bx, by + bh - (k + 1) * bh / rows), bw, bh / rows,
                                 fc=BROADCAST if k else "white", ec=TEAL, lw=0.7, ls="--", zorder=2))
    d.ax.add_patch(Rectangle((bx, by), bw, bh, fc="none", ec="black", lw=0.9, zorder=3))
    d.axes(bx, by, bh, xaxis="x", yaxis="y")
    d.wire([(7.8, 4.1), (8.45, 4.1), (8.45, 4.62), (9.1, 4.62)], color=TEAL, ls="--")
    d.ax.text(8.45, 5.05, "broadcasting", fontsize=11, color=TEAL, ha="center", va="bottom", style="italic")
    d.label(bx + bw / 2, by - 0.1, texts["repeated"], size=10.5, color=GREY)
    d.shape(bx + bw / 2, by - 0.62, "(10, 5)")

    # element-wise on the (x, y) grid
    d.operator(13.0, 2.4, r"$\vee$")
    d.wire([(bx + bw, 3.85), (12.3, 3.85), (12.3, 2.55), (12.55, 2.55)])
    d.wire([(7.8, 1.5), (12.3, 1.5), (12.3, 2.25), (12.55, 2.25)])
    d.tensor(14.0, 1.45, 2.3, 1.9, depth=0, xaxis="x", yaxis="y")
    d.wire([(13.45, 2.4), (14.0, 2.4)])
    d.label(15.15, 1.35, f"${G}(\\mathrm{{Eq}}(x, c_1) \\vee \\mathrm{{Eq}}(x, y))$", size=13)
    d.shape(15.15, 0.85, "(10, 5)")
    d.save(path)


def not_constants(path, texts):
    """Not(Eq(c1, c2)): no free variable, a single number in and out."""
    d = Diagram(12.5, 3.2)
    d.tensor(0.6, 2.05, 1.4, 0.55, depth=0.15)
    d.label(1.3, 1.95, f"${G}(c_1)$")
    d.tensor(0.6, 0.45, 1.4, 0.55, depth=0.15)
    d.label(1.3, 0.35, f"${G}(c_2)$")

    d.predicate(3.6, 1.5, r"$\mathrm{Eq}$")
    d.wire([(2.0, 2.325), (2.9, 2.325), (2.9, 1.65), (3.225, 1.65)])
    d.wire([(2.0, 0.725), (2.9, 0.725), (2.9, 1.35), (3.225, 1.35)])
    d.tensor(4.6, 1.2, 0.6, 0.6, depth=0)
    d.wire([(3.975, 1.5), (4.6, 1.5)])
    d.label(4.9, 1.1, f"${G}(\\mathrm{{Eq}}(c_1, c_2))$", size=13)
    d.shape(4.9, 0.63, "0.0178")

    d.operator(6.5, 1.5, r"$\neg$", size=24)
    d.wire([(5.2, 1.5), (6.05, 1.5)])
    d.tensor(7.6, 1.2, 0.6, 0.6, depth=0)
    d.wire([(6.95, 1.5), (7.6, 1.5)])
    d.label(7.9, 1.1, f"${G}(\\neg\\,\\mathrm{{Eq}}(c_1, c_2))$", size=13)
    d.shape(7.9, 0.63, "1 − 0.0178 = 0.9822")
    d.label(9.3, 1.75, texts["no_axis"], size=10.5, color=GREY, ha="left")
    d.save(path)


def forall_one_axis(path, texts):
    """Forall(x, Eq(x, y)): the axis of x is aggregated away, (10, 5) -> (5,)."""
    d = Diagram(12.5, 4.2)
    d.tensor(0.6, 2.75, 2.4, 0.55, xaxis="x")
    d.label(1.8, 2.65, f"${G}(x)$")
    d.tensor(0.9, 0.9, 1.8, 0.55, xaxis="y")
    d.label(1.8, 0.8, f"${G}(y)$")

    d.predicate(4.3, 2.1, r"$\mathrm{Eq}$")
    d.wire([(3.0, 3.025), (3.6, 3.025), (3.6, 2.25), (3.925, 2.25)])
    d.wire([(2.7, 1.175), (3.6, 1.175), (3.6, 1.95), (3.925, 1.95)])
    d.tensor(5.3, 1.15, 2.4, 1.9, depth=0, xaxis="x", yaxis="y")
    d.wire([(4.675, 2.1), (5.3, 2.1)])
    d.label(6.5, 1.05, f"${G}(\\mathrm{{Eq}}(x, y))$")
    d.shape(6.5, 0.58, "(10, 5)")

    d.operator(8.75, 2.1, r"$\forall x$", r=0.55, size=16)
    d.wire([(7.7, 2.1), (8.2, 2.1)])
    d.tensor(10.2, 1.15, 0.55, 1.9, depth=0, yaxis="y")
    d.wire([(9.3, 2.1), (10.2, 2.1)])
    d.label(10.47, 1.05, f"${G}(\\forall x\\, \\mathrm{{Eq}}(x, y))$", size=13)
    d.shape(10.47, 0.58, "(5,)")
    d.label(8.75, 3.35, texts["axis_gone"], size=10.5, color=GREY)
    d.save(path)


DIAGRAMS = {
    "schema-et-meme-variable": and_same_variable,
    "schema-ou-broadcasting": or_broadcast,
    "schema-non-constantes": not_constants,
    "schema-pour-tout-un-axe": forall_one_axis,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "images",
                        help="output folder (default: diagrams/images/)")
    parser.add_argument("--lang", choices=["fr", "en", "all"], default="all",
                        help="language of the notes drawn in the diagrams")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    languages = ["fr", "en"] if args.lang == "all" else [args.lang]
    for lang in languages:
        suffix = "" if lang == "fr" else "-en"
        for name, draw in DIAGRAMS.items():
            path = args.out / f"{name}{suffix}.png"
            draw(path, TEXTS[lang])
            print(path)


if __name__ == "__main__":
    main()
