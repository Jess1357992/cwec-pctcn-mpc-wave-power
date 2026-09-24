"""Create a publication-ready architecture diagram for the implemented PC-TCN."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs" / "submission_figures"

BLUE = "#3B6FB6"
BLUE_LIGHT = "#E8F0FA"
ORANGE = "#D9792B"
ORANGE_LIGHT = "#FCEBDD"
GREEN = "#4C956C"
GREEN_LIGHT = "#E4F2E9"
PURPLE = "#7A62A3"
PURPLE_LIGHT = "#EEE9F5"
GRAY = "#667085"
GRAY_LIGHT = "#F2F4F7"
BLACK = "#1F2937"


def rounded_box(ax, x, y, w, h, text, face, edge, fontsize=8.5,
                linewidth=1.25, radius=0.12, weight="normal", zorder=2):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.035,rounding_size={radius}",
        linewidth=linewidth, edgecolor=edge, facecolor=face, zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=BLACK, weight=weight, zorder=zorder + 1,
            linespacing=1.22)
    return patch


def arrow(ax, start, end, color=GRAY, linewidth=1.3, style="-|>",
          connection="arc3", zorder=1, mutation=11):
    patch = FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=mutation,
        linewidth=linewidth, color=color, connectionstyle=connection,
        shrinkA=2, shrinkB=2, zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.linewidth": 0.8,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(16.0, 7.8), constrained_layout=False)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 8.8)
    ax.axis("off")

    # Panel (a): model overview.
    ax.text(0.2, 8.48, "(a) PC-TCN overview", fontsize=12, weight="bold", color=BLACK)
    ax.text(17.8, 8.48, "Causal direct multi-step forecasting", fontsize=8.5,
            color=GRAY, ha="right")

    rounded_box(ax, 0.25, 6.25, 1.65, 1.15,
                "Power history\n$\\mathbf{x}\\in\\mathbb{R}^{156\\times1}$\n(15.6 s at 10 Hz)",
                BLUE_LIGHT, BLUE, 8.3, weight="bold")
    rounded_box(ax, 2.2, 6.38, 1.35, 0.89,
                "Input projection\nConv1D, $k=1$\n32 channels",
                BLUE_LIGHT, BLUE, 8.0)

    # Residual stack container and six dilation blocks.
    stack = FancyBboxPatch((3.9, 5.90), 4.55, 1.80,
                           boxstyle="round,pad=0.05,rounding_size=0.14",
                           linewidth=1.35, edgecolor=BLUE, facecolor="#F8FAFD", zorder=1)
    ax.add_patch(stack)
    ax.text(6.175, 7.48, "Causal FiLM residual stack", ha="center", va="center",
            fontsize=9.2, weight="bold", color=BLACK)
    dilations = [1, 2, 4, 8, 16, 32]
    for idx, dilation in enumerate(dilations):
        x = 4.10 + idx * 0.69
        rounded_box(ax, x, 6.15, 0.52, 1.03,
                    f"Block\n{idx + 1}\n$d={dilation}$", BLUE_LIGHT, BLUE,
                    7.1, linewidth=1.0, radius=0.07)
        if idx < len(dilations) - 1:
            arrow(ax, (x + 0.52, 6.665), (x + 0.69, 6.665), BLUE,
                  linewidth=1.0, mutation=8)
    ax.text(6.175, 5.72, "2 causal convolutions per block; theoretical receptive field = 253 samples",
            ha="center", va="center", fontsize=7.4, color=GRAY)

    rounded_box(ax, 8.78, 6.38, 1.35, 0.89,
                "Final causal state\n$\\mathbf{h}_{156}\\in\\mathbb{R}^{32}$",
                BLUE_LIGHT, BLUE, 8.0)
    rounded_box(ax, 10.48, 6.38, 1.05, 0.89,
                "Concatenate\n$[\\mathbf{h}_{156};\\mathbf{e}_c]$",
                PURPLE_LIGHT, PURPLE, 7.8)
    rounded_box(ax, 11.88, 6.38, 1.05, 0.89,
                "Dense 64\nSwish",
                PURPLE_LIGHT, PURPLE, 8.2)
    rounded_box(ax, 13.28, 6.30, 1.25, 1.05,
                "Dense 48\nreshape\n$16\\times3$ logits",
                PURPLE_LIGHT, PURPLE, 8.0)
    rounded_box(ax, 14.88, 6.30, 1.27, 1.05,
                "Monotone\nordered-output\nmapping",
                GREEN_LIGHT, GREEN, 8.0)
    rounded_box(ax, 16.50, 6.17, 1.25, 1.31,
                "Forecast\n$\\hat{\\mathbf{Y}}\\in[0,1]^{16\\times3}$\nP10 / P50 / P90\n(1.6 s)",
                GREEN_LIGHT, GREEN, 7.9, weight="bold")

    for start, end in [
        ((1.90, 6.825), (2.20, 6.825)),
        ((3.55, 6.825), (3.90, 6.825)),
        ((8.45, 6.825), (8.78, 6.825)),
        ((10.13, 6.825), (10.48, 6.825)),
        ((11.53, 6.825), (11.88, 6.825)),
        ((12.93, 6.825), (13.28, 6.825)),
        ((14.53, 6.825), (14.88, 6.825)),
        ((16.15, 6.825), (16.50, 6.825)),
    ]:
        arrow(ax, start, end, BLUE if end[0] <= 10.13 else PURPLE,
              linewidth=1.35, mutation=10)

    # Context pathway.
    rounded_box(ax, 0.55, 4.20, 2.30, 1.10,
                "Physical context $\\mathbf{c}\\in\\mathbb{R}^{8}$\n$H_s, T_e, T_p, \\omega_0/\\omega_p, \\zeta,$\n$L, b_{eff}, P_{rated}$",
                ORANGE_LIGHT, ORANGE, 8.2, weight="bold")
    rounded_box(ax, 3.30, 4.28, 1.70, 0.94,
                "Context MLP\nDense 32 + Swish\nDense 32 + Swish",
                ORANGE_LIGHT, ORANGE, 8.0)
    ax.text(1.70, 3.96, "Independently varying here: $H_s$, $T_e$, $\\omega_0/\\omega_p$, and $\\zeta$",
            ha="center", va="center", fontsize=7.5, color=GRAY)
    arrow(ax, (2.85, 4.75), (3.30, 4.75), ORANGE, linewidth=1.4)

    # FiLM bus: one injection into every residual block.
    ax.plot([4.98, 8.20], [5.45, 5.45], color=ORANGE, lw=1.45, zorder=1)
    arrow(ax, (5.00, 4.75), (5.00, 5.45), ORANGE, linewidth=1.4)
    for idx in range(6):
        xmid = 4.10 + idx * 0.69 + 0.26
        arrow(ax, (xmid, 5.45), (xmid, 6.15), ORANGE, linewidth=1.05, mutation=8)
    ax.text(8.30, 5.44, "FiLM at both convolutions\nin every residual block",
            ha="left", va="center", fontsize=7.7, color=ORANGE)

    # Context embedding also enters the prediction head.
    arrow(ax, (5.00, 4.68), (10.99, 6.38), ORANGE, linewidth=1.25,
          connection="angle3,angleA=0,angleB=90", mutation=9)

    ax.text(17.78, 5.66, "72,048 trainable parameters", ha="right", va="center",
            fontsize=7.8, color=GRAY)

    # Divider.
    ax.plot([0.2, 17.8], [3.62, 3.62], color="#D0D5DD", lw=0.9)

    # Panel (b): internals of one residual block.
    ax.text(0.2, 3.28, "(b) One causal FiLM residual block", fontsize=12,
            weight="bold", color=BLACK)

    rounded_box(ax, 0.45, 1.60, 1.05, 0.74, "Block input\n$\\mathbf{H}$", GRAY_LIGHT, GRAY, 8.2)
    rounded_box(ax, 1.92, 1.55, 1.15, 0.84, "Causal Conv1D\n$k=3$, dilation $d$\n32 channels", BLUE_LIGHT, BLUE, 7.7)
    rounded_box(ax, 3.43, 1.67, 0.72, 0.60, "Layer\nNorm", BLUE_LIGHT, BLUE, 7.5)
    rounded_box(ax, 4.52, 1.67, 0.74, 0.60, "FiLM", ORANGE_LIGHT, ORANGE, 8.0, weight="bold")
    rounded_box(ax, 5.63, 1.60, 1.05, 0.74, "Swish\nDropout 0.10", BLUE_LIGHT, BLUE, 7.6)
    rounded_box(ax, 7.05, 1.55, 1.15, 0.84, "Causal Conv1D\n$k=3$, dilation $d$\n32 channels", BLUE_LIGHT, BLUE, 7.7)
    rounded_box(ax, 8.56, 1.67, 0.72, 0.60, "Layer\nNorm", BLUE_LIGHT, BLUE, 7.5)
    rounded_box(ax, 9.65, 1.67, 0.74, 0.60, "FiLM", ORANGE_LIGHT, ORANGE, 8.0, weight="bold")
    rounded_box(ax, 10.76, 1.60, 1.05, 0.74, "Swish\nDropout 0.10", BLUE_LIGHT, BLUE, 7.6)

    add = Circle((12.35, 1.97), 0.27, edgecolor=BLUE, facecolor="white", lw=1.3, zorder=2)
    ax.add_patch(add)
    ax.text(12.35, 1.97, "+", ha="center", va="center", fontsize=13,
            color=BLUE, weight="bold", zorder=3)
    rounded_box(ax, 12.93, 1.60, 1.03, 0.74, "Block output", GRAY_LIGHT, GRAY, 8.0)

    sequential = [
        ((1.50, 1.97), (1.92, 1.97)), ((3.07, 1.97), (3.43, 1.97)),
        ((4.15, 1.97), (4.52, 1.97)), ((5.26, 1.97), (5.63, 1.97)),
        ((6.68, 1.97), (7.05, 1.97)), ((8.20, 1.97), (8.56, 1.97)),
        ((9.28, 1.97), (9.65, 1.97)), ((10.39, 1.97), (10.76, 1.97)),
        ((11.81, 1.97), (12.08, 1.97)), ((12.62, 1.97), (12.93, 1.97)),
    ]
    for start, end in sequential:
        arrow(ax, start, end, BLUE, linewidth=1.15, mutation=9)

    # Residual shortcut.
    ax.plot([1.50, 1.68, 1.68, 11.98], [1.97, 1.97, 2.82, 2.82], color=GRAY, lw=1.15)
    arrow(ax, (11.98, 2.82), (12.35, 2.24), GRAY, linewidth=1.15,
          connection="arc3,rad=-0.12", mutation=9)
    ax.text(6.80, 2.97, "identity shortcut (or $1\\times1$ projection if channel dimensions differ)",
            ha="center", va="center", fontsize=7.5, color=GRAY)

    # Context injection below the block.
    rounded_box(ax, 5.90, 0.35, 2.15, 0.72,
                "Context embedding $\\mathbf{e}_c$\n(shared across time steps)",
                ORANGE_LIGHT, ORANGE, 8.0)
    arrow(ax, (6.45, 1.07), (4.89, 1.67), ORANGE, linewidth=1.15,
          connection="arc3,rad=0.08", mutation=9)
    arrow(ax, (7.50, 1.07), (10.02, 1.67), ORANGE, linewidth=1.15,
          connection="arc3,rad=-0.08", mutation=9)

    # FiLM definition and ablation note.
    rounded_box(ax, 14.45, 1.56, 3.05, 1.14,
                "Feature-wise linear modulation\n$[\\boldsymbol{\\gamma},\\boldsymbol{\\beta}]=\\mathrm{Linear}(\\mathbf{e}_c)$\n$\\widetilde{\\mathbf{H}}=\\mathbf{H}\\odot(1+\\boldsymbol{\\gamma})+\\boldsymbol{\\beta}$",
                ORANGE_LIGHT, ORANGE, 8.2, weight="normal")
    rounded_box(ax, 14.45, 0.30, 3.05, 0.86,
                "TCN ablation: FiLM disabled;\ncontext zeroed before concatenation\n(46,704 parameters)",
                GRAY_LIGHT, GRAY, 7.2)

    ax.text(17.8, 0.05,
            "All temporal convolutions are causal; the 16-step horizon is predicted in one forward pass.",
            ha="right", va="bottom", fontsize=7.5, color=GRAY)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUT_DIR / "pctcn_architecture"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.08)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.08)
    fig.savefig(stem.with_suffix(".png"), dpi=400, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"Saved: {stem.with_suffix('.svg')}")
    print(f"Saved: {stem.with_suffix('.pdf')}")
    print(f"Saved: {stem.with_suffix('.png')}")


if __name__ == "__main__":
    main()
