"""Plot paired test deltas to the reference LoRA from analyze.py."""

import argparse
import csv
from pathlib import Path
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=Path("results"))
    args = p.parse_args()
    with (args.results / "paired_intervals.csv").open() as f:
        rows = [r for r in csv.DictReader(f) if r["reference"] == "lora_ref"]
    labels = {
        "lora_r32": "Ранг 8 → 32",
        "lora_alpha64": "Alpha 16 → 64",
        "lora_drop15": "Dropout 0,05 → 0,15",
        "lora_all": "q/v → 7 проекций",
        "lora_r32_alpha64": "Ранг 32 и alpha 64",
    }
    names = [k for k in labels if any(r["run"] == k for r in rows)]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for ax, metric, title, color in zip(
        axes,
        ("ndcg_at_10", "macro_f1"),
        ("nDCG@10", "Macro-F1"),
        ("#175b8e", "#6a4690"),
    ):
        ax.axvline(0, color="#778492", linewidth=1, linestyle="--")
        for i, name in enumerate(names):
            r = next(r for r in rows if r["run"] == name and r["metric"] == metric)
            val, lo, hi = [
                100 * float(r[k]) for k in ("difference", "ci_low", "ci_high")
            ]
            ax.hlines(i, lo, hi, color=color, lw=2.4)
            ax.plot([lo, hi], [i, i], "|", color=color, markersize=8)
            ax.plot(val, i, "o", color=color, markersize=6)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(axis="x", alpha=0.2)
        ax.set_xlabel("Изменение, пункты шкалы 0–100")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(list(range(len(names))), [labels[k] for k in names])
    axes[0].invert_yaxis()
    fig.suptitle(
        "Изменение относительно опорной LoRA", x=0.04, ha="left", fontweight="bold"
    )
    fig.text(
        0.04,
        0.02,
        "Точки показывают разности на тесте. Отрезки — 95% интервалы парного бутстрапа по запросам.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    fig.savefig(args.results / "lora_effects.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
