"""Build factual tables and plots from saved benchmark outputs; never retrain."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()
    root = args.results
    runs = json.loads((root / "runs.json").read_text())
    suite = json.loads((root / "suite.json").read_text())
    successful = [run for run in runs if run["status"] == "completed" and run.get("summary", {}).get("tokens_per_second")]
    if not successful:
        raise RuntimeError("No successful measured experiments")
    baseline = next(run for run in successful if run["name"] == "baseline")["summary"]
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 150, "savefig.dpi": 160})
    labels = [run["name"] for run in successful]
    colors = ["#27754f" if name == "final" else "#616b75" if name == "baseline" else "#397bb1" for name in labels]
    fig, axes = plt.subplots(1, 2, figsize=(13, max(5, len(labels) * 0.42)), sharey=True)
    for ax, metric, title in zip(axes, ("tokens_per_second", "peak_allocated_gib"),
                                 ("Полезные токены/с (после 10 шагов прогрева)", "Пиковая выделенная GPU-память, GiB")):
        values = [run["summary"][metric] for run in successful]
        bars = ax.barh(labels, values, color=colors)
        ax.bar_label(bars, fmt="%.0f" if metric == "tokens_per_second" else "%.2f", padding=4)
        ax.set_xlabel(title)
        ax.set_xlim(0, max(values) * 1.20)
        ax.grid(axis="x", alpha=0.18)
        ax.set_axisbelow(True)
    axes[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(figures / "speed_memory.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for run in successful:
        history_path = root / "experiments" / run["name"] / "loss_history.json"
        if not history_path.exists():
            continue
        history = json.loads(history_path.read_text())
        history = [point for point in history if point.get("optimizer_step") is not None]
        if not history:
            continue
        linewidth = 2.4 if run["name"] in ("baseline", "final") else 1.1
        for ax, x in zip(axes, ("elapsed_seconds", "optimizer_step")):
            ax.plot([point[x] for point in history], [point["loss"] for point in history],
                    label=run["name"], linewidth=linewidth)
    for ax, xlabel in zip(axes, ("Время обучения, с", "Шаг оптимизатора")):
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Train loss (логирование каждые 10 шагов)")
        ax.grid(alpha=0.18)
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "train_loss.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, max(5, len(labels) * 0.42)))
    for i, run in enumerate(successful):
        summary = run["summary"]
        ax.scatter(summary["validation_loss"], i, color=colors[i], s=55)
        ax.annotate(f"{summary['validation_loss']:.4f}; {summary['train_seconds']:.1f} с",
                    (summary["validation_loss"], i), xytext=(8, 0), textcoords="offset points", va="center")
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlabel("Validation loss; рядом указана фактическая длительность обучения")
    ax.margins(x=0.3)
    ax.grid(axis="x", alpha=0.18)
    fig.tight_layout()
    fig.savefig(figures / "validation_loss.png", bbox_inches="tight")
    plt.close(fig)

    lines = ["| Запуск | Статус | Лимит, с | Факт, с | Токены/с | S к baseline | Peak allocated, GiB | Peak reserved, GiB | Validation loss | Шаги |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]

    def show(value, precision=2):
        return "—" if value is None else f"{value:.{precision}f}"

    for run in runs:
        summary = run.get("summary", {})
        speed = summary.get("tokens_per_second")
        ratio = speed / baseline["tokens_per_second"] if speed else None
        lines.append(f"| `{run['name']}` | {run['status']} | {run['config']['seconds']} | "
                     f"{show(summary.get('train_seconds'))} | {show(speed, 1)} | {show(ratio, 3)} | "
                     f"{show(summary.get('peak_allocated_gib'))} | {show(summary.get('peak_reserved_gib'))} | "
                     f"{show(summary.get('validation_loss'), 5)} | {summary.get('optimizer_steps', '—')} |")
    (root / "metrics.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(json.dumps({"suite_status": suite["status"], "speedup": suite.get("speedup")}, indent=2))


if __name__ == "__main__":
    main()
