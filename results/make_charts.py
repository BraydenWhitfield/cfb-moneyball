"""Make the results/ charts from the saved outputs in data/processed/.

    python results/make_charts.py

Needs: python -m cfb.master, python -m cfb.modeling.run,
python -m cfb.modeling.prior_plays, python -m cfb.modeling.group_analysis.
Colors: validated two-slot categorical palette (blue, orange) on a light
surface; gray for context marks.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

from cfb.modeling.data import load_master, population, split, test_variants  # noqa: E402
from cfb.modeling.evaluate import evaluate  # noqa: E402
from cfb.modeling.groups import eb_shrink, persistence, power_curve  # noqa: E402
from cfb.processed import PROCESSED_DIR  # noqa: E402

OUT = Path(__file__).parent
MODELS = PROCESSED_DIR / "models"

SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#b9b8b2"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10.5, "text.color": INK,
    "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.titlesize": 12.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "figure.dpi": 150,
})


def title(fig, text, sub):
    fig.suptitle(text, x=0.02, ha="left", fontsize=14, fontweight="bold", color=INK)
    fig.text(0.02, 0.905, sub, ha="left", fontsize=10, color=INK_2)


def bar(ax, x, h, w, color):
    """Bar with a rounded data end and a square baseline."""
    r = min(w * 0.18, abs(h) / 2) if h else 0
    y0 = min(0, h)
    ax.add_patch(FancyBboxPatch((x - w / 2, y0), w, abs(h), boxstyle=f"round,pad=0,rounding_size={r}",
                                linewidth=0, facecolor=color, mutation_aspect=1))
    base = 0 if h >= 0 else h / 2
    ax.add_patch(Rectangle((x - w / 2, base), w, abs(h) / 2, linewidth=0, facecolor=color))


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("wrote", name)


# 1. Mean value by star tier --------------------------------------------------------

def chart_tiers(master):
    tiers = ["2", "3", "4", "5", "unrated"]
    labels = ["2-star", "3-star", "4-star", "5-star", "Unrated"]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    w = 0.34
    for i, (kind, color, name) in enumerate([("recruit", BLUE, "Recruits"),
                                             ("transfer", ORANGE, "Transfers")]):
        d = population(master, kind)
        g = d.groupby("market_tier").par_window.agg(["mean", "size"]).reindex(tiers)
        for j, (m, n) in enumerate(zip(g["mean"], g["size"])):
            x = j + (i - 0.5) * (w + 0.04)
            bar(ax, x, m, w, color)
            ax.text(x, m + 1.5, f"{m:.0f}", ha="center", va="bottom", fontsize=9, color=INK)
            ax.text(x, -4.5, f"n={n}", ha="center", va="top", fontsize=7.5, color=MUTED)
        ax.plot([], [], color=color, linewidth=8, label=name)
    ax.set_xticks(range(len(tiers)), labels)
    ax.set_xlim(-0.6, len(tiers) - 0.4)
    ax.set_ylim(-9, 95)
    ax.axhline(0, color=INK_2, linewidth=0.8)
    ax.set_ylabel("Mean value (points above replacement)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")
    title(fig, "Stars predict value on average",
          "Mean value over the first 3 seasons (recruits who signed with a school) or "
          "2 seasons (transfers)")
    fig.subplots_adjust(top=0.84)
    save(fig, "1_value_by_tier.png")


# 2. Rating vs value ------------------------------------------------------------------

def chart_rating(master):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.4), sharey=True)
    for ax, kind, color, name in [(axes[0], "recruit", BLUE, "Recruits"),
                                  (axes[1], "transfer", ORANGE, "Transfers")]:
        d = population(master, kind)
        d = d[d.rating.notna()]
        rho = d.rating.rank().corr(d.par_window.rank())
        ax.scatter(d.rating, d.par_window.clip(-60, 250), s=9, color=color, alpha=0.25,
                   linewidths=0)
        bins = pd.qcut(d.rating, 10, duplicates="drop")
        med = d.groupby(bins, observed=True).agg(r=("rating", "median"), v=("par_window", "mean"))
        ax.plot(med.r, med.v, color=INK, linewidth=2, marker="o", markersize=5,
                label="Mean of each rating decile")
        ax.set_title(f"{name}  (rank correlation {rho:.2f}, n={len(d):,})", fontsize=11)
        ax.set_xlabel("Market rating (247 composite / portal rating)")
        ax.axhline(0, color=INK_2, linewidth=0.8)
    axes[0].set_ylabel("Value over the window (clipped at 250)")
    axes[0].legend(loc="upper left", fontsize=9)
    title(fig, "Rating explains only a little of who produces",
          "Each dot is one recruit or transfer; most players produce little, a few produce a lot")
    fig.subplots_adjust(top=0.8)
    save(fig, "2_rating_vs_value.png")


# 3. Full model vs baseline+ -----------------------------------------------------------

def chart_models(master):
    preds = pd.read_parquet(MODELS / "predictions.parquet")
    rows = []
    for kind in ["recruit", "transfer"]:
        df = population(master, kind)
        train, test = split(df, kind)
        p = preds[preds.acquisition_type.eq(kind)].set_index("acquisition_id")
        variants = test_variants(train, test, kind)
        if kind == "transfer":
            variants = {k: variants[k] for k in ["all", "matched_only"]}
        for variant, rows_ in variants.items():
            names = [f"{l}:{s}" for l in ["ridge", "lgbm"] for s in ["baseline_plus", "full"]]
            sub = {n: p.loc[rows_.acquisition_id, n].to_numpy() for n in names}
            _, diffs = evaluate(rows_.reset_index(drop=True), sub, n_boot=1000,
                                references=("baseline_plus",))
            for r in diffs.itertuples():
                rows.append({"kind": kind, "variant": variant, "learner": r.model.split(":")[0],
                             "metric": r.metric, "diff": r.diff, "lo": r.lo, "hi": r.hi})
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "model_vs_baseline_plus.csv", index=False)
    labels = {("recruit", "all"): "Recruits (2023 class)",
              ("transfer", "all"): "Transfers (2024), all",
              ("transfer", "matched_only"): "Transfers (2024), matched only"}
    metrics = [("spearman", "Pooled rank correlation"),
               ("spearman_by_pos", "Rank correlation within position")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    for ax, (metric, mtitle) in zip(axes, metrics):
        y = 0
        ticks, ticklabels = [], []
        for key, label in labels.items():
            for learner, color, off in [("ridge", BLUE, 0.15), ("lgbm", ORANGE, -0.15)]:
                r = d[(d.kind == key[0]) & (d.variant == key[1]) & (d.learner == learner)
                      & (d.metric == metric)].iloc[0]
                ax.plot([r.lo, r.hi], [y + off] * 2, color=color, linewidth=2,
                        solid_capstyle="round")
                ax.plot(r["diff"], y + off, "o", color=color, markersize=7,
                        markeredgecolor=SURFACE, markeredgewidth=1.5)
            ticks.append(y)
            ticklabels.append(label)
            y -= 1
        ax.axvline(0, color=INK_2, linewidth=1)
        ax.set_ylim(y + 0.5, 0.9)
        ax.set_title(mtitle, fontsize=11)
        ax.set_xlabel("Full model minus baseline+ (95% CI)")
        ax.set_yticks(ticks, ticklabels)
        ax.grid(axis="y", visible=False)
    axes[0].plot([], [], "o-", color=BLUE, label="Ridge")
    axes[0].plot([], [], "o-", color=ORANGE, label="LightGBM")
    axes[0].legend(loc="upper right", fontsize=9)
    title(fig, "Extra signing-day information doesn't beat the market out of time",
          "Baseline+ = rating + position + level of the signing school. "
          "CIs that cross 0 mean no reliable gain")
    fig.subplots_adjust(top=0.8, wspace=0.08)
    save(fig, "3_full_vs_baseline_plus.png")


# 4. QB prior plays ----------------------------------------------------------------------

def chart_prior():
    res = pd.read_parquet(MODELS / "prior_plays_results.parquet").reset_index()
    res = res[res.test_rows.eq("origin_matched")]
    checks = list(dict.fromkeys(res.check))
    short = ["2024 test, 2-season value\n(primary)", "2024 test, 1-season value\n(secondary)",
             "2025 test, 1-season value\n(secondary)"]
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    for i, check in enumerate(checks):
        for pos, off in [("QB", 0.0), ("RB", 0.22), ("WR", 0.44)]:
            r = res[(res.check == check) & (res.position == pos)].iloc[0]
            color = ORANGE if pos == "QB" else GRAY
            x = i + off - 0.22
            ax.plot([x, x], [r.gain_lo, r.gain_hi], color=color, linewidth=2,
                    solid_capstyle="round")
            ax.plot(x, r.gain, "o", color=color, markersize=7, markeredgecolor=SURFACE,
                    markeredgewidth=1.5)
            ax.text(x, r.gain_hi + 0.012, pos, ha="center", fontsize=8.5,
                    color=INK if pos == "QB" else MUTED,
                    fontweight="bold" if pos == "QB" else "normal")
    ax.axhline(0, color=INK_2, linewidth=1)
    ax.set_xticks(range(len(checks)), short, fontsize=9)
    ax.set_ylabel("Gain in rank correlation from\nadding prior plays (95% CI)")
    ax.grid(axis="x", visible=False)
    title(fig, "Prior playing time helps rank QB transfers - suggestive, not proven",
          "Within position, controlling for portal rating, stars and destination level; "
          "transfers matched at their old school")
    fig.subplots_adjust(top=0.8)
    save(fig, "4_qb_prior_plays.png")


# 5 & 6. Group persistence and power -------------------------------------------------------

def chart_groups():
    df = pd.read_parquet(MODELS / "group_residuals.parquet")
    train, test = df[df.split.eq("train")], df[df.split.eq("test")]
    eff = eb_shrink(train.resid_w, train.coach)
    later = test.dropna(subset=["coach"]).groupby("coach").resid_w.agg(
        mean_2023="mean", size_2023="size")
    g = eff.join(later, how="inner")
    stat = persistence(train, test, "coach", n_boot=1000)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    other = g.drop(index="Andy Avalos", errors="ignore")
    ax.scatter(other.shrunk, other.mean_2023, s=other.size_2023 * 6, color=GRAY, alpha=0.7,
               linewidths=0)
    if "Andy Avalos" in g.index:
        b = g.loc["Andy Avalos"]
        ax.scatter(b.shrunk, b.mean_2023, s=110, color=ORANGE,
                   edgecolors=SURFACE, linewidths=2, zorder=3)
        ax.annotate(f"Boise State (Andy Avalos)\n2021-22: +{b['mean']:.0f} raw, "
                    f"+{b.shrunk:.1f} after shrinkage\n2023: {b.mean_2023:+.1f} ({int(b.size_2023)} recruits)",
                    (b.shrunk, b.mean_2023), xytext=(4.6, 15), textcoords="data",
                    fontsize=9, color=INK,
                    arrowprops=dict(arrowstyle="-", color=INK_2, linewidth=1))
    ax.axhline(0, color=INK_2, linewidth=0.8)
    ax.axvline(0, color=INK_2, linewidth=0.8)
    ax.set_xlabel("Coach effect from 2021-22 classes (shrunk toward 0)")
    ax.set_ylabel("Same coach's 2023 class: mean residual")
    title(fig, "Last year's over-performers don't repeat",
          f"FBS head coaches; dot size = 2023 recruits. Correlation {stat['corr']:+.3f} "
          f"(95% CI {stat['corr_lo']:+.2f} to {stat['corr_hi']:+.2f})")
    fig.subplots_adjust(top=0.84)
    save(fig, "5_group_persistence.png")

    taus = (1, 2, 3, 5, 8)   # same grid and seed as groups_report.txt
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for key, color, label in [("school", BLUE, "School"), ("coach", ORANGE, "Head coach"),
                              ("school_pos", INK_2, "School x position")]:
        sigma2 = eb_shrink(train.resid_w, train[key]).attrs["sigma2"]
        pc = power_curve(train, test, key, sigma2, taus=taus, n_sim=300)
        ax.plot(pc.tau, 100 * pc.power, "o-", color=color, linewidth=2, markersize=6,
                label=label, markeredgecolor=SURFACE, markeredgewidth=1.2)
    ax.axhline(80, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(1.0, 82, "80% detection", fontsize=8.5, color=MUTED)
    ax.set_xlabel("True size of group effects (SD, in points of value)")
    ax.set_ylabel("Chance the test detects them (%)")
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right")
    title(fig, "Big group effects would have shown up",
          "Simulated with the real group sizes: effects of ~5+ points are ruled out, "
          "small ones are not")
    fig.subplots_adjust(top=0.84)
    save(fig, "6_power_curve.png")


if __name__ == "__main__":
    master = load_master()
    chart_tiers(master)
    chart_rating(master)
    chart_models(master)
    chart_prior()
    chart_groups()
