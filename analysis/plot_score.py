import json
import itertools
import importlib
import os
import re
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path

try:
    ttest_ind = getattr(importlib.import_module("scipy.stats"), "ttest_ind")
except ImportError:
    ttest_ind = None

# ===============================
#        CONFIG
# ===============================

ANALYSIS_DIR = Path(__file__).resolve().parent
base_path = ANALYSIS_DIR / "bundles"
DEFAULT_BUNDLE_PREFIX = ""
os.environ.setdefault("MPLCONFIGDIR", str(ANALYSIS_DIR.parent / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ANALYSIS_DIR.parent / ".cache"))
(ANALYSIS_DIR.parent / ".matplotlib").mkdir(exist_ok=True)
(ANALYSIS_DIR.parent / ".cache").mkdir(exist_ok=True)
(ANALYSIS_DIR.parent / ".cache" / "fontconfig").mkdir(exist_ok=True)
SPEED_RE = re.compile(
    r"\[p(?P<p1pos>\([^]]+\))_cs\((?P<p1cs>[^)]+)\)_ws\((?P<p1ws>[^)]+)\)\]_"
    r"\[p(?P<p2pos>\([^]]+\))_cs\((?P<p2cs>[^)]+)\)_ws\((?P<p2ws>[^)]+)\)\]"
)


def _num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _same(a, b):
    return a is not None and abs(a - b) < 1e-9


def condition_label(folder_name):
    match = SPEED_RE.search(folder_name)
    prefix = "c_" if folder_name.endswith("_with_collision") else "nc_"
    task = "enc" if folder_name.startswith("encouraged") else "bl"

    if not match:
        return folder_name

    p1cs = _num(match.group("p1cs"))
    p1ws = _num(match.group("p1ws"))
    p2cs = _num(match.group("p2cs"))
    p2ws = _num(match.group("p2ws"))

    if _same(p1ws, 0.7) and _same(p2cs, 0.4):
        speed = "optimal_p1run_p2cut"
    elif _same(p1cs, 0.4) and _same(p2ws, 0.7):
        speed = "optimal_p1cut_p2run"
    elif _same(p1cs, 1.0) and _same(p1ws, 1.0) and _same(p2cs, 1.0) and _same(p2ws, 1.0):
        speed = "superstar"
    elif _same(p1cs, 1.0) and _same(p1ws, 0.4) and _same(p2cs, 0.2) and _same(p2ws, 1.0):
        speed = "mixed"
    elif _same(p1cs, 1.0) and _same(p1ws, 0.2) and _same(p2cs, 1.0) and _same(p2ws, 1.0):
        speed = "asymetric_slow_walker"
    elif _same(p1cs, 1.0) and _same(p1ws, 0.1) and _same(p2cs, 0.1) and _same(p2ws, 1.0):
        speed = "mixed_extreme"
    else:
        speed = f"p1_cs{p1cs}_ws{p1ws}__p2_cs{p2cs}_ws{p2ws}"

    return f"{prefix}{task}_{speed}"


def condition_speed_label(folder_name):
    label = condition_label(folder_name)
    return label.split("_", 2)[2] if label.startswith(("c_", "nc_")) and label.count("_") >= 2 else label


def discover_folders():
    if not base_path.is_dir():
        return {}
    folders = {}
    bundle_prefix = os.getenv("BUNDLE_PREFIX", DEFAULT_BUNDLE_PREFIX)
    include_encouraged = os.getenv("INCLUDE_ENCOURAGED", "1") == "1"
    collision_only = os.getenv("COLLISION_ONLY", "1") == "1"
    allowed_conditions = {
        item.strip()
        for item in os.getenv("ALLOWED_CONDITIONS", "mixed,optimal_p1run_p2cut,optimal_p1cut_p2run,superstar,asymetric_slow_walker,mixed_extreme").split(",")
        if item.strip()
    }
    for folder in sorted(p for p in base_path.iterdir() if p.is_dir()):
        if bundle_prefix and not folder.name.startswith(bundle_prefix):
            continue
        if not include_encouraged and folder.name.startswith("encouraged"):
            continue
        if collision_only and not folder.name.endswith("_with_collision"):
            continue
        label = condition_label(folder.name)
        if allowed_conditions and condition_speed_label(folder.name) not in allowed_conditions:
            continue
        folders[label] = folder
    return folders


# ===============================
#      LOAD SCORES
# ===============================

def load_scores(folder_path):
    scores = []
    if not folder_path.is_dir():
        return scores
    for sub in sorted(p for p in folder_path.iterdir() if p.is_dir()):
        meta = sub / "meta.json"
        if meta.is_file():
            try:
                with open(meta) as f:
                    j = json.load(f)
                if "final_score" in j:
                    scores.append(j["final_score"])
            except Exception:
                pass
    return scores


def main():
    parser = ArgumentParser()
    parser.add_argument("--output", default=str(ANALYSIS_DIR / "score_distributions.png"))
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    matplotlib = importlib.import_module("matplotlib")
    matplotlib.use("Agg")
    plt = importlib.import_module("matplotlib.pyplot")

    folders = discover_folders()
    all_scores = {name: load_scores(path) for name, path in folders.items()}
    all_scores = {name: vals for name, vals in all_scores.items() if vals}

    if not all_scores:
        raise SystemExit("No bundle scores found. Run create_bundles.py first.")

    # ===============================
    #      PRINT MEANS + T-TESTS
    # ===============================

    print("\n=== GROUP MEANS ===")
    for name, vals in all_scores.items():
        print(f"{name:35s} mean = {sum(vals) / len(vals):.2f}   n={len(vals)}")

    print("\n=== ALL PAIRWISE T-TESTS ===")
    if ttest_ind is None:
        print("Skipped: scipy is not installed.")
    for a, b in itertools.combinations(all_scores.keys(), 2):
        if ttest_ind is None:
            continue
        if len(all_scores[a]) > 1 and len(all_scores[b]) > 1:
            t, p = ttest_ind(all_scores[a], all_scores[b], equal_var=False)
            print(f"{a:35s} vs {b:35s}   t={t:.3f}, p={p:.5f}")
        else:
            print(f"{a:35s} vs {b:35s}   NOT ENOUGH DATA")

    # ===============================
    #      PLOTTING (FULLY DYNAMIC)
    # ===============================

    ordered_keys = sorted(all_scores.keys())
    data = [all_scores[k] for k in ordered_keys]
    labels = ordered_keys

    plt.figure(figsize=(max(12, len(labels) * 1.4), 6))

    parts = plt.violinplot(data, showmeans=False, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor('#8da0cb')
        pc.set_alpha(0.75)
        pc.set_edgecolor("black")

    dot_spacing = 0.03
    dot_size = 40

    for xi, vals in enumerate(data, start=1):
        counts = Counter(vals)
        for score, count in sorted(counts.items()):
            offsets = [(i - (count - 1) / 2) * dot_spacing for i in range(count)]
            x_positions = [xi + off for off in offsets]
            plt.scatter(
                x_positions,
                [score] * count,
                color="gray",
                edgecolor="black",
                s=dot_size,
                zorder=3,
                linewidths=0.3,
            )

    for xi, vals in enumerate(data, start=1):
        plt.scatter(
            xi,
            sum(vals) / len(vals),
            color="magenta",
            marker="D",
            s=130,
            edgecolor="black",
            zorder=4,
        )

    plt.xticks(range(1, len(labels) + 1), labels, rotation=45, ha="right")
    plt.ylabel("Final Score")
    plt.title("Final Score Distributions Across All Conditions")
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.output, dpi=200)
    print(f"\nSaved plot to {args.output}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
