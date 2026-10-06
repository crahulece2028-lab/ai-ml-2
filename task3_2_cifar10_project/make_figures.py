"""Regenerates the standalone artifact figures for task 3.2 from the stored CSVs.

Every figure here is derived from an artifact that training already wrote, so it costs a second and
needs neither the model nor the dataset. The two figures that do need inference
(`cifar10_sample_predictions.png`, `cifar10_misclassified.png`) are behind `--predictions` and
should be run when the CPU is free.

    python make_figures.py                 # CSV-derived figures
    python make_figures.py --predictions   # plus the two inference-based figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

ACCENT = "#1F3A5F"
WARM = "#C1666B"
GRID = {"color": "#DDE3EA", "linewidth": 0.6}
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 10, "axes.titleweight": "bold"})


def load(name: str, index_col=None) -> pd.DataFrame:
    path = ART / name
    if not path.exists():
        raise SystemExit(f"missing artifact: {path}\nRun train_cifar10.py first.")
    return pd.read_csv(path, index_col=index_col)


def save(fig, name: str) -> None:
    out = ART / name
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out.name}")


# ------------------------------------------------------------------ training curves
def training_curves() -> None:
    h = load("cifar10_training_history.csv")
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.3))

    ax = axes[0]
    ax.plot(h["epoch"], h["accuracy"], color=WARM, lw=1.8, label="train")
    ax.plot(h["epoch"], h["val_accuracy"], color=ACCENT, lw=1.8, label="validation")
    best = int(h.loc[h["val_accuracy"].idxmax(), "epoch"])
    ax.plot(best, h["val_accuracy"].max(), "o", ms=6, mfc="none",
            mec=ACCENT, mew=1.8)
    ax.annotate(f"best {h['val_accuracy'].max():.4f}\n(epoch {best})",
                (best, h["val_accuracy"].max()), textcoords="offset points",
                xytext=(6, -22), fontsize=7.5, color=ACCENT)
    ax.set_title("Accuracy")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    ax.plot(h["epoch"], h["loss"], color=WARM, lw=1.8, label="train")
    ax.plot(h["epoch"], h["val_loss"], color=ACCENT, lw=1.8, label="validation")
    ax.set_title("Loss (label-smoothed cross-entropy)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    if "epoch_seconds" in h:
        ax.plot(h["epoch"], h["epoch_seconds"], color="#7A8B99", lw=1.5, marker="o", ms=3)
        ax.axhline(h["epoch_seconds"].median(), color=ACCENT, ls="--", lw=1.2)
        ax.annotate(f"median {h['epoch_seconds'].median():.0f}s",
                    (h["epoch"].iloc[-1], h["epoch_seconds"].median()),
                    textcoords="offset points", xytext=(-58, 8), fontsize=7.5, color=ACCENT)
    ax.set_title("Seconds per epoch")
    ax.set_xlabel("epoch")
    ax.set_ylabel("seconds")
    ax.set_ylim(bottom=0)

    for ax in axes:
        ax.grid(True, axis="y", **GRID)
    fig.suptitle("Task 3.2 - CIFAR-10 training history (augmented run)", y=1.04,
                 fontsize=11, fontweight="bold")
    save(fig, "cifar10_training_curves.png")


# ------------------------------------------------------- augmentation comparison
def augmentation_comparison() -> None | str:
    """Guarded on the final comparison CSV, not on the histories.

    The no-augmentation history is written after every epoch, so keying off its existence would
    happily draw a 3-epoch control against a 20-epoch run and call it a comparison. The CSV is
    written once, at the end of the control.
    """
    csv = ART / "cifar10_augmentation_comparison_20ep.csv"
    if not csv.exists():
        print("skip cifar10_augmentation_comparison.png (control not finished)")
        return None

    paths = {"Augmented": ART / "cifar10_training_history.csv",
             "No augmentation": ART / "cifar10_training_history_noaug.csv"}
    if not all(p.exists() for p in paths.values()):
        print("skip cifar10_augmentation_comparison.png (missing history)")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.3))
    for label, path in paths.items():
        h = pd.read_csv(path)
        colour = ACCENT if label == "Augmented" else WARM
        axes[0].plot(h["epoch"], h["val_accuracy"], color=colour, lw=1.9, label=label)
        axes[1].plot(h["epoch"], h["loss"], color=colour, lw=1.9, label=label)

    c = pd.read_csv(csv)
    a, p = c[c["augment"]].iloc[0], c[~c["augment"]].iloc[0]
    d = 100 * (a["test_acc"] - p["test_acc"])
    axes[0].set_title(f"Best validation\naugmented {100*a['best_val_acc']:.2f}%  vs  "
                      f"plain {100*p['best_val_acc']:.2f}%")
    axes[1].set_title(f"Final test accuracy\naugmented {100*a['test_acc']:.2f}%  vs  "
                      f"plain {100*p['test_acc']:.2f}%   ({d:+.2f} pts)")

    axes[0].set_ylabel("validation accuracy")
    axes[1].set_ylabel("training loss")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(True, **GRID)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Does augmentation earn its cost? 20 epochs, everything else identical",
                 y=1.05, fontsize=10.5, fontweight="bold")
    return save(fig, "cifar10_augmentation_comparison.png")


# -------------------------------------------------------------- confusion matrix
def confusion_matrix(tag: str = "", title: str = "CIFAR-10 confusion matrix (row-normalised %)") -> None:
    cm = load(f"cifar10_confusion_matrix{tag}.csv", index_col=0).loc[CLASSES, CLASSES]
    v = cm.values.astype(float)
    pct_mat = v / v.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(pct_mat, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(10), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(10), CLASSES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, pad=12)

    for i in range(10):
        for j in range(10):
            shade = "white" if pct_mat[i, j] > 55 else ("#1a1a1a" if pct_mat[i, j] > 22
                                                        else "#8894a0")
            ax.text(j, i, f"{pct_mat[i, j]:.0f}%", ha="center", va="center",
                    fontsize=7, color=shade)
            if v[i, j] and i != j and v[i, j] >= 45:
                ax.text(j, i + 0.34, f"n={int(v[i, j])}", ha="center", va="center",
                        fontsize=5.6, color=shade)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("% of true class", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    fig.text(0.5, -0.045,
             "Off-diagonal cells at n>=45 are annotated. In the augmented arm frog absorbs the "
             "animals it cannot identify; with augmentation off that collapses and the residual "
             "error is cat<->dog.",
             ha="center", fontsize=6.8, color="#555555")
    save(fig, f"cifar10_confusion_matrix{tag}.png")


# ------------------------------------------------------------- per-class accuracy
def per_class(tag: str = "", title: str = "Per-class recall and precision, sorted by recall",
              note: str = "frog has the best recall and a poor precision: it absorbs the animals "
                          "it cannot identify. Ranking by recall alone would rank the model's "
                          "worst over-predicter first.") -> None:
    cm = load(f"cifar10_confusion_matrix{tag}.csv", index_col=0).loc[CLASSES, CLASSES]
    v = cm.values.astype(float)
    support, predicted, correct = v.sum(axis=1), v.sum(axis=0), np.diag(v)
    recall = correct / support * 100
    precision = correct / predicted * 100

    order = np.argsort(recall)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    y = np.arange(10)
    h = 0.38
    ax.barh(y + h / 2, recall[order], height=h, color=ACCENT, label="recall")
    ax.barh(y - h / 2, precision[order], height=h, color=WARM, label="precision")
    ax.set_yticks(y, [CLASSES[i] for i in order])
    ax.axvline(100 * (correct.sum() / support.sum()), color="#333333", ls="--", lw=1,
               label=f"overall {100 * correct.sum() / support.sum():.2f}%")
    for i, idx in enumerate(order):
        ax.text(recall[idx] + 1.2, i + h / 2, f"{recall[idx]:.1f}", va="center",
                fontsize=6.8, color=ACCENT)
        ax.text(precision[idx] + 1.2, i - h / 2, f"{precision[idx]:.1f}", va="center",
                fontsize=6.8, color=WARM)
    ax.set_xlim(0, 108)
    ax.set_xlabel("percent")
    ax.set_title(title, pad=10)
    ax.grid(True, axis="x", **GRID)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.text(0.5, -0.03, note, ha="center", fontsize=6.8, color="#555555")
    save(fig, f"cifar10_per_class_accuracy{tag}.png")


# ------------------------------------------- inference-based figures (model needed)
def predictions() -> None:
    import os
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import keras

    from train_cifar10 import load_cifar10

    _, _, x_test, y_test = load_cifar10()
    model = keras.models.load_model(ART / "cifar10_model.keras", compile=False)
    probs = model.predict(x_test, batch_size=256, verbose=0)
    pred = probs.argmax(axis=1)
    true = np.asarray(y_test).reshape(-1)
    acc = (pred == true).mean()
    print(f"reloaded-model test accuracy {acc:.4f}")

    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(2, 6, figsize=(11.5, 4.4))
    shown = 0
    for i in range(200):
        if shown == 12:
            break
        if pred[i] == true[i]:
            continue
        ax = axes[shown // 6, shown % 6]
        ax.imshow(x_test[i] / 255.0)
        ax.set_title(f"true {CLASSES[true[i]]}\npred {CLASSES[pred[i]]}"
                     f"  {100 * probs[i, pred[i]]:.0f}%", fontsize=6.6, color=WARM)
        ax.axis("off")
        shown += 1
    fig.suptitle("Misclassified test images (model shown, label = truth / prediction / confidence)",
                 y=1.03, fontsize=10, fontweight="bold")
    save(fig, "cifar10_misclassified.png")

    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(4, 6, figsize=(11.5, 7.2))
    for slot, ax in enumerate(axes.ravel()):
        cls = slot % 10
        idx = np.where(true == cls)[0]
        i = int(rng.choice(idx))
        ax.imshow(x_test[i] / 255.0)
        ok = pred[i] == true[i]
        ax.set_title(f"{CLASSES[cls]}\n{'ok' if ok else 'WRONG: ' + CLASSES[pred[i]]}",
                     fontsize=6.8, color="#333333" if ok else WARM)
        ax.axis("off")
    fig.suptitle("One random test image per class (augmented arm, the model reloaded from disk)",
                 y=1.005, fontsize=10, fontweight="bold")
    save(fig, "cifar10_sample_predictions.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", action="store_true",
                    help="also render the two figures that need a forward pass")
    args = ap.parse_args()

    training_curves()
    augmentation_comparison()
    confusion_matrix()
    per_class()
    if (ART / "cifar10_confusion_matrix_noaug.csv").exists():
        confusion_matrix(
            "_noaug", "FINAL model (no augmentation), row-normalised %")
        per_class(
            "_noaug", "FINAL model: recall and precision, sorted by recall",
            "frog is still the most over-predicted class (1375 predictions for 1000 frogs) but "
            "at 66.7% precision rather than 35.8%. The weakest classes are now dog, cat and "
            "horse, confused with each other.")
    if args.predictions:
        predictions()