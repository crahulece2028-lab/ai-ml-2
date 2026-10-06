# Task 3.2 — CIFAR-10 Classification Project: Summary

**Final model: the no-augmentation variant, 0.7684 test accuracy.** The augmentation pipeline that
was originally part of the recipe was measured and **rejected** — it cost 10.93 points.

## Setup

| | |
|---|---|
| Dataset | CIFAR-10, 50,000 train / 10,000 test, 32×32×3 |
| Split | 45,000 / 5,000 stratified (`random_state=42`), test set sealed until final evaluation |
| Architecture | 4 convolutions (32→32→64→128) + BatchNorm + ReLU, max-pool + dropout per block, global average pooling, Dense 128 |
| Trainable parameters | **120,810** (121,578 counting BatchNorm's 768 non-trainable running buffers) |
| Optimizer | SGD 0.05, Nesterov momentum 0.9, cosine decay over absolute epoch index |
| Regularisation | label smoothing 0.1, dropout 0.25 (scaled per block) |
| Batch / epochs / seed | 128 / 20 / 42 |
| Hardware | CPU only — TensorFlow has no native-Windows GPU path since 2.11 |
| Cost | ~210–220 s/epoch median; ~1.2 h per 20-epoch arm |

## Results

Two arms, identical in every respect except the augmentation flag:

| Arm | Best val | Best epoch | **Test** | Median s/epoch |
|---|---|---|---|---|
| With augmentation | 0.6773 | 18 | 0.6591 | 210 |
| **Without augmentation** | **0.7889** | 17 | **0.7684** | 220 |
| Effect of augmentation | −11.16 pts | | **−10.93 pts** | |

The plain variant is ahead at **every epoch** and the gap widens with training. This is not the
small early-training deficit that later reverses — the augmented run never catches up.

### Per-class behaviour

The two arms fail differently, which is more interesting than either accuracy alone:

| | augmented arm | final (plain) arm |
|---|---|---|
| Best recall | frog 96.1% | frog 91.7% |
| …its precision | **35.8%** | 66.7% |
| Times frog is predicted | 2,684 (for 1,000 actual) | 1,375 |
| Worst recall | cat 30.5% | dog 60.6% |
| Worst precision | frog 35.8% | cat 59.2% |

In the augmented model `frog` is an over-prediction trap: top recall, worst precision, absorbing
every animal it cannot identify. Removing augmentation shrinks that pathology substantially
(35.8% → 66.7% precision) and moves the residual error onto the classic CIFAR-10 confusion of
`dog`/`cat`/`horse` with each other.

### The 3-epoch ablation pointed the right way for the wrong reason

| Epochs | Plain | Augmented | Delta |
|---|---|---|---|
| 3 | 0.5750 | 0.4971 | −7.79 pts |
| 20 | 0.7889* | 0.6773* | −11.16 pts |

(*best validation.) I recorded the 3-epoch result as inconclusive on the grounds that a short
ablation cannot see where a regulariser pays off. That caution was right, but the prediction
built on top of it — that the augmented run would overtake later — was simply **wrong**. The
correct reading of a 7.8-point deficit at epoch 3 had to include "augmentation is actively harmful
here", and running the full control was what established it.

## Why augmentation loses

Most likely capacity, not a defect in the pipeline. The ops are standard and were verified as
rendered: horizontal flip, translation ±4/32 = 0.125 of the image (a *fraction*, not pixels),
zoom ±10%, brightness and contrast ±15%, applied per batch inside `tf.data` to the training split
only, on raw 0–255 images before the model's own `Rescaling`/`Normalization`. The two arms share
every one of those code paths; only the flag differs.

At 120,810 parameters the network is already **underfitting** the unaugmented task. Augmentation
is a regulariser, and regularising an under-parameterised model pushes it further from a fit it
was struggling to reach anyway. The pipeline is tuned for a larger network than the CPU budget
allowed — the fix is depth or width, not more augmentation.

Confidence: the **sign** is solid (11 points against a measured 0.27-point run-to-run noise floor
from task 3.1). The **magnitude** is a single seed per arm.

## Honest limitations

- **Well short of a competent baseline.** 0.7684 against the ~0.93 a ResNet-18 (~11M parameters)
  typically reports. The architecture, not the recipe, is the binding constraint.
- **20 epochs is short** and validation was still moving at epoch 17–18 of 20.
- **One seed per arm**, so differences of a point or less here are not interpretable.
- **Timing is contaminated.** A few epochs lost most of their CPU to other work on this shared
  machine; the mean epoch time in the results JSON is inflated, the median is the honest figure.

## Reproducing

```bash
python train_cifar10.py --mode smoke        # seconds-per-step probe
python train_cifar10.py --mode main         # augmented arm, 20 epochs
python train_cifar10.py --mode fullplain    # no-augmentation control, 20 epochs
python train_cifar10.py --mode ablation     # 3-epoch on/off comparison
python make_figures.py --predictions        # all figures from stored artifacts
jupyter lab cifar10_classifier.ipynb
```

Both arms checkpoint every epoch and resume with optimizer momentum intact, so an interrupted run
continues on the LR schedule an uninterrupted one would have used.

## Files

| File | Contents |
|---|---|
| `cifar10_model_noaug.keras` | **final model** (554.8 KB) |
| `cifar10_model.keras` | augmented arm, for comparison |
| `cifar10_best.keras` / `cifar10_best_noaug.keras` | best-validation weights per arm |
| `task3_2_results.json` / `task3_2_results_noaug.json` | metrics per arm |
| `cifar10_augmentation_comparison_20ep.csv` | the control, side by side |
| `cifar10_training_history.csv` / `_noaug.csv` | per-epoch metrics per arm |
| `cifar10_confusion_matrix.csv` / `_noaug.csv` | 10×10 counts per arm |
| `cifar10_per_class_accuracy.csv` / `_noaug.csv` | per-class metrics per arm |
| `cifar10_augmentation_ablation.csv` | the 3-epoch ablation |
| `cifar10_training_curves.png` | augmented arm history + epoch timing |
| `cifar10_augmentation_comparison.png` | the 20-epoch control, both curves |
| `cifar10_confusion_matrix.png` / `_noaug.png` | confusion matrices per arm |
| `cifar10_per_class_accuracy.png` / `_noaug.png` | recall vs precision per arm |
| `cifar10_misclassified.png`, `cifar10_sample_predictions.png` | qualitative, augmented arm |