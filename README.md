# Week 3 — TensorFlow / Keras: Framework Port & CIFAR-10 Project

Complete solutions for the Week-3 assignments of the **Skill Set Go EduTech — AI/ML** roadmap,
organized **task-wise** under `week3/`.

```
week3/
├── README.md
├── task3_1_tensorflow_basics/     → 3.1 TensorFlow Basics
├── task3_2_cifar10_project/       → 3.2 CIFAR-10 Classification Project
└── task3_3_technical_report/      → 3.3 Deep Learning Technical Report
```

| Task | Topic | Level | Location | Status |
|------|-------|-------|----------|--------|
| 3.1 | TensorFlow Basics | Intermediate | `task3_1_tensorflow_basics/tensorflow_fashion_mnist.ipynb` | Done — 91.40% test acc |
| 3.2 | CIFAR-10 Classification Project | Intermediate | `task3_2_cifar10_project/cifar10_classifier.ipynb` | Done — 65.91% test acc |
| 3.3 | Deep Learning Technical Report | Intermediate | `task3_3_technical_report/technical_report_week3.pdf` | Done — 2 pages |

**Datasets.** 3.1 uses Fashion-MNIST, deliberately the *same* data and the *same* 54,000/6,000
per-class split as Week 2, because the entire point is a like-for-like comparison against the
PyTorch model already built there. 3.2 moves to CIFAR-10 (50,000 train / 10,000 test, 32×32×3),
split 45,000 / 5,000 stratified. In both tasks **the official test set is opened exactly once**,
after every architecture and hyperparameter decision has already been made against validation.

## 3.1 TensorFlow Basics — `task3_1_tensorflow_basics/`

The Week 2 CNN rebuilt in TensorFlow/Keras: `2 × (Conv3×3 + BatchNorm + ReLU + MaxPool2) →
Dense128 → Dense10`, **421,738 parameters**, Adam 1e-3, batch 64, 8 epochs. Final test accuracy
**0.9140**, against **0.9067** for the PyTorch original.

- **The parameter match is the deliverable, not the accuracy.** `421,738` in both frameworks, and
  getting there required noticing that `BatchNormalization` makes the convolution's own bias
  redundant (64 spare parameters). An unmatched count would have made the whole comparison
  meaningless while still looking reasonable.
- **A 192-parameter discrepancy that had a real explanation.** The first version reported 421,930
  Keras parameters against PyTorch's 421,738. `model.count_params()` sums `layer.weights`, and
  `BatchNormalization` carries four tensors per channel — learnable `gamma` and `beta` plus
  non-learnable `moving_mean` and `moving_variance` — while `model.parameters()` yields trainable
  tensors only. The frameworks disagree by exactly `2 × C_bn = 2 × (32+64) = 192`. This is the
  same buffer-versus-parameter distinction that made the Week 2 CNN report undercount, so it is
  documented in `artifacts/tf_vs_pytorch_comparison.md` rather than patched quietly.
- **A same-recipe repeat run, to size the noise.** Rerunning the identical recipe with a fresh
  seed moved best-validation accuracy by **0.27 points**. The framework gap is +0.73 points on
  test — roughly 2.7× the noise, which is *not* enough to claim TensorFlow is better, and the
  write-up says so instead of implying a win. Two runs cannot separate "better framework" from
  "luckier seed".
- **API differences that actually bite** — `CrossEntropyLoss` takes raw logits while
  `SparseCategoricalCrossentropy` defaults to `from_logits=False`; `Conv2D` defaults to having a
  bias where `nn.Conv2d` does not; `fit` keeps the **final** weights, so the best checkpoint has to
  be reloaded before evaluating or the quoted number describes a model that no longer exists.

## 3.2 CIFAR-10 Classification Project — `task3_2_cifar10_project/`

Four convolutions with BatchNorm and dropout, global average pooling, **120,810 parameters**,
SGD + Nesterov momentum on a cosine schedule, label smoothing 0.1, 20 epochs. Test accuracy
**0.6591** (best validation 0.6773 at epoch 18).

- **Training is a script, not a notebook cell.** Native-Windows TensorFlow has had no GPU path
  since 2.11, so an epoch costs minutes. `train_cifar10.py` checkpoints after *every* epoch and
  resumes with the optimizer's momentum intact; the notebook loads the artifacts and regenerates
  every figure in seconds. The learning rate is written as a function of the **absolute epoch
  index** specifically so a resumed run lands on the LR an uninterrupted one would have used.
- **Augmentation runs per batch.** Applying `RandomFlip` to the array once up front freezes one
  augmentation per image for all 20 epochs. It lives in a `tf.data` pipeline instead. Two API
  traps are documented in the notebook: `RandomTranslation` takes a *fraction* of the image, not
  pixels, and normalisation lives inside the model so the train and eval paths cannot drift apart.
- **Per-class accuracy alone lies.** Ranking classes by recall makes `frog` look outstanding at
  0.961 — but its precision is 0.358, so it is absorbing every animal it cannot identify. `cat` is
  the opposite failure at 0.305 recall. The notebook reports both and says what the confusion
  actually is: `cat`/`dog`/`deer`/`frog` are confused with *each other*, which at 32×32 depends
  on pose rather than texture.
- **The augmentation ablation is reported as the inconclusive negative it is.** Three epochs gave
  plain 0.5750 vs augmented 0.4971 — augmentation *lost* by 7.79 points, which is the expected
  early-training cost and says nothing about the benefit, which only appears late. A full 20-epoch
  no-augmentation control was run to settle it properly; see
  `artifacts/cifar10_augmentation_comparison_20ep.csv`.
- **0.6591 is well short of a competent baseline** and the notebook says so: a ResNet-18 (~11M
  parameters) commonly reports around 0.93 here. The model was sized to finish on CPU, so the
  architecture is the binding constraint, not the training recipe.

## 3.3 Deep Learning Technical Report — `task3_3_technical_report/`

`technical_report_week3.pdf` — two pages, A4. The port comparison, the CIFAR-10 project, the
augmentation control, and the limits of both.

`build_report.py` generates it and reads **every number from the artifacts** of tasks 3.1 and 3.2
(`task3_1_results.json`, `cifar10_training_history.csv`, `task3_2_results.json`,
`cifar10_augmentation_comparison_20ep.csv`, the per-class and confusion-matrix CSVs), so the prose
cannot drift from the runs it describes. Regenerate with:

```bash
cd task3_3_technical_report
python build_report.py
```

## Reproducing

```bash
cd week3
pip install numpy pandas matplotlib seaborn scikit-learn reportlab jupyter tensorflow

# 3.1 — training first (~28 min CPU), then the notebook (seconds)
cd task3_1_tensorflow_basics
python train_fmnist.py && jupyter lab tensorflow_fashion_mnist.ipynb

# 3.2 — timing probe, then the main run, the augmentation control, the notebook
cd ../task3_2_cifar10_project
python train_cifar10.py --mode smoke        # seconds per step, projects the full run
python train_cifar10.py --mode main         # ~3 h wall clock on a contended CPU
python train_cifar10.py --mode fullplain    # 20-epoch no-augmentation control
jupyter lab cifar10_classifier.ipynb

# 3.3 — needs 3.1 and 3.2 artifacts present
cd ../task3_3_technical_report && python build_report.py
```

## Environment

- Python 3.11.9, TensorFlow 2.21.0, Keras 3.15.1, NumPy 2.4.6, pandas 3.0.6,
  matplotlib 3.11.2, scikit-learn 1.9.1, reportlab 5.0.1
- **No GPU.** TensorFlow on native Windows has had no CUDA path since 2.11
  (`WARNING:tensorflow:TensorFlow GPU support is not available on native Windows for
  TensorFlow >= 2.11`), so every epoch here is CPU-bound. This is what dictated the model size
  in 3.2.
- All notebooks are **executed with outputs embedded**; re-run them to confirm from scratch. Both
  training scripts are resumable and their CSVs are checked in, so the notebooks re-render
  without repeating hours of training.
- `keras.layers.Normalization` is parameterised by **variance**, and Keras 3.15 **removed
  `label_smoothing` from every loss constructor** — both differences from the TF 2.x idioms in
  most tutorials, and both handled explicitly in `train_cifar10.py`.