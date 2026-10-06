# Task 3.1 — TensorFlow/Keras vs PyTorch on Fashion-MNIST

Same architecture, same nominal recipe, same dataset split, implemented twice. This note records
what actually differed between the two frameworks and what the numbers do and do not support.

## Setup

| | |
|---|---|
| Architecture | Conv(32,3) → Conv(64,3) → MaxPool → Dropout(0.25) → Dense(128) → Dense(10) |
| Normalisation | BatchNorm after each conv, ReLU after each BN |
| Optimiser | Adam, lr 1e-3 |
| Batch size | 64 (844 steps/epoch over the 54,000-image train split) |
| Epochs | 8 |
| Framework versions | TensorFlow 2.21.0 / Keras 3.15.1 (CPU), PyTorch (Week 2) |
| Trainable parameters | **421,738 in both** |

The parameter match is the load-bearing part. It was only reached after disabling the Keras
`Conv2D` biases: `BatchNormalization` already supplies a learnable shift, so keeping the
convolution's own bias adds redundant parameters (64 of them, here) and breaks parity.

## The parameter-counting trap

My first version of the comparison cell reported **421,930** Keras parameters against PyTorch's
**421,738** and I could not account for the 192-parameter gap on what is provably the same
architecture.

`model.count_params()` sums the length of `layer.weights`, and `BatchNormalization` carries four
tensors per channel: the learnable `gamma` and `beta`, plus the non-learnable `moving_mean` and
`moving_variance`. PyTorch's `model.parameters()` yields trainable tensors only. The frameworks
therefore disagree by exactly `2 x C` summed over BatchNorm layers — `2 x (32 + 64) = 192` here.

This is the same buffer-versus-parameter distinction that made the Week 2 CNN report undercount
by 192 parameters, so it is worth stating explicitly rather than patching quietly: compare
`model.trainable_weights` against `model.parameters()`, and be suspicious whenever two frameworks
disagree by a round number.

## Results

| Run | Final val acc | Best val acc | **Test acc** | Wall clock |
|---|---|---|---|---|
| Week 2, PyTorch | 0.9043 | 0.9043 (epoch 8) | **0.9067** | ~13.5 min |
| Task 3.1, TensorFlow | 0.9213 | 0.9213 (epoch 8) | **0.9140** | ~15.0 min |
| Task 3.1, TensorFlow repeat (same recipe, new seed) | 0.9135 | 0.9187 (epoch 5) | not evaluated | ~13.4 min |

On the held-out test set TensorFlow ends **+0.73 points** above PyTorch (0.9140 vs 0.9067). On
validation the final-epoch gap is larger, **+1.70 points** (0.9213 vs 0.9043).

## What that gap does and does not show

It does not show that TensorFlow is the better framework. The repeat run exists precisely to
measure how much of this is noise, and it moves the best-validation number by **0.27 points**
with the framework held completely fixed.

So the honest reading is:

- The **test-set** gap (+0.73) is only about 2.7x the same-recipe spread (+0.27). That is weak
  evidence: a couple more seeds would plausibly erase it. Any claim that the frameworks differ
  in quality is not supported by this data.
- The **validation** gap (+1.70) is larger, but validation accuracy at a single final epoch is
  the noisiest number in the table — it depends on exactly where training happened to stop.
- What the repeat *does* establish is that differences below roughly 1 pp on this setup are not
  interpretable, which means the per-class accuracies in either report should be read as
  approximate rather than exact.
- To actually settle it you would need several seeds per framework and a distribution of
  outcomes. Two runs per framework cannot.

Unseeded sources of variation between runs: weight initialisation, dropout masks, and shuffle
order. The main runs were seeded (`seed = 42`) but the repeat deliberately was not, which is what
makes it a noise probe.

## Where the errors actually are

Per-class test accuracy (`artifacts/tf_per_class_accuracy.csv`) reproduces the well-known
Fashion-MNIST structure rather than anything framework-specific:

| Class | Accuracy | Most confused with |
|---|---|---|
| Shirt | 0.727 | T-shirt/top |
| Pullover | 0.850 | Shirt |
| T-shirt/top | 0.851 | Shirt |
| Coat | 0.872 | Shirt |
| Bag | 0.981 | Dress |

Four of the five hardest classes are upper garments that differ mainly in sleeve length, and
they collapse into one another. Trouser, Sandal, Sneaker, Bag and Ankle boot — classes separable
by silhouette alone — all exceed 0.96. Both frameworks produce this same shape, which is the
clearest evidence that the residual error is a property of the data, not of the library.

## API differences that actually bite

| Concern | PyTorch | Keras |
|---|---|---|
| Loss convention | `nn.CrossEntropyLoss()` takes **raw logits** | `SparseCategoricalCrossentropy()` defaults to `from_logits=False`, so it takes **probabilities** |
| Output activation | none — log-softmax happens inside the loss | explicit `softmax` on the last layer |
| Optimiser step | `optimizer.step()` after `backward()` | handled inside `train_on_batch` / `fit` |
| Conv biases | `nn.Conv2d(..., bias=False)` explicit | `use_bias=False` — **not** the default |
| Parameter count | `model.parameters()` — trainable only | `count_params()` **also counts BN running buffers** |
| Best weights | `best_state = copy.deepcopy(model.state_dict())` | `ModelCheckpoint(save_best_only=True)`; `fit` otherwise keeps the **final** weights |

The last row is the one most likely to produce a quietly wrong report. `fit` does not restore the
best epoch's weights, so any "best epoch" number quoted without reloading the checkpoint describes
a model that no longer exists. Both reports here reload the saved checkpoint before evaluating.

## CPU note

TensorFlow on native Windows has no GPU path from 2.11 onward (`WARNING:tensorflow:TensorFlow GPU
support is not available on native Windows...`), so both frameworks ran CPU-only here. Epoch times
are therefore comparable; the ~10% difference in wall clock between the PyTorch and TensorFlow
runs is within run-to-run variation on a shared machine and should not be read as a framework
speed benchmark.