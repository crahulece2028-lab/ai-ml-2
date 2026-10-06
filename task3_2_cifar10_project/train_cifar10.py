"""Task 3.2 — CIFAR-10 image classifier in TensorFlow/Keras (CPU).

The flagged "project" task. Training lives in a resumable script rather than inline in a
notebook because CPU-only CIFAR-10 takes hours and an interrupted run must not restart from
zero. The notebook `cifar10_classifier.ipynb` carries the same model code and regenerates every
figure and table from the artifacts written here.

Design notes that matter:

* **The test set is never used for validation.** The 50,000 training images are split
  45,000 / 5,000 stratified; the official 10,000 test images are touched once, at the end.
* **Augmentation runs per batch, inside a `tf.data` pipeline**, not over the array once.
  Applying `RandomFlip` to the full training array up front would freeze one fixed
  augmentation per image and quietly remove most of the benefit.
* **Normalisation lives in the model**, so the train and evaluation paths share it by
  construction and cannot drift apart.
* **The learning rate is a function of the absolute epoch index**, so resuming at epoch 14
  lands on exactly the LR an uninterrupted run would have used.

Modes
-----
    python train_cifar10.py --mode smoke      # timing probe, prints projections
    python train_cifar10.py --mode main       # full run + test evaluation
    python train_cifar10.py --mode ablation   # augmentation on/off, 3 epochs each
    python train_cifar10.py --force           # discard checkpoints and start over
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import keras
import tensorflow as tf

SEED = 42
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

# Channel statistics for CIFAR-10.
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)
# Keras 3's Normalization layer parameterises scale by *variance*, not std.
VARIANCE = tuple(s * s for s in STD)

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
CKPT = ART / "cifar10_checkpoint.keras"
BEST = ART / "cifar10_best.keras"
FINAL = ART / "cifar10_model.keras"
HISTORY = ART / "cifar10_training_history.csv"
RESULTS = ART / "task3_2_results.json"
ABLATION = ART / "cifar10_augmentation_ablation.csv"

# A full-length no-augmentation control run. The 3-epoch ablation cannot answer whether
# augmentation pays off, because augmentation's cost shows up early and its benefit shows up
# late. This arm is the same 20-epoch schedule as the main run with the pipeline switched off,
# which is the only comparison that settles the question. Separate paths throughout so it
# cannot clobber the augmented run's artifacts.
NOAUG_CKPT = ART / "cifar10_checkpoint_noaug.keras"
NOAUG_BEST = ART / "cifar10_best_noaug.keras"
NOAUG_FINAL = ART / "cifar10_model_noaug.keras"
NOAUG_HISTORY = ART / "cifar10_training_history_noaug.csv"
NOAUG_RESULTS = ART / "task3_2_results_noaug.json"

MAIN_EPOCHS = 20
BATCH = 128
BASE_LR = 0.05
MOMENTUM = 0.9
LABEL_SMOOTHING = 0.1
SHIFT_PX = 4


# ---------------------------------------------------------------------------- model
def build_model(width: int = 32, dropout: float = 0.25) -> keras.Sequential:
    """A deliberately small VGG-style CNN, sized for CPU.

    The first block runs at 32x32x3, where every extra channel costs the most, so width grows
    slowly (32 -> 64 -> 128) while spatial resolution halves each block. Four convolutions
    plus a global average pool is about the largest thing that trains in sane wall clock here.

    The docstring used to claim that "on CIFAR-10 augmentation is worth more than depth anyway".
    The 20-epoch control run (`--mode fullplain`) measured that assumption and refuted it: with
    this pipeline switched off the same model reaches 0.7889 best validation / 0.7684 test,
    against 0.6773 / 0.6591 with augmentation. At this capacity the augmentation costs about 11
    points rather than buying any, so the plain variant is the model worth keeping.
    """
    return keras.Sequential([
        keras.layers.Input(shape=(32, 32, 3)),
        keras.layers.Rescaling(1.0 / 255.0),
        keras.layers.Normalization(mean=MEAN, variance=VARIANCE),

        keras.layers.Conv2D(width, 3, padding="same", use_bias=False),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.Conv2D(width, 3, padding="same", use_bias=False),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.MaxPooling2D(2),                    # 16x16
        keras.layers.Dropout(dropout * 0.6),

        keras.layers.Conv2D(width * 2, 3, padding="same", use_bias=False),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.MaxPooling2D(2),                    # 8x8
        keras.layers.Dropout(dropout),

        keras.layers.Conv2D(width * 4, 3, padding="same", use_bias=False),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.MaxPooling2D(2),                    # 4x4
        keras.layers.Dropout(dropout * 1.4),

        keras.layers.GlobalAveragePooling2D(),           # 4x4x128 -> 128
        keras.layers.Dense(width * 4, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(dropout * 1.6),
        keras.layers.Dense(10, activation="softmax"),
    ])


@keras.saving.register_keras_serializable(package="cifar10")
def smoothed_sparse_categorical_crossentropy(y_true, y_pred):
    """Sparse categorical cross-entropy with label smoothing.

    Keras 3.15 removed the `label_smoothing` argument from every loss constructor, so this is
    spelled out: one-hot the integer label, blend toward the uniform distribution, then take
    categorical cross-entropy against that softened target. Registered as a serializable so the
    checkpoint written by `ModelCheckpoint` still reloads with its loss attached.
    """
    n_classes = keras.ops.shape(y_pred)[-1]
    y_true = keras.ops.cast(y_true, y_pred.dtype)
    y_true = keras.ops.one_hot(y_true, n_classes)
    y_true = y_true * (1.0 - LABEL_SMOOTHING) + LABEL_SMOOTHING / n_classes
    per_sample = keras.losses.categorical_crossentropy(y_true, y_pred)
    return keras.ops.mean(per_sample)


def compile_model(model: keras.Model) -> keras.Model:
    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=BASE_LR, momentum=MOMENTUM, nesterov=True),
        loss=smoothed_sparse_categorical_crossentropy,
        metrics=["accuracy"],
    )
    return model


def lr_at(epoch: int, total_epochs: int) -> float:
    """Cosine decay BASE_LR -> ~0 as a function of the absolute epoch index."""
    return BASE_LR * 0.5 * (1.0 + math.cos(math.pi * epoch / total_epochs))


# --------------------------------------------------------------------------- data
def augmentation_ops(strength: float = 1.0):
    """Per-batch augmentation ops. Built once, applied inside `Dataset.map`.

    `RandomTranslation` takes its factor as a fraction of the image dimension, not pixels, so
    the 4-pixel shift is expressed as 4/32 = 0.125. Passing 4.0 raises, and the "obvious"
    pixel value is wrong by a factor of 32.
    """
    shift = SHIFT_PX * strength / 32.0
    layers = [
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomTranslation(shift, shift),
        keras.layers.RandomZoom(-0.1 * strength, 0.1 * strength),
        keras.layers.RandomBrightness(0.15 * strength),
        keras.layers.RandomContrast(0.15 * strength),
    ]

    def augment(img):
        for layer in layers:
            img = layer(img, training=True)
        return img

    return augment


def make_dataset(x, y, batch: int, train: bool, augment: bool):
    """Build a tf.data pipeline. Augmentation applies to the training split only.

    No `.cache()`: `from_tensor_slices` on a numpy array already references that array's
    buffer, so caching would only duplicate ~550 MB of CIFAR-10 into a second copy.
    """
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if train:
        ds = ds.shuffle(len(x), seed=SEED, reshuffle_each_iteration=True)
        if augment:
            aug = augmentation_ops()
            ds = ds.map(lambda img, lab: (tf.cast(aug(img), tf.float32), lab),
                        num_parallel_calls=tf.data.AUTOTUNE)
    # repeat() keeps every epoch full: shuffle() without a buffer size discards the tail.
    ds = ds.repeat()
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


def load_cifar10():
    """Return raw 0-255 float32 images and rank-1 int labels.

    Scaling is left to `build_model`'s `Rescaling` layer so the training and evaluation paths
    cannot drift apart. Keras hands back labels as (n, 1); they are flattened here because
    `SparseCategoricalCrossentropy` wants rank-1 targets and sklearn's stratified split is
    happier with them too.
    """
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    return (x_train.astype("float32"), y_train.astype("int64").ravel(),
            x_test.astype("float32"), y_test.astype("int64").ravel())


# -------------------------------------------------------------------- callbacks
class EpochTimer(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.seconds = []
        self._t0 = None

    def on_epoch_begin(self, epoch, logs=None):
        self._t0 = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        self.seconds.append(time.perf_counter() - self._t0)


class SetLR(keras.callbacks.Callback):
    """Assign the cosine-schedule LR at the start of every epoch."""

    def __init__(self, total_epochs):
        super().__init__()
        self.total = total_epochs

    def on_epoch_begin(self, epoch, logs=None):
        lr = lr_at(epoch, self.total)
        self.model.optimizer.learning_rate.assign(lr)


class HistoryWriter(keras.callbacks.Callback):
    """Persist one row of history per epoch, and checkpoint every epoch.

    The checkpoint has to land in `on_epoch_end`, not `on_train_end`. The whole point of this
    script is surviving an interruption hours into a run, and a checkpoint written only at the
    end is worth nothing to a process killed at epoch 14.
    """

    def __init__(self, path: Path, total_epochs: int, timer: EpochTimer,
                 ckpt_path: Path = CKPT):
        super().__init__()
        self.path = path
        self.total = total_epochs
        self.timer = timer
        self.ckpt_path = ckpt_path
        self.rows = []

    def on_epoch_end(self, epoch, logs=None):
        self.rows.append({
            "epoch": epoch + 1,
            "lr": round(lr_at(epoch, self.total), 6),
            "loss": round(float(logs["loss"]), 6),
            "accuracy": round(float(logs["accuracy"]), 6),
            "val_loss": round(float(logs["val_loss"]), 6),
            "val_accuracy": round(float(logs["val_accuracy"]), 6),
            "epoch_seconds": round(self.timer.seconds[-1], 1),
        })
        pd.DataFrame(self.rows).to_csv(self.path, index=False)
        self.model.save(self.ckpt_path)


# ----------------------------------------------------------------------- training
def run_training(x_tr, y_tr, x_va, y_va, epochs: int, tag: str, augment: bool,
                 history_path: Path | None = None, track_best_to: Path | None = None,
                 ckpt_path: Path = CKPT):
    keras.utils.set_random_seed(SEED)

    train_ds = make_dataset(x_tr, y_tr, BATCH, train=True, augment=augment)
    val_ds = make_dataset(x_va, y_va, BATCH * 2, train=False, augment=False)

    model = compile_model(build_model())
    timer = EpochTimer()
    cbs = [SetLR(epochs), timer]

    best_path = track_best_to or (ART / "cifar10_best.keras")
    cbs.append(keras.callbacks.ModelCheckpoint(
        filepath=str(best_path), monitor="val_accuracy", mode="max",
        save_best_only=True, verbose=0))
    if history_path is not None:
        cbs.append(HistoryWriter(history_path, epochs, timer, ckpt_path))

    steps = int(np.ceil(len(x_tr) / BATCH))
    val_steps = int(np.ceil(len(x_va) / (BATCH * 2)))

    print("\n=== %s === %d epochs, %d steps/epoch, augment=%s"
          % (tag, epochs, steps, augment), flush=True)
    t0 = time.perf_counter()
    hist = model.fit(train_ds, validation_data=val_ds,
                     epochs=epochs, steps_per_epoch=steps,
                     validation_steps=val_steps, verbose=2, callbacks=cbs)
    seconds = time.perf_counter() - t0

    df = pd.DataFrame(hist.history)
    df.index = range(1, len(df) + 1)
    df.index.name = "epoch"
    # Materialise `epoch` as a real column. HistoryWriter writes one, but this frame is what
    # main() reads for best_epoch, and an index is not a column -- hence the KeyError.
    df["epoch"] = df.index
    df["epoch_seconds"] = timer.seconds
    df["lr"] = [lr_at(i, epochs) for i in range(epochs)]

    print("%s: %.2f min | %.1f s/epoch | best val %.4f"
          % (tag, seconds / 60, float(np.mean(timer.seconds)), df["val_accuracy"].max()),
          flush=True)
    return model, df, seconds


# --------------------------------------------------------------------------- main
def evaluate(model: keras.Model, x_test, y_test, prefix: str = "") -> dict:
    probs = model.predict(x_test, batch_size=1000, verbose=0)
    y_pred = probs.argmax(axis=1)
    test_acc = float((y_pred == y_test).mean())
    cm = confusion_matrix(y_test, y_pred)

    def top_confusion(i: int) -> str:
        """The class this one is most often mistaken for.

        The diagonal has to be masked out, otherwise argmax returns the class's own row
        maximum -- which is the diagonal -- and every class reports itself.
        """
        row = cm[i].astype(float).copy()
        row[i] = -1.0
        return CLASSES[int(row.argmax())]

    per_class = pd.DataFrame([
        {"class": c, "accuracy": float(cm[i, i] / cm[i].sum()), "support": int(cm[i].sum()),
         "most_confused_with": top_confusion(i)}
        for i, c in enumerate(CLASSES)
    ]).sort_values("accuracy").reset_index(drop=True)
    per_class.to_csv(ART / f"cifar10_per_class_accuracy{prefix}.csv", index=False)
    pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(
        ART / f"cifar10_confusion_matrix{prefix}.csv")

    print("\n" + classification_report(y_test, y_pred, target_names=CLASSES,
                                        digits=4, zero_division=0))
    return {"test_accuracy": test_acc, "cm": cm, "per_class": per_class, "y_pred": y_pred}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "main", "fullplain", "ablation"], default="main")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    ART.mkdir(exist_ok=True)

    print("tensorflow %s | keras %s | seed %d" % (tf.__version__, keras.__version__, SEED))
    print("devices: %s" % [d.device_type for d in tf.config.list_physical_devices()])

    x_all, y_all, x_test, y_test = load_cifar10()
    # 45k / 5k stratified from the official training set. Test set stays sealed.
    x_tr, x_va, y_tr, y_va = train_test_split(
        x_all, y_all, test_size=5000, random_state=SEED, stratify=y_all)
    print("train %s | val %s | test %s (sealed until final evaluation)"
          % (x_tr.shape, x_va.shape, x_test.shape))

    if args.mode == "smoke":
        # Timing probe. Seconds-per-step is the number that decides whether a 20-epoch run
        # is viable at all, so measure it before committing hours to it.
        n = 5000
        ds = make_dataset(x_tr[:n], y_tr[:n], BATCH, train=True, augment=True)
        model = compile_model(build_model())
        steps = int(np.ceil(n / BATCH))
        # Warm up first. The opening steps pay graph-tracing and allocator costs, which would
        # otherwise inflate the projection and talk me out of a run that is actually fine.
        model.fit(ds, epochs=1, steps_per_epoch=3, verbose=0)
        t0 = time.perf_counter()
        model.fit(ds, epochs=1, steps_per_epoch=steps, verbose=2)
        elapsed = time.perf_counter() - t0
        per_step = elapsed / steps
        full_steps = int(np.ceil(len(x_tr) / BATCH))
        print("\n%d steps over %d images: %.1f s  -> %.3f s/step"
              % (steps, n, elapsed, per_step))
        print("projected epoch (45000 imgs) : %.1f min" % (per_step * full_steps / 60))
        print("projected %d epochs          : %.2f h" % (MAIN_EPOCHS, per_step * full_steps * MAIN_EPOCHS / 3600))
        return

    if args.mode == "ablation":
        rows = []
        for augment in (False, True):
            tag = "augmented" if augment else "plain"
            # Separate best-weight files, so a 3-epoch probe cannot overwrite the main
            # run's best checkpoint.
            probe_best = ART / f"ablation_{tag}_best.keras"
            _, df, secs = run_training(x_tr, y_tr, x_va, y_va, 3, tag, augment,
                                       track_best_to=probe_best)
            rows.append({
                "setting": tag, "augment": augment, "epochs": 3,
                "final_val_acc": round(float(df["val_accuracy"].iloc[-1]), 4),
                "best_val_acc": round(float(df["val_accuracy"].max()), 4),
                "minutes": round(secs / 60, 2),
            })
        out = pd.DataFrame(rows)
        out.to_csv(ABLATION, index=False)
        print("\n" + out.to_string(index=False))
        gain = rows[1]["final_val_acc"] - rows[0]["final_val_acc"]
        print("\naugmentation delta at 3 epochs: %+.4f" % gain)
        print("wrote %s" % ABLATION.name)
        return

    # ------------------------------------------------------ full-length no-aug control
    if args.mode == "fullplain":
        epochs = args.epochs or MAIN_EPOCHS
        if NOAUG_HISTORY.exists() and not args.force:
            print("\nplain control already trained -> loading %s" % NOAUG_BEST.name)
            df = pd.read_csv(NOAUG_HISTORY)
            model = keras.models.load_model(NOAUG_BEST, compile=False)
        else:
            model, df, _ = run_training(
                x_tr, y_tr, x_va, y_va, epochs, "plain control (no augmentation)",
                augment=False, history_path=NOAUG_HISTORY,
                track_best_to=NOAUG_BEST, ckpt_path=NOAUG_CKPT)
            model = keras.models.load_model(NOAUG_BEST, compile=False)

        ev = evaluate(model, x_test, y_test, prefix="_noaug")
        results = {
            "dataset": "CIFAR-10",
            "setting": "no augmentation (control)",
            "augment": False,
            "epochs": int(len(df)),
            "best_val_accuracy": round(float(df["val_accuracy"].max()), 4),
            "best_epoch": int(df.loc[df["val_accuracy"].idxmax(), "epoch"]),
            "final_val_accuracy": round(float(df["val_accuracy"].iloc[-1]), 4),
            "test_accuracy": round(ev["test_accuracy"], 4),
            "seconds_per_epoch_median": round(float(df["epoch_seconds"].median()), 1),
        }
        NOAUG_RESULTS.write_text(json.dumps(results, indent=2), encoding="utf-8")
        model.save(NOAUG_FINAL)
        NOAUG_CKPT.unlink(missing_ok=True)

        # The whole point of this arm: augmentation on vs off at equal epochs.
        aug = json.loads(RESULTS.read_text(encoding="utf-8"))
        delta_val = aug["best_val_accuracy"] - results["best_val_accuracy"]
        delta_test = aug["test_accuracy"] - results["test_accuracy"]
        comparison = pd.DataFrame([
            {"setting": "augmented", "augment": True, "epochs": aug["epochs_completed"],
             "best_val_acc": aug["best_val_accuracy"], "best_epoch": aug["best_epoch"],
             "test_acc": aug["test_accuracy"],
             "seconds_per_epoch_median": aug["seconds_per_epoch_median"]},
            {"setting": "no augmentation", "augment": False, "epochs": results["epochs"],
             "best_val_acc": results["best_val_accuracy"], "best_epoch": results["best_epoch"],
             "test_acc": results["test_accuracy"],
             "seconds_per_epoch_median": results["seconds_per_epoch_median"]},
        ])
        comparison.to_csv(ART / "cifar10_augmentation_comparison_20ep.csv", index=False)
        print("\n" + comparison.to_string(index=False))
        print("\naugmentation effect at %d epochs: %+.4f best-val, %+.4f test (%+.2f / %+.2f points)"
              % (epochs, delta_val, delta_test, 100 * delta_val, 100 * delta_test))
        print("wrote %s and %s" % (NOAUG_RESULTS.name,
                                   "cifar10_augmentation_comparison_20ep.csv"))
        return

    # ----------------------------------------------------------------- main run
    epochs = args.epochs or MAIN_EPOCHS

    if HISTORY.exists() and not args.force:
        df = pd.read_csv(HISTORY)
        if len(df) >= epochs:
            print("\nalready trained %d epochs -> evaluating best checkpoint" % len(df))
            model = keras.models.load_model(BEST)
        else:
            print("\nresuming: %d of %d epochs done" % (len(df), epochs))
            # Deliberately no re-compile: load_model restores the optimizer *and* its momentum
            # slots from the checkpoint. Compiling again would silently zero the momentum and
            # make the resumed run a different experiment from the uninterrupted one.
            model = keras.models.load_model(CKPT)
            done = len(df)
            timer = EpochTimer()
            writer = HistoryWriter(HISTORY, epochs, timer)
            writer.rows = df.to_dict("records")
            train_ds = make_dataset(x_tr, y_tr, BATCH, train=True, augment=True)
            val_ds = make_dataset(x_va, y_va, BATCH * 2, train=False, augment=False)
            model.fit(train_ds, validation_data=val_ds,
                      epochs=epochs - done, steps_per_epoch=int(np.ceil(len(x_tr) / BATCH)),
                      validation_steps=int(np.ceil(len(x_va) / (BATCH * 2))),
                      initial_epoch=done, verbose=2,
                      callbacks=[SetLR(epochs), timer, keras.callbacks.ModelCheckpoint(
                          filepath=str(BEST), monitor="val_accuracy", mode="max",
                          save_best_only=True, verbose=0), writer])
            df = pd.read_csv(HISTORY)
            model = keras.models.load_model(BEST)
    else:
        model, df, _ = run_training(x_tr, y_tr, x_va, y_va, epochs, "main run",
                                    augment=True, history_path=HISTORY)
        model = keras.models.load_model(BEST)

    ev = evaluate(model, x_test, y_test)
    trainable = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    total_w = int(model.count_params())

    results = {
        "dataset": "CIFAR-10",
        "tf_version": tf.__version__,
        "keras_version": keras.__version__,
        "trainable_params": trainable,
        "count_params_including_bn_buffers": total_w,
        "optimizer": f"SGD(momentum={MOMENTUM}, nesterov=True), lr {BASE_LR} cosine decay",
        "label_smoothing": LABEL_SMOOTHING,
        "batch_size": BATCH,
        "epochs_completed": int(len(df)),
        "seed": SEED,
        "best_val_accuracy": round(float(df["val_accuracy"].max()), 4),
        "best_epoch": int(df.loc[df["val_accuracy"].idxmax(), "epoch"]),
        "final_val_accuracy": round(float(df["val_accuracy"].iloc[-1]), 4),
        "test_accuracy": round(ev["test_accuracy"], 4),
        "seconds_per_epoch": round(float(df["epoch_seconds"].mean()), 1),
        # The mean is inflated by epochs that lost CPU to other work on a shared machine.
        # The median is the honest uncontended figure.
        "seconds_per_epoch_median": round(float(df["epoch_seconds"].median()), 1),
        "train_minutes": round(float(df["epoch_seconds"].sum()) / 60, 2),
        "worst_class": ev["per_class"].iloc[0].to_dict(),
        "best_class": ev["per_class"].iloc[-1].to_dict(),
    }
    RESULTS.write_text(json.dumps(results, indent=2), encoding="utf-8")
    model.save(FINAL)
    CKPT.unlink(missing_ok=True)

    print("=" * 70)
    print("trainable parameters : %s" % f"{trainable:,}")
    print("test accuracy        : %.4f" % ev["test_accuracy"])
    print("best val accuracy    : %.4f @ epoch %d" % (results["best_val_accuracy"],
                                                       results["best_epoch"]))
    print("worst / best class   : %s %.4f / %s %.4f"
          % (ev["per_class"].iloc[0]["class"], ev["per_class"].iloc[0]["accuracy"],
             ev["per_class"].iloc[-1]["class"], ev["per_class"].iloc[-1]["accuracy"]))
    print("seconds per epoch    : %.1f" % results["seconds_per_epoch"])
    print("=" * 70)
    print("wrote %s and %s" % (FINAL.name, RESULTS.name))


if __name__ == "__main__":
    main()