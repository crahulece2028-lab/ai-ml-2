"""Train the task 3.1 TensorFlow/Keras Fashion-MNIST model.

Training lives here rather than inline in the notebook because it is ~an hour of CPU
work: keeping it in a script means an interrupted run resumes instead of restarting,
and the notebook can be re-executed in seconds to regenerate every figure and table
from the saved artifacts.

The notebook `tensorflow_fashion_mnist.ipynb` contains the same model and training
code and reproduces every number below from these artifacts.

Usage
-----
    python train_fmnist.py            # trains only what is missing
    python train_fmnist.py --force    # discard artifacts and retrain from scratch
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

os_env = __import__("os").environ
os_env.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os_env.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import keras
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

SEED = 42
EPOCHS, BATCH, LR = 8, 64, 1e-3

ART = Path(__file__).resolve().parent / "artifacts"
MODEL_PATH = ART / "tf_fashion_mnist.keras"
HISTORY_PATH = ART / "tf_training_history.csv"
NOISE_PATH = ART / "tf_same_recipe_repeat.csv"
RESULTS_PATH = ART / "task3_1_results.json"

CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def build_model() -> keras.Sequential:
    """Week 2's SmallCNN, transcribed. use_bias=False is required for the
    parameter count to match PyTorch's 421,738 — Keras defaults Conv2D to
    use_bias=True, and BatchNorm already supplies a learned shift."""
    return keras.Sequential([
        keras.layers.Input(shape=(28, 28, 1)),
        keras.layers.Conv2D(32, 3, padding="same", use_bias=False),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(64, 3, padding="same", use_bias=False),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.MaxPooling2D(2),
        keras.layers.Flatten(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(10, activation="softmax"),
    ])


def compile_model() -> keras.Sequential:
    m = build_model()
    # from_logits=False is the Keras default and expects probabilities, which is why
    # build_model() ends in softmax. PyTorch's CrossEntropyLoss is the opposite
    # convention and expects raw logits.
    m.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LR),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return m


class BestWeights(keras.callbacks.Callback):
    """Snapshot weights whenever validation accuracy improves — the Keras equivalent
    of Week 2's `if val > best_val: snapshot = deepcopy(state_dict())`."""

    def __init__(self) -> None:
        super().__init__()
        self.best_val = -np.inf
        self.best_epoch: int | None = None
        self.snapshot = None

    def on_epoch_end(self, epoch, logs=None):
        v = logs["val_accuracy"]
        if v > self.best_val:
            self.best_val = float(v)
            self.best_epoch = epoch + 1
            self.snapshot = self.model.get_weights()


class EpochTimer(keras.callbacks.Callback):
    def __init__(self) -> None:
        super().__init__()
        self.seconds: list[float] = []
        self._t0 = None

    def on_epoch_begin(self, epoch, logs=None):
        self._t0 = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        self.seconds.append(time.perf_counter() - self._t0)


def load_data():
    (x_tr_all, y_tr_all), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    y_tr_all = y_tr_all.astype("int64")
    y_test = y_test.astype("int64")
    x_tr_all = x_tr_all.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_tr, x_val, y_tr, y_val = train_test_split(
        x_tr_all, y_tr_all, test_size=6000, random_state=SEED, stratify=y_tr_all
    )
    return x_tr, y_tr, x_val, y_val, x_test, y_test


def week2_reference() -> dict:
    """Read the Week 2 PyTorch numbers straight out of its checkpoint so this
    script cannot disagree with the model it is comparing against."""
    ckpt_path = (Path(__file__).resolve().parents[2]
                 / "week2" / "task2_2_cnn_classifier" / "artifacts" / "cnn_fashion_mnist.pt")
    if not ckpt_path.exists():
        return {}
    import torch

    ck = torch.load(ckpt_path, weights_only=False)
    sd = ck["model_state_dict"]
    params = sum(v.numel() for k, v in sd.items()
                 if v.dtype.is_floating_point and "running_" not in k)
    return {"parameters": int(params), "test_accuracy": float(ck["test_accuracy"]),
            "best_val_accuracy": float(ck["val_accuracy"]), "best_epoch": int(ck["best_epoch"])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.force:
        for p in (MODEL_PATH, HISTORY_PATH, NOISE_PATH, RESULTS_PATH):
            p.unlink(missing_ok=True)

    ART.mkdir(exist_ok=True)
    keras.utils.set_random_seed(SEED)

    print("tensorflow %s | keras %s | seed %d" % (tf.__version__, keras.__version__, SEED))
    print("devices: %s" % [d.device_type for d in tf.config.list_physical_devices()])

    x_tr, y_tr, x_val, y_val, x_test, y_test = load_data()
    ref = week2_reference()

    model = compile_model()
    # Keras' model.count_params() includes BatchNorm's moving_mean / moving_variance,
    # which are running buffers, not learnable parameters — so it reports 192 more than
    # PyTorch's model.parameters() on an otherwise identical network. Compare the
    # *trainable* counts, or the two frameworks will disagree by exactly 2*C_bn.
    keras_total_weights = int(model.count_params())
    keras_trainable = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    keras_params = keras_trainable
    print("keras trainable params : %s" % f"{keras_params:,}")
    print("keras count_params()   : %s  (includes %d BatchNorm running buffers)"
          % (f"{keras_total_weights:,}", keras_total_weights - keras_trainable))
    if ref:
        print("pytorch params         : %s  (identical: %s)"
              % (f"{ref['parameters']:,}", ref["parameters"] == keras_params))

    # ---- run A: the model we keep ------------------------------------------------
    if MODEL_PATH.exists() and HISTORY_PATH.exists() and not args.force:
        print("\nrun A already trained -> loading %s" % MODEL_PATH.name)
        model = keras.models.load_model(MODEL_PATH)
        hist = pd.read_csv(HISTORY_PATH)
        best_val = float(hist["val_accuracy"].max())
        best_epoch = int(hist.loc[hist["val_accuracy"].idxmax(), "epoch"])
        seconds_a = float("nan")
    else:
        tracker, timer = BestWeights(), EpochTimer()
        print("\n=== run A (%d epochs) ===" % EPOCHS)
        t0 = time.perf_counter()
        h = model.fit(x_tr, y_tr, validation_data=(x_val, y_val),
                      epochs=EPOCHS, batch_size=BATCH, shuffle=True, verbose=2,
                      callbacks=[tracker, timer])
        seconds_a = time.perf_counter() - t0
        hist = pd.DataFrame(h.history)
        hist.index = range(1, len(hist) + 1)
        hist.index.name = "epoch"
        hist["epoch_seconds"] = timer.seconds
        model.set_weights(tracker.snapshot)
        best_val, best_epoch = tracker.best_val, tracker.best_epoch
        model.save(MODEL_PATH)
        hist.to_csv(HISTORY_PATH)
        print("\nrun A: %.2f min | best val %.4f @ epoch %s | saved model"
              % (seconds_a / 60, best_val, best_epoch))

    # ---- run B: identical recipe, to size the noise -----------------------------
    if NOISE_PATH.exists() and not args.force:
        repeat = pd.read_csv(NOISE_PATH)
        print("\nrun B already trained -> loading repeat curve")
    else:
        print("\n=== run B (identical recipe, noise estimate) ===")
        t0 = time.perf_counter()
        h2 = compile_model().fit(x_tr, y_tr, validation_data=(x_val, y_val),
                                 epochs=EPOCHS, batch_size=BATCH, shuffle=True, verbose=2)
        seconds_b = time.perf_counter() - t0
        repeat = pd.DataFrame(h2.history)
        repeat.index = range(1, len(repeat) + 1)
        repeat.index.name = "epoch"
        repeat["epoch_seconds"] = np.nan
        repeat.to_csv(NOISE_PATH)
        print("\nrun B: %.2f min | best val %.4f" % (seconds_b / 60, repeat["val_accuracy"].max()))

    same_recipe_gap = abs(float(hist["val_accuracy"].max()) - float(repeat["val_accuracy"].max()))

    # ---- test evaluation --------------------------------------------------------
    probs = model.predict(x_test, batch_size=1000, verbose=0)
    y_pred = probs.argmax(axis=1)
    test_acc = float((y_pred == y_test).mean())
    test_loss = float(np.mean(keras.losses.sparse_categorical_crossentropy(y_test, probs)))

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
        {
            "class": name,
            "accuracy": float(cm[i, i] / cm[i].sum()),
            "support": int(cm[i].sum()),
            "most_confused_with": top_confusion(i),
        }
        for i, name in enumerate(CLASSES)
    ]).sort_values("accuracy").reset_index(drop=True)
    per_class.to_csv(ART / "tf_per_class_accuracy.csv", index=False)
    pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(ART / "tf_confusion_matrix.csv")

    results = {
        "framework": "tensorflow-keras",
        "tf_version": tf.__version__,
        "keras_version": keras.__version__,
        "parameters": keras_params,
        "keras_trainable_params": keras_trainable,
        "keras_count_params_including_bn_buffers": keras_total_weights,
        "pytorch_parameters": ref.get("parameters"),
        "architectures_identical": ref.get("parameters") == keras_params if ref else None,
        "optimizer": f"Adam(lr={LR})",
        "epochs": EPOCHS,
        "batch_size": BATCH,
        "seed": SEED,
        "best_val_accuracy": float(best_val),
        "best_epoch": int(best_epoch),
        "test_accuracy": test_acc,
        "test_loss": test_loss,
        "pytorch_test_accuracy": ref.get("test_accuracy"),
        "same_recipe_best_val_gap_points": float(100 * same_recipe_gap),
        "run_a_minutes": None if np.isnan(seconds_a) else round(seconds_a / 60, 2),
        "worst_class": per_class.iloc[0].to_dict(),
        "best_class": per_class.iloc[-1].to_dict(),
    }
    if ref and ref.get("test_accuracy"):
        results["test_delta_points"] = round(100 * (test_acc - ref["test_accuracy"]), 2)

    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n" + "=" * 66)
    print(classification_report(y_test, y_pred, target_names=CLASSES, digits=4, zero_division=0))
    print("test accuracy            : %.4f" % test_acc)
    print("worst / best class       : %s %.4f / %s %.4f"
          % (per_class.iloc[0]["class"], per_class.iloc[0]["accuracy"],
             per_class.iloc[-1]["class"], per_class.iloc[-1]["accuracy"]))
    if ref.get("test_accuracy"):
        print("PyTorch Week 2 test acc  : %.4f" % ref["test_accuracy"])
        print("difference               : %+.2f points" % (100 * (test_acc - ref["test_accuracy"])))
    print("same-recipe best-val gap : %.2f points" % (100 * same_recipe_gap))
    print("=" * 66)
    print("wrote %s" % RESULTS_PATH.name)


if __name__ == "__main__":
    main()