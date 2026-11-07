"""Faithful re-implementation of the YODA pipeline (Vioque et al. 2020; github.com/MVioque/YODA @0c8097d).

The original YODA.py cannot run as published (``train_test_split`` is never imported and it depends
on the abandoned ``keras_metrics`` package), so the steps below re-implement it line by line:

* 13 base characteristics -> 3 non-passband columns + all 45 pairwise passband colours = 48 observables
  (same loop order as YODA.py)
* StandardScaler -> PCA, number of components chosen by YODA.py's rule var[i+1]/var[i] <= 0.01
* Dense(580, relu, L2 .01) -> Dropout(.5) -> BatchNorm -> Dense(580, relu, L2 .01) -> Dropout(.5)
  -> BatchNorm -> Dense(3, softmax, L2 .01); categorical cross-entropy; Adamax; full batch;
  class_weight='balanced'; validation_split=0.1; early stopping on validation PMS precision
  (p >= 0.5) with patience 50, mode max, no weight restore (as in YODA.py).

``predict_proba`` deliberately performs NO input validation, as in the original: this is the
behaviour that claim C2 is tested against. ``validate_inputs`` (src/a5/guard.py) is only used when
the improvement is switched on.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field

import numpy as np

# Column order of the 13 base characteristics (3 non-passband + 10 passbands, as in YODA.py).
NONPASS = ["rHa", "Gvar", "Vhtg"]
PASSBANDS = ["BPmag", "Gmag", "RPmag", "Jmag", "Hmag", "Ksmag", "W1mag", "W2mag", "W3mag", "W4mag"]
BASE13 = NONPASS + PASSBANDS


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def expand_48(x13: np.ndarray) -> np.ndarray:
    """Replicates YODA.py: keep first 3 columns, then colours x[:, a+3] - x[:, b+4+a]."""
    x13 = np.asarray(x13, dtype=float)
    if x13.ndim != 2 or x13.shape[1] != 13:
        raise ValueError(f"expected (n, 13) array, got {x13.shape}")
    cols = [x13[:, 0], x13[:, 1], x13[:, 2]]
    n = x13.shape[1]
    a = 0
    while a + 3 < n:
        b = 0
        while b + 4 + a < n:
            cols.append(x13[:, a + 3] - x13[:, b + 4 + a])
            b += 1
        a += 1
    return np.column_stack(cols)


def choose_components(variance: np.ndarray) -> int:
    """YODA.py rule: first i with variance[i+1]/variance[i] <= 0.01 -> i+1 components."""
    for i in range(len(variance) - 1):
        if variance[i + 1] / variance[i] <= 0.01:
            return i + 1
    return len(variance)


@dataclass
class YodaModel:
    seed: int
    max_epochs: int = 3000
    patience: int = 50
    neurons: int = 580
    dropout: float = 0.5
    l2: float = 0.01
    scaler: object = None
    pca: object = None
    model: object = None
    n_components: int = 0
    history: dict = field(default_factory=dict)
    epochs_run: int = 0

    # ------------------------------------------------------------------ features
    def features(self, x13: np.ndarray) -> np.ndarray:
        return self.pca.transform(self.scaler.transform(expand_48(x13)))

    # ------------------------------------------------------------------ training
    def fit(self, x13: np.ndarray, y: np.ndarray, verbose: int = 0) -> "YodaModel":
        import keras
        from keras import layers, regularizers
        from sklearn import preprocessing
        from sklearn.decomposition import PCA
        from sklearn.utils import class_weight

        set_global_seed(self.seed)
        x48 = expand_48(x13)
        self.scaler = preprocessing.StandardScaler().fit(x48)
        xs = self.scaler.transform(x48)
        full = PCA(n_components=xs.shape[1], random_state=self.seed).fit(xs)
        self.n_components = choose_components(full.explained_variance_)
        self.pca = PCA(n_components=self.n_components, random_state=self.seed).fit(xs)
        xr = self.pca.transform(xs).astype("float32")

        classes = np.unique(y)
        w = class_weight.compute_class_weight("balanced", classes=classes, y=y)
        cw = {int(c): float(v) for c, v in zip(classes, w)}
        onehot = keras.utils.to_categorical(y, num_classes=3)

        reg = regularizers.l2(self.l2)
        m = keras.Sequential([
            keras.Input(shape=(self.n_components,)),
            layers.Dense(self.neurons, activation="relu", kernel_regularizer=reg),
            layers.Dropout(self.dropout),
            layers.BatchNormalization(),
            layers.Dense(self.neurons, activation="relu", kernel_regularizer=reg),
            layers.Dropout(self.dropout),
            layers.BatchNormalization(),
            layers.Dense(3, activation="softmax", kernel_regularizer=reg),
        ])
        m.compile(loss="categorical_crossentropy", optimizer="adamax",
                  metrics=[keras.metrics.Precision(thresholds=0.5, class_id=1, name="precision")])
        stop = keras.callbacks.EarlyStopping(monitor="val_precision", mode="max", patience=self.patience,
                                             min_delta=0, restore_best_weights=False)
        h = m.fit(xr, onehot, epochs=self.max_epochs, validation_split=0.1, batch_size=len(xr),
                  class_weight=cw, callbacks=[stop], verbose=verbose, shuffle=True)
        self.model = m
        self.history = {k: [float(v) for v in vals] for k, vals in h.history.items()}
        self.epochs_run = len(h.history["loss"])
        return self

    # ------------------------------------------------------------------ inference
    def predict_proba(self, x13: np.ndarray) -> np.ndarray:
        """No validation (faithful to YODA.py)."""
        f = self.features(x13).astype("float32")
        return self.model.predict(f, batch_size=65536, verbose=0)


def classify(p: np.ndarray, thr: float = 0.5) -> np.ndarray:
    """Class at p >= thr: 1 = PMS, 2 = CBe, 0 = other / no class reaches thr.

    With 3 softmax outputs, at most one class can have p >= 0.5 (ties at exactly 0.5 resolve to the
    lower index). NaN probabilities map to -1 ('undefined').
    """
    p = np.asarray(p, dtype=float)
    out = np.zeros(len(p), dtype=int)
    out[p[:, 2] >= thr] = 2
    out[p[:, 1] >= thr] = 1
    out[~np.isfinite(p).all(axis=1)] = -1
    return out
