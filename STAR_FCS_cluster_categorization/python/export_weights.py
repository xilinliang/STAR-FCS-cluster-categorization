"""
Export a trained scikit-learn MLPClassifier or a PyTorch nn.Sequential of
Linear/ReLU/Tanh layers to the plain-text format StFcsMLP.h reads.

    # sklearn
    from export_weights import from_sklearn
    from_sklearn(clf, scaler, "fcs_ecal_category_mlp.txt")

    # pytorch
    from export_weights import from_torch
    from_torch(model, mean, std, "fcs_ecal_category_mlp.txt")

The last layer is written with activation 'softmax' so the maker gets
probabilities and its confidence cut means something.

There is deliberately no ONNX dependency here: nothing in the STAR library
stack at RCF loads ONNX, and for a network of this size a text file plus a
50-line forward pass is less trouble than a new external dependency.
"""

import numpy as np


def _write(path, mean, std, layers):
    """layers: list of (W, b, act) with W shaped (nout, nin)."""
    nin = layers[0][0].shape[1]
    mean = np.zeros(nin) if mean is None else np.asarray(mean, dtype=np.float64).ravel()
    std = np.ones(nin) if std is None else np.asarray(std, dtype=np.float64).ravel()
    assert mean.size == nin and std.size == nin, "mean/std length must match n inputs"

    def fmt(a):
        return " ".join(f"{v:.8g}" for v in np.asarray(a).ravel())

    with open(path, "w") as f:
        f.write("STARFCSMLP 1\n")
        f.write(f"NIN {nin}\n")
        f.write("MEAN " + fmt(mean) + "\n")
        f.write("STD " + fmt(std) + "\n")
        f.write(f"NLAYER {len(layers)}\n")
        for W, b, act in layers:
            nout, ni = W.shape
            f.write(f"LAYER {ni} {nout} {act}\n")
            f.write("W " + fmt(W) + "\n")   # row major, W[o*nin + i]
            f.write("B " + fmt(b) + "\n")
    print(f"wrote {path}: nin={nin} layers={len(layers)} nout={layers[-1][0].shape[0]}")


def from_sklearn(clf, scaler=None, path="fcs_ecal_category_mlp.txt"):
    """clf: sklearn.neural_network.MLPClassifier (activation 'relu' or 'tanh').
    scaler: the StandardScaler you fitted on X, or None if you did not scale."""
    act = {"relu": "relu", "tanh": "tanh", "logistic": "sigmoid"}[clf.activation]
    layers = []
    n = len(clf.coefs_)
    for i, (W, b) in enumerate(zip(clf.coefs_, clf.intercepts_)):
        a = "softmax" if i == n - 1 else act
        layers.append((np.asarray(W).T, np.asarray(b), a))  # sklearn stores (nin, nout)
    mean = getattr(scaler, "mean_", None) if scaler is not None else None
    std = getattr(scaler, "scale_", None) if scaler is not None else None
    _write(path, mean, std, layers)


def from_torch(model, mean=None, std=None, path="fcs_ecal_category_mlp.txt"):
    """model: torch.nn.Sequential of Linear / ReLU / Tanh / Sigmoid.
    Do NOT include a final softmax in the module: it is added here."""
    import torch.nn as nn

    layers = []
    pending = None
    for m in model:
        if isinstance(m, nn.Linear):
            if pending is not None:
                layers.append(pending)
            pending = [m.weight.detach().cpu().numpy(), m.bias.detach().cpu().numpy(), "linear"]
        elif isinstance(m, nn.ReLU):
            pending[2] = "relu"
        elif isinstance(m, nn.Tanh):
            pending[2] = "tanh"
        elif isinstance(m, nn.Sigmoid):
            pending[2] = "sigmoid"
        else:
            raise TypeError(f"unsupported layer {type(m)} - StFcsMLP only does dense layers")
    layers.append(pending)
    layers[-1][2] = "softmax"
    _write(path, mean, std, [tuple(l) for l in layers])


def selftest(path, X, proba):
    """Re-implement StFcsMLP.eval in numpy and compare against the framework's
    own probabilities. Run this before you trust the C++ side."""
    tok = []
    for line in open(path):
        line = line.split("#")[0]
        tok += line.split()
    i, nin, layers = 0, 0, []
    while i < len(tok):
        k = tok[i]
        if k == "STARFCSMLP":
            i += 2
        elif k == "NIN":
            nin = int(tok[i + 1]); i += 2
        elif k == "MEAN":
            mean = np.array(tok[i + 1:i + 1 + nin], dtype=float); i += 1 + nin
        elif k == "STD":
            std = np.array(tok[i + 1:i + 1 + nin], dtype=float); i += 1 + nin
        elif k == "NLAYER":
            i += 2
        elif k == "LAYER":
            ni, no, act = int(tok[i + 1]), int(tok[i + 2]), tok[i + 3]; i += 4
            assert tok[i] == "W"; i += 1
            W = np.array(tok[i:i + ni * no], dtype=float).reshape(no, ni); i += ni * no
            assert tok[i] == "B"; i += 1
            b = np.array(tok[i:i + no], dtype=float); i += no
            layers.append((W, b, act))
        else:
            i += 1
    v = (np.asarray(X, dtype=float) - mean) / np.where(std == 0, 1, std)
    for W, b, act in layers:
        v = v @ W.T + b
        if act == "relu":
            v = np.maximum(v, 0)
        elif act == "tanh":
            v = np.tanh(v)
        elif act == "sigmoid":
            v = 1 / (1 + np.exp(-v))
        elif act == "softmax":
            e = np.exp(v - v.max(axis=1, keepdims=True))
            v = e / e.sum(axis=1, keepdims=True)
    d = np.abs(v - np.asarray(proba)).max()
    print(f"max |exported - framework| = {d:.3e}")
    assert d < 1e-4, "exported weights do not reproduce the framework output"
    return v
