"""
Load the tree written by StFcsClusterFeatureMaker and build exactly the same
feature vector the C++ inference maker builds.

The 13-feature scalar vector is the CONTRACT between this file and
StFcsMLCategoryMaker::features(). If you change one, change the other, or the
model will see different numbers online than it saw in training.

Also provides the tower image (11x11) for image-based models, and the truth
labels for supervised training.

    from star_features import load
    d = load("fcsEcalClusterFeatures.root")
    X, img, y = d["X"], d["img"], d["y"]

Requires: uproot, numpy  (pip install uproot awkward numpy)
"""

import numpy as np
import uproot

NW = 11  # must match StFcsClusterFeatureMaker::kNW

FEATURE_NAMES = [
    "logE",
    "nTowers",
    "sigmaMax",
    "sigmaMin",
    "sigmaRatio",
    "theta",
    "seedFrac",
    "e2Frac",
    "e1e2Asym",
    "sigX",
    "sigY",
    "sigXY",
    "nNeighbor",
]

BRANCHES = [
    "run", "event", "det", "clid", "ncluDet",
    "e", "x", "y", "starx", "stary", "starz", "eta", "phi", "pt",
    "sigmaMax", "sigmaMin", "theta", "nTowers", "nNeighbor", "nPoints",
    "catStar", "chi2ndf1", "chi2ndf2",
    "seedId", "seedRow", "seedCol", "seedE", "seedFrac", "e2Frac", "e1e2Frac",
    "sigX", "sigY", "sigXY", "img", "mask",
    "nTrk", "trkId", "trkPid", "trkParent", "trkE", "trkPtot",
    "truthNPhoton", "truthSameParent", "truthPurity",
]


def build_features(t):
    """13-column float32 array, same order as StFcsMLCategoryMaker::features()."""
    e = np.asarray(t["e"], dtype=np.float64)
    smax = np.asarray(t["sigmaMax"], dtype=np.float64)
    smin = np.asarray(t["sigmaMin"], dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        logE = np.where(e > 0, np.log(np.maximum(e, 1e-12)), -10.0)
        ratio = np.where(smax > 0, smin / np.maximum(smax, 1e-12), 0.0)
    X = np.column_stack([
        logE,
        np.asarray(t["nTowers"], dtype=np.float64),
        smax,
        smin,
        ratio,
        np.asarray(t["theta"], dtype=np.float64),
        np.asarray(t["seedFrac"], dtype=np.float64),
        np.asarray(t["e2Frac"], dtype=np.float64),
        np.asarray(t["e1e2Frac"], dtype=np.float64),
        np.asarray(t["sigX"], dtype=np.float64),
        np.asarray(t["sigY"], dtype=np.float64),
        np.asarray(t["sigXY"], dtype=np.float64),
        np.asarray(t["nNeighbor"], dtype=np.float64),
    ])
    return X.astype(np.float32)


def build_images(t, normalize=True, keep_outside=False):
    """(N, NW, NW) float32 images centred on the seed tower.

    Cells outside the detector are stored as -1 by the maker. With
    keep_outside=False they become 0 and you lose the acceptance-edge
    information; keep them if your model should know about edges.
    """
    img = np.asarray(t["img"], dtype=np.float32).reshape(-1, NW, NW)
    if not keep_outside:
        img = np.where(img < 0, 0.0, img)
    if normalize:
        tot = img.sum(axis=(1, 2), keepdims=True)
        img = np.divide(img, tot, out=np.zeros_like(img), where=tot > 0)
    return img


def build_labels(t, purity_cut=0.8):
    """Truth category in the STAR convention.

        1 = single photon      (one photon dominates the cluster)
        2 = two photons        (two photons above threshold, e.g. a merged pi0)
        0 = anything else      (hadronic, pile-up, low purity)

    Returns (y, valid). valid is False where there is no usable truth, e.g. on
    data or where the deposit is too impure to label.

    This is the piece to argue about with your ePIC definition: adapt the rule
    here rather than in the C++ so the label definition stays in one place.
    """
    nph = np.asarray(t["truthNPhoton"], dtype=np.int32)
    pur = np.asarray(t["truthPurity"], dtype=np.float32)
    y = np.zeros(len(nph), dtype=np.int64)
    y[nph == 1] = 1
    y[nph >= 2] = 2
    valid = (nph >= 0) & ((nph >= 2) | (pur >= purity_cut))
    return y, valid


def load(filename, treename="clusters", purity_cut=0.8):
    with uproot.open(filename) as f:
        t = f[treename].arrays(BRANCHES, library="np")
    y, valid = build_labels(t, purity_cut)
    return {
        "raw": t,
        "X": build_features(t),
        "img": build_images(t),
        "y": y,
        "valid": valid,
        "catStar": np.asarray(t["catStar"], dtype=np.int64),
        "e": np.asarray(t["e"], dtype=np.float32),
        "det": np.asarray(t["det"], dtype=np.int64),
        "chi2ndf1": np.asarray(t["chi2ndf1"], dtype=np.float32),
        "chi2ndf2": np.asarray(t["chi2ndf2"], dtype=np.float32),
    }


def star_baseline(t, honour_ntowers=False):
    """Reproduce StFcsClusterMaker::categorization(), so you can quantify what
    the model has to beat.

    The code in StFcsClusterMaker reads:

        if (nTowers < 5) setCategory(1);          # no return!
        if (sigma > 1/2.5 + 0.003*e + 7.0/e) setCategory(2);
        else if (sigma < 1/2.1 - 0.001*e + 2.0/e) setCategory(1);
        else setCategory(0);

    the nTowers<5 assignment is overwritten unconditionally by the sigma block,
    so the effective cut is sigma-only. That is what this function reproduces by
    default; honour_ntowers=True gives the presumably intended behaviour, and
    comparing the two tells you how much that bug is costing you.
    """
    e = np.asarray(t["e"], dtype=np.float64)
    s = np.asarray(t["sigmaMax"], dtype=np.float64)
    nt = np.asarray(t["nTowers"], dtype=np.int32)
    with np.errstate(divide="ignore", invalid="ignore"):
        hi = 1.0 / 2.5 + 0.003 * e + 7.0 / np.maximum(e, 1e-6)
        lo = 1.0 / 2.1 - 0.001 * e + 2.0 / np.maximum(e, 1e-6)
    cat = np.zeros(len(e), dtype=np.int64)
    cat[s > hi] = 2
    cat[s < lo] = 1
    if honour_ntowers:
        cat[nt < 5] = 1
    return cat


if __name__ == "__main__":
    import sys

    d = load(sys.argv[1] if len(sys.argv) > 1 else "fcsEcalClusterFeatures.root")
    print("clusters:", len(d["y"]))
    print("features:", d["X"].shape, "images:", d["img"].shape)
    print("with usable truth:", int(d["valid"].sum()))
    for c in (0, 1, 2):
        print(f"  STAR category {c}: {(d['catStar'] == c).sum():8d}"
              f"   truth {c}: {((d['y'] == c) & d['valid']).sum():8d}")
