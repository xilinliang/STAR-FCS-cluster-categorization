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
NW5 = 5  # must match StFcsMLCategoryMaker::kNW5

# must match StFcsMLCategoryMaker::kTowerFractions and kTowerFractions in trainTMVA.C
TOWER_FRACTIONS = True
W0 = 4.5  # must match StFcsMLCategoryMaker::kW0x10 / 10

FEATURE_NAMES_13 = [
    "logE", "nTowers", "sigmaMax", "sigmaMin", "sigmaRatio", "theta",
    "seedFrac", "e2Frac", "e1e2Asym", "sigX", "sigY", "sigXY", "nNeighbor",
]

FEATURE_NAMES_34 = (
    ["e", "x", "y", "nHits", "radius", "dispersion", "sigmaMin", "sigmaMax"]
    + [f"t{r}{c}" for r in range(NW5) for c in range(NW5)]
    + ["eOut"]
)


def feature_names(feature_set=13):
    return FEATURE_NAMES_34 if feature_set == 34 else FEATURE_NAMES_13

BRANCHES = [
    "run", "event", "det", "clid", "ncluDet",
    "e", "x", "y", "starx", "stary", "starz", "eta", "phi", "pt",
    "sigmaMax", "sigmaMin", "theta", "nTowers", "nNeighbor", "nPoints",
    "catStar", "chi2ndf1", "chi2ndf2",
    "seedId", "seedRow", "seedCol", "seedE", "seedFrac", "e2Frac", "e1e2Frac",
    "xw", "yw", "sigX", "sigY", "sigXY", "img", "mask",
    "nTrk", "trkId", "trkPid", "trkParent", "trkE", "trkPtot",
    "truthNPhoton", "truthSameParent", "truthPurity",
]


def build_features(t, feature_set=13):
    """Feature matrix, same order and definitions as
    StFcsMLCategoryMaker::features(clu, db, feature_set).

    feature_set=13 (default): shape summary variables, read straight off the
        dumped tree.
    feature_set=34: the ePIC-style set, derived from the cluster's own towers
        (img * mask) so training and inference see the same numbers.
    """
    if feature_set != 34:
        return _build_features_13(t)
    return _build_features_34(t)


def _build_features_13(t):
    e = np.asarray(t["e"], dtype=np.float64)
    smax = np.asarray(t["sigmaMax"], dtype=np.float64)
    smin = np.asarray(t["sigmaMin"], dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        logE = np.where(e > 0, np.log(np.maximum(e, 1e-12)), 0.0)
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
    assert X.shape[1] == len(FEATURE_NAMES_13) == 13
    return X.astype(np.float32)


def _build_features_34(t):
    e = np.asarray(t["e"], dtype=np.float64)
    n = len(e)
    half = NW // 2
    half5 = NW5 // 2

    img = np.asarray(t["img"], dtype=np.float64).reshape(n, NW, NW)
    mask = np.asarray(t["mask"], dtype=np.float64).reshape(n, NW, NW)
    cl = np.where(mask > 0, np.maximum(img, 0.0), 0.0)  # cluster towers only

    # offsets of every pixel from the seed tower, then absolute row/column
    dr = np.arange(NW) - half
    dc = np.arange(NW) - half
    DR, DC = np.meshgrid(dr, dc, indexing="ij")
    row = np.asarray(t["seedRow"], dtype=np.float64)[:, None, None] + DR[None]
    col = np.asarray(t["seedCol"], dtype=np.float64)[:, None, None] + DC[None]

    xw = np.asarray(t["xw"], dtype=np.float64)[:, None, None]
    yw = np.asarray(t["yw"], dtype=np.float64)[:, None, None]
    xc = np.asarray(t["x"], dtype=np.float64)[:, None, None]
    yc = np.asarray(t["y"], dtype=np.float64)[:, None, None]
    d2 = ((col - xc) * xw) ** 2 + ((row - yc) * yw) ** 2

    # radius: linear energy weights
    wtot = cl.sum(axis=(1, 2))
    radius = np.sqrt(np.divide((cl * d2).sum(axis=(1, 2)), wtot,
                               out=np.zeros(n), where=wtot > 0))

    # dispersion: logarithmic weights w_i = max(0, W0 + ln(E_i/E))
    with np.errstate(divide="ignore", invalid="ignore"):
        lw = W0 + np.log(np.divide(cl, e[:, None, None], out=np.zeros_like(cl),
                                   where=(cl > 0) & (e[:, None, None] > 0)))
    lw = np.where((cl > 0) & (lw > 0), lw, 0.0)
    lwtot = lw.sum(axis=(1, 2))
    dispersion = np.sqrt(np.divide((lw * d2).sum(axis=(1, 2)), lwtot,
                                   out=np.zeros(n), where=lwtot > 0))

    t5 = cl[:, half - half5:half + half5 + 1, half - half5:half + half5 + 1]
    t5 = t5.reshape(n, NW5 * NW5)
    eOut = e - t5.sum(axis=1)

    if TOWER_FRACTIONS:
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(e > 0, 1.0 / np.maximum(e, 1e-12), 0.0)[:, None]
        t5 = t5 * scale
        eOut = eOut * scale[:, 0]

    X = np.column_stack([
        e,
        np.asarray(t["x"], dtype=np.float64),
        np.asarray(t["y"], dtype=np.float64),
        np.asarray(t["nTowers"], dtype=np.float64),
        radius,
        dispersion,
        np.asarray(t["sigmaMin"], dtype=np.float64),
        np.asarray(t["sigmaMax"], dtype=np.float64),
        t5,
        eOut,
    ])
    assert X.shape[1] == len(FEATURE_NAMES_34) == 34
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


def load(filename, treename="clusters", purity_cut=0.8, feature_set=13):
    with uproot.open(filename) as f:
        t = f[treename].arrays(BRANCHES, library="np")
    y, valid = build_labels(t, purity_cut)
    return {
        "raw": t,
        "X": build_features(t, feature_set),
        "names": feature_names(feature_set),
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
