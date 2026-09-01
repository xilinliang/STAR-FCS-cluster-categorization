# Porting an ePIC-style cluster categorization framework into STAR FCS ECal

Target: `FCS ECal north/south` (detector id 0 and 1), running at RCF on top of
`FCS-ECal-pi0-reconstruction`. Written to slot into that repo with no changes to
your existing maker.

---

## 1. The hook already exists

STAR's FCS reconstruction has exactly the concept you want to replace, and it is
already wired into the photon fitter. `StFcsCluster` carries an `int category()`,
and `StFcsPointMaker` branches on it:

```cpp
switch (c->category()) {
  case 0:  chi1 = fit1PhotonCluster(c,&point0);
           chi2 = fit2PhotonCluster(c,&point1,&point2); break;   // try both
  case 1:  chi1 = fit1PhotonCluster(c,&point0);          break;   // 1 photon
  case 2:  chi2 = fit2PhotonCluster(c,&point1,&point2);  break;   // 2 photons
}
```

So the port is not "add an ML stage to STAR reconstruction". It is: **compute
`category` with your model instead of with the hard-coded cut**, in a maker that
sits between `StFcsClusterMaker` and `StFcsPointMaker`. Everything downstream —
points, your π⁰ finder, the gain-calibration iteration — picks it up for free.

The cut you are replacing, `StFcsClusterMaker::categorization()`:

```cpp
if (cluster->nTowers() < 5) cluster->setCategory(1);            // <-- no return
const double sigma = cluster->sigmaMax();
const double e     = cluster->energy();
if      (sigma > 1/2.5 + 0.003*e + 7.0/e) cluster->setCategory(2);
else if (sigma < 1/2.1 - 0.001*e + 2.0/e) cluster->setCategory(1);
else                                      cluster->setCategory(0);
```

Note the first assignment falls through and is overwritten unconditionally, so
the effective STAR categorizer today is sigma-only; `nTowers < 5` does nothing.
Worth confirming against the version of `StFcsClusterMaker` in your library
before you quote it — and worth a line in your talk, because it is part of the
baseline you will be measured against. `python/star_features.py::star_baseline`
reproduces both variants.

## 2. What maps onto what

| ePIC side | STAR FCS side |
|---|---|
| `edm4eic::CalorimeterHit` | `StFcsHit` — `id()`, `energy()`, `getGeantTracks()` |
| `edm4eic::Cluster` | `StFcsCluster` — `energy()`, `x()`, `y()`, `sigmaMax/Min()`, `theta()`, `nTowers()`, `hits()` |
| cluster → subclusters / splitting | `StFcsPoint` produced by `StFcsPointMaker`'s 1γ/2γ Minuit shower-shape fit |
| JANA2 algorithm + factory | `StMaker` subclass in `StRoot/`, added to the chain in the run macro |
| podio collection I/O | `StEvent` → `StFcsCollection::clusters(det)` |
| MC truth via `edm4hep::MCParticle` associations | `StFcsHit::getGeantTracks()` (filled by `StFcsFastSimulatorMaker`) + the `g2t_track` table |
| cell (x,y,z) in cluster local frame | `x()`, `y()` are in **column/row units**, not cm — convert with `StFcsDb::getStarXYZfromColumnRow(det, col, row)` |

Two gotchas that cost people a day each:

- **Cluster x/y are column/row, not centimetres.** Your existing maker already
  does this correctly (`getStarXYZfromColumnRow`), so copy that habit into any
  feature you define in cm.
- **Row/column indices are 1-based** in `StFcsDb::getRowNumber/getColumnNumber`.
  ECal is 34 rows × 22 columns = 748 towers per half, tower pitch ~5.5 cm.

## 3. What is in this package

```
StRoot/StFcsClusterFeatureMaker/   dump one tree entry per ECal cluster
StRoot/StFcsMLCategoryMaker/       apply the trained model, set category()
  └ StFcsMLP.h                     dependency-free dense-net evaluator (2nd backend)
trainTMVA.C                        TMVA multiclass training on the dumped tree
runMudst_ml.C                      your runMudst.C with both makers wired in
python/star_features.py            same 13 features, in numpy, + truth labels
python/export_weights.py           sklearn/PyTorch → StFcsMLP text format
```

Drop the two `StRoot/` directories into your working folder next to
`StFcsPi0FinderForEcal`, then `stardev; cons`.

**The 13-feature vector is a contract.** It is defined in three places that must
never drift: `StFcsMLCategoryMaker::features()`, `star_features.build_features()`,
and the `varname[]` list in `trainTMVA.C`. Names matter too — TMVA matches
training to application by variable *name*.

```
logE  nTowers  sigmaMax  sigmaMin  sigmaRatio  theta
seedFrac  e2Frac  e1e2Asym  sigX  sigY  sigXY  nNeighbor
```

The feature maker additionally stores an 11×11 tower-energy image centred on the
seed tower, plus a mask of which towers clustering assigned to this cluster.
That is there for an image/GNN model of the ePIC kind; the 13 scalars are the
TMVA path.

## 4. Sequence I would actually follow

1. **Dump features on simulation.** `mode=0` in `runMudst_ml.C`, on a sample with
   `StFcsFastSimulatorMaker` in the chain — it calls `addGeantTrack(track_p, de)`
   on every hit, which is where the labels come from. Also dump a data sample:
   you need the data/MC comparison of the input distributions before you trust
   anything trained on simulation.
2. **Check the label definition.** `build_labels()` in `star_features.py` calls a
   cluster "2 photons" when ≥2 photons each deposit >10% of the cluster energy.
   That is a guess at your ePIC convention — replace it with yours; it is the one
   place the physics definition lives.
3. **Train.** `root4star -b -q 'trainTMVA.C("feat.root","FcsCat")'` gives you
   multiclass BDTG and MLP weight files plus the usual TMVA GUI output.
4. **Apply in QA mode first.** `mlcat->setMode(0)` changes nothing but fills the
   STAR-category vs ML-category migration matrix. Look at it before you let the
   model touch reconstruction.
5. **Turn it on and measure the physics.** `setMode(1)`, rerun your π⁰ chain, and
   compare `h1_inv_mass_point` / `h1_inv_mass_cluster` before and after: peak
   position, width, signal/background, and yield per tower. That last one matters
   for your gain iteration — if the model changes which clusters get split, the
   per-tower mass fits shift, and you want to know that before a calibration
   pass, not after.

## 5. TMVA specifics

- **Train with the ROOT that `root4star` uses.** Run `trainTMVA.C` under
  `root4star` after `starver`. TMVA weight XML is not reliably portable across
  ROOT major versions, and a model trained in a conda ROOT that silently fails to
  book in the chain is the single most common way this goes wrong.
- `AnalysisType=multiclass` and `Reader::EvaluateMulticlass` return one response
  per class; the maker takes the argmax. `setMode(2)` plus `setConfidence(x)`
  only overrides STAR's category when the winning response exceeds `x`, which is
  a conservative way to deploy: you keep the old behaviour everywhere the model
  is unsure.
- `gSystem->Load("libTMVA")` must come **before** loading
  `StFcsMLCategoryMaker`.
- BDTG is the sane first method here — 13 low-level shape variables, no scaling
  needed, and it trains in minutes. Use the MLP as a cross-check, not as the
  headline.

If instead you want to keep your ePIC model as-is (PyTorch/sklearn), set
`setBackend(1)` and export with `python/export_weights.py`. That path needs no
ROOT/TMVA at all and is verified: the C++ evaluator reproduces the Python
forward pass to <1e-7. A convolutional or graph model does **not** fit that
format — for those, either extend `StFcsMLP.h` with conv layers or keep
inference offline on the dumped tree.

## 6. Things to decide

- **Class definition.** Three classes matching STAR's 0/1/2 is what makes the
  drop-in work. If your ePIC framework has a different taxonomy (photon /
  merged π⁰ / hadron / noise), you need a mapping onto 0/1/2 — or a wider change
  where the extra classes are stored on the cluster and used downstream. The
  first is a week; the second is a proposal to the FCS group.
- **Data/MC.** The FCS fast simulator is a fast simulator. Shape variables like
  `sigmaMax` and `seedFrac` are exactly the ones that suffer. Plan on comparing
  the 13 input distributions in data and MC, and on a reweighting or a
  data-driven check (e.g. π⁰ mass peak in a low-multiplicity sample) before
  claiming a performance number.
- **Where it runs.** If this is only for your π⁰ analysis, a private library and
  your scheduler jobs are enough. If it is meant for the collaboration's
  production chain, it needs to go into `star-sw` with Akio's sign-off, and the
  weight file needs a home (`StarDb` or a versioned path) rather than a relative
  filename in a macro.

## 7. Not verified here

I wrote these against the STAR doxygen for `StFcsCluster`, `StFcsHit`,
`StFcsCollection` and `StFcsDb`, but nothing in `StRoot/` was compiled — there is
no STAR library stack in this session. Expect to fix include paths and maybe a
signature or two on the first `cons`. The one piece that *is* tested is
`StFcsMLP.h` against `export_weights.py`.
