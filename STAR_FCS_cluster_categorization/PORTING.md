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
baseline you will be measured against. You do not have to reimplement it: the
feature dumper stores whatever `StFcsClusterMaker` decided in the `catStar`
branch, so the baseline is one branch away in every comparison.

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
StRoot/StFcsClusterFeatureMaker/   dump one tree entry per ECal cluster (MuDst)
StRoot/StFcsMLCategoryMaker/       apply the trained model, set category() (MuDst)
  ├ StFcsClusterFeatures.h         THE definition of the input variables
  └ StFcsMLP.h                     dense-net evaluator (non-TMVA backend only)
StRoot/StFcsPicoCategoryMaker/     apply the model to StPicoDst input
trainTMVA.C                        TMVA multiclass training on the dumped tree
testFeatures.C                     smoke test for the feature definitions
runMudst_ml.C                      your runMudst.C with both makers wired in
runPicoDst_ml.C                    the same, for picoDst input
BUILD.md                           SL7 container, cons, and job submission
optional_python/                   numpy reader + PyTorch exporter, not needed
```

Pure C++/ROOT — nothing to install, TMVA comes with ROOT. See `BUILD.md` for the
container and `cons` details. Drop the two `StRoot/` directories into your
working folder next to `StFcsPi0FinderForEcal` and build.

### The input variables

Defined once, in `StFcsClusterFeatures.h`. Both `trainTMVA.C` and
`StFcsMLCategoryMaker` include that header and call the same `compute()`, so
training and inference literally run the same code — there is no second
implementation to drift. Two sets, chosen with `setFeatureSet()` and the third
argument of `trainTMVA.C`:

**Set 13 — the default, start here.**

| # | Name | Definition | Where it comes from |
|---|---|---|---|
| 0 | `logE` | ln(E) | `clu->energy()` |
| 1 | `nTowers` | towers in the cluster | `clu->nTowers()` |
| 2 | `sigmaMax` | width along the major principal axis | `clu->sigmaMax()` |
| 3 | `sigmaMin` | width along the minor axis | `clu->sigmaMin()` |
| 4 | `sigmaRatio` | `sigmaMin / sigmaMax` | derived |
| 5 | `theta` | principal-axis angle | `clu->theta()` |
| 6 | `seedFrac` | e₁ / E | loop over `clu->hits()` |
| 7 | `e2Frac` | e₂ / E | loop over `clu->hits()` |
| 8 | `e1e2Asym` | (e₁ − e₂)/(e₁ + e₂) | loop over `clu->hits()` |
| 9 | `sigX` | √(⟨col²⟩ − ⟨col⟩²), energy-weighted | loop over `clu->hits()` |
| 10 | `sigY` | √(⟨row²⟩ − ⟨row⟩²), energy-weighted | loop over `clu->hits()` |
| 11 | `sigXY` | ⟨col·row⟩ − ⟨col⟩⟨row⟩ | loop over `clu->hits()` |
| 12 | `nNeighbor` | neighbouring clusters | `clu->nNeighbor()` |

e₁ and e₂ are the highest and second-highest tower energies in the cluster.

Only two sources feed this. Six variables are read straight off `StFcsCluster`,
where `StFcsClusterMaker` has already computed them — `sigmaMax`, `sigmaMin` and
`theta` being the eigen-decomposition of the energy-weighted covariance matrix
of the tower positions. The other seven come from a single loop over
`clu->hits()`, where each `StFcsHit` gives `energy()` and an `id()` that
`StFcsDb::getRowNumber` / `getColumnNumber` turn into a row and column. **Row and
column are cell units, not cm** — the STAR FCS convention, which is why `sigX`
and `sigY` are dimensionless here while `radius` in set 34 is in cm.

Why these separate one photon from two: a merged π⁰ splits its energy between
towers, so `seedFrac` drops and `e1e2Asym` → 0, while a single photon
concentrates in one tower. `sigmaMax` grows along the axis joining the two
showers while `sigmaMin` does not, so `sigmaRatio` → 0 for a genuine two-photon
cluster and stays near 1 for a round one. `logE` is in because every one of
those thresholds moves with energy — which is exactly what STAR's hand-tuned cut
encodes with its `+0.003*e` and `7.0/e` terms, and what the model gets to learn
from data instead.

One redundancy worth knowing about: variables 2–5 and 9–11 are the *same*
second-moment matrix in two bases. `sigmaMax`/`sigmaMin`/`theta` are its
eigenvalues and rotation angle; `sigX`/`sigY`/`sigXY` are its raw elements.
Feeding a BDT both is not wrong — the rotation-invariant pair is the more
physical, the raw moments keep the detector frame — but they are strongly
correlated, so do not read the TMVA variable ranking as if they were independent
handles. Dropping `sigX`/`sigY`/`sigXY` costs almost nothing and takes you to 10.

Shape summary only, no positional tower grid: much less room to learn a
fast-simulator artefact than set 34, trains in a couple of minutes, and every
variable is one you can plot against data on its own and defend.

**Set 34 — the ePIC-style set, for the comparison.**

```
 0  e            cluster energy [GeV]
 1  x            centroid, COLUMN units (not cm)
 2  y            centroid, ROW units
 3  nHits        towers in the cluster
 4  radius       sqrt(Σ E_i d_i² / Σ E_i) [cm], d_i from the centroid
 5  dispersion   same with log weights w_i = max(0, 4.5 + ln(E_i/E)) [cm]
 6  sigmaMin
 7  sigmaMax
 8..32  t00..t44 5×5 tower energies around the seed, cluster towers only
 33 eOut         cluster energy outside that 5×5
```

The set is part of the contract with the weight file, so it goes in the file name
(`FcsCat13_BDTG.weights.xml`). A mismatch shows up as TMVA refusing to book the
method, because the variable names do not match — noisy, which is what you want.

`root -b -q testFeatures.C` checks the definitions against synthetic 1γ and 2γ
tower patterns. Run it after any change to the header, before retraining.

The feature maker additionally stores an 11×11 tower-energy image centred on the
seed tower, plus a mask of which towers clustering assigned to this cluster. The
5×5 is cut out of that, so switching to 7×7 or to a CNN needs no new production.

### Choices baked into set 34, and why

- **Tower energies are stored as fractions of E** (`kTowerFractions = true`, in
  all three files — change it in all three or the model silently sees different
  numbers online than in training). Raw tower energies in GeV make the model
  learn the energy spectrum of the training sample, which is precisely the thing
  that differs most between the fast simulator and data. `e` is still input 0, so
  no information is lost — it is just factorized.
- **`x` and `y` are a hazard in your particular analysis.** They let the model
  learn detector-region-specific behaviour, including which towers are badly
  gained. You are using this inside a gain-calibration loop, so that is circular:
  the categorizer could encode the miscalibration it is supposed to be blind to,
  and the per-tower π⁰ mass fits would inherit it. I kept them because you asked
  for the 34, but I would train once with and once without and compare the
  per-tower mass fits before shipping. Dropping them means deleting entries 1–2
  from all three lists and retraining.
- **The 5×5 is not rotated.** A two-photon cluster has an axis; without rotating
  into the principal-axis frame (`theta` is right there in `StFcsCluster`) the
  model has to learn the same shape in every orientation, which is a real cost
  for a BDT with 25 positionally-indexed inputs. Options, in increasing effort:
  add `theta` as a 35th variable, rotate the 5×5 into the principal-axis frame
  before flattening, or move to a CNN on the 11×11 image.
- **`radius` and `dispersion` use the definitions in the header comment.** If
  your ePIC framework defines them differently — and "dispersion" in particular
  is used for at least three different quantities in the literature — change
  them in `StFcsClusterFeatures.h` (one place) and rerun `testFeatures.C`.
- **Granularity is not transferable.** FCS ECal towers are ~5.5 cm; a 5×5 there
  covers a different number of Molière radii than a 5×5 in the ePIC ECal. The
  same 34 variables therefore mean physically different things in the two
  detectors. Retrain on STAR simulation from scratch — do not port ePIC weights.

## 4. Sequence I would actually follow

1. **Dump features on simulation.** `mode=0` in `runMudst_ml.C`, on a sample with
   `StFcsFastSimulatorMaker` in the chain — it calls `addGeantTrack(track_p, de)`
   on every hit, which is where the labels come from. Also dump a data sample:
   you need the data/MC comparison of the input distributions before you trust
   anything trained on simulation.
2. **Check the label definition.** The class assignment at the top of the event
   loop in `trainTMVA.C` calls a cluster "2 photons" when ≥2 photons each deposit
   >10% of the cluster energy (the 10% lives in `StFcsClusterFeatureMaker`'s
   `mTruthFrac`). That is a guess at your ePIC convention — replace it with
   yours; it is the one place the physics definition lives.
3. **Train.** `root4star -b -q 'trainTMVA.C("feat.root","FcsCat",13)'` gives you
   multiclass BDTG and MLP weight files plus the usual TMVA GUI output. Repeat
   with `34` when you want the comparison — same command, different weight file.
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
- BDTG is the sane first method — no input scaling needed and it trains in
  minutes. The MLP is a cross-check, not the headline.
- When you do run set 34, watch the TMVA variable ranking: if the 5×5 towers rank
  far below `sigmaMax` and `radius`, the tree is not extracting shower shape from
  the raw grid, and a CNN on the stored 11×11 image is the better use of your
  time than more trees. 34 inputs is also more room to overfit a fast-simulator
  artefact — watch the overtraining check.
- Nothing to install. TMVA is part of ROOT and ROOT is part of the STAR stack;
  the SL7 container is read-only anyway. `BUILD.md` has the full recipe.

The `kTextMLP` backend and `optional_python/` exist only for the case where you
would rather train outside ROOT (PyTorch/sklearn). For TMVA they are dead weight
— ignore them.

## 6. Things to decide

- **Class definition.** Three classes matching STAR's 0/1/2 is what makes the
  drop-in work. If your ePIC framework has a different taxonomy (photon /
  merged π⁰ / hadron / noise), you need a mapping onto 0/1/2 — or a wider change
  where the extra classes are stored on the cluster and used downstream. The
  first is a week; the second is a proposal to the FCS group.
- **Data/MC.** The FCS fast simulator is a fast simulator. Shape variables like
  `sigmaMax` and `seedFrac` are exactly the ones that suffer. Plan on comparing
  the input distributions in data and MC, and on a reweighting or a
  data-driven check (e.g. π⁰ mass peak in a low-multiplicity sample) before
  claiming a performance number.
- **Where it runs.** If this is only for your π⁰ analysis, a private library and
  your scheduler jobs are enough. If it is meant for the collaboration's
  production chain, it needs to go into `star-sw` with Akio's sign-off, and the
  weight file needs a home (`StarDb` or a versioned path) rather than a relative
  filename in a macro.

## 7. picoDst input

`StFcsPicoCategoryMaker` + `runPicoDst_ml.C` run the categorization on
`StPicoDst` files instead of MuDst. One thing decides how that path looks:

**picoDst does not store which towers belong to a cluster.**
`StPicoDstMaker::fillFcsClusters()` copies the scalar summary of each
`StMuFcsCluster` — id, detectorId, category, nTowers, x, y, sigmaMin, sigmaMax,
theta, chi2Ndf1/2Photon, four-momentum — and drops `StMuFcsCluster::hits()`.
`FcsHits` is written as a separate flat collection with no back-pointer. The
association exists in the MuDst one step upstream and is lost in the conversion.

So seven of the thirteen variables (`seedFrac`, `e2Frac`, `e1e2Asym`, `sigX`,
`sigY`, `sigXY`, and `nNeighbor`) cannot be computed from a picoDst. Hence
**feature set 6**: `logE`, `nTowers`, `sigmaMax`, `sigmaMin`, `sigmaRatio`,
`theta` — defined identically to the first six of set 13, in the same
`compute()`, so a model trained on MuDst applies to pico input unchanged.
`testFeatures.C` asserts that equality rather than trusting it.

The workflow, then:

```sh
# train on MuDst (that is where the GEANT truth links are), with set 6
root4star -b -q 'trainTMVA.C("feat.root","FcsCat",6)'

# look at pico input with no model at all, first
root4star -b -q 'runPicoDst_ml.C("pi0.e30.vz0.run3.picoDst.root",-1,0)'

# then apply
root4star -b -q 'runPicoDst_ml.C("pi0...root",-1,1,"weights/FcsCat6_BDTG.weights.xml")'
```

Three limits of the pico path, all structural rather than fixable in this code:

- **The category cannot steer the photon fit.** On MuDst the whole point is that
  `StFcsMLCategoryMaker` runs *before* `StFcsPointMaker`, which then fits one or
  two photons accordingly. A picoDst is already reconstructed; nothing runs
  downstream. Here the category can only act as a selection on which clusters
  enter an analysis, so the π⁰ histograms compare *selections*, not refits. The
  real gain needs the MuDst chain.
- **No `StFcsPoint` collection in picoDst**, so the π⁰ QA pairs clusters.
- **`chi2Ndf1Photon` / `chi2Ndf2Photon` are only filled if `StFcsPointMaker` ran
  in the chain that produced the picoDst.** In the `pi0.e30.vz0.run3` sample
  they are identically zero for every cluster, so there is no fitter baseline to
  compare against on that file.

If you want the full 13 on pico input, the clean fix is upstream: add the tower
indices to `StPicoFcsCluster` and have `fillFcsClusters()` fill them, then
re-produce. That is a `star-sw` change plus a production pass — worth it if pico
becomes the main format for this analysis, not worth it for a study.

## 8. Not verified here

I wrote the makers against the STAR doxygen for `StFcsCluster`, `StFcsHit`,
`StFcsCollection` and `StFcsDb`, but nothing that touches `StRoot/` was compiled
— there is no STAR library stack in this session. Expect to fix an include path
or a signature on the first `cons`.

What *is* compiled and tested: `StFcsClusterFeatures.h` (builds clean under
`g++ -Wall`, and `testFeatures.C` passes all its invariant checks), and
`StFcsMLP.h` against the weight exporter.

For the picoDst path specifically: the set-6 feature computation was run through
`compute(6, ...)` on all 130 ECal clusters above 0.5 GeV in
`pi0.e30.vz0.run3.picoDst.root` and agrees with an independent calculation to
float precision, including the 8 clusters with `sigmaMax == 0` where
`sigmaRatio` must not divide by zero. The ROOT I/O around it —
`StPicoDstMaker`, the branch names, `StPicoDst::fcsCluster()` — is written from
the `star-sw` headers and has not been run; that needs a `cons` build.
