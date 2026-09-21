# How to run this

Five steps, in order. Steps 1–3 happen once per training sample; step 4 is what
you repeat.

```
  0. build                cons, inside the SL7 container
  1. training sample      runFzd_ml.C      .fzd     -> feat_*.root
                          or runMudst_ml.C  mode 0   on a MuDst with MC arrays
                          or runPicoDst_ml.C mode 0  on a picoDst with MC arrays
  2. check the truth      three draw commands on feat_*.root
  3. train                trainTMVA.C+    feat   -> weights/*.xml
     evaluate             evalCategory.C+ feat + weights -> efficiency, purity, plots
  4. apply                runMudst_ml.C   (MuDst)  or  runPicoDst_ml.C (picoDst)
```

All three step-1 macros write the **same tree**, called `clusters`, with the same
branches, so step 3 does not care which produced it and `hadd` can merge them.

There are **two kinds of truth**, and they survive differently. That single fact
explains the shape of everything else:

- **Hit level** — which GEANT track deposited energy in which tower. Exists only
  in a GEANT chain: `StFcsHit::getGeantTracks()`, filled by
  `StFcsFastSimulatorMaker`. `StMuFcsHit` and `StPicoFcsHit` store detector id,
  id, adc and energy and nothing more, so this is **lost** in a MuDst or
  picoDst. It gives `trkPid`, `trkE`, `truthNPhoton`.
- **Generator level** — the particles that were generated, their momenta and
  vertices. This **does** survive: `StMuMcTrack` is constructed directly from
  `g2t_track_st` and keeps `GePid()`, `IdVx()`, `E()`, `Pxyz()`, with
  `StMuMcVertex::XyzV()`. It gives `mcLabel`, `mcSep`, `mcZgg` — and `mcLabel`
  is the label you should train on.

So you can build a labelled training sample from a `.fzd` **or** from a MuDst
that was produced with the MC arrays. The `.fzd` additionally gives the
hit-level branches; the MuDst is usually already sitting on disk from your
normal BFC pass.

### How truth level and detector level correspond

They are never matched up between files. Step 1 is a single job that reads the
generated particles **and reconstructs them**, so both levels are in memory in
the same event, and the correspondence is made there — per cluster — and frozen
into `feat.root`:

```
   pi0.e30.vz0.run1.fzd
        |
        |  fzin        St_geant_Maker
        v
   TRUTH LEVEL         g2t_track, g2t_vertex
                       the generated pi0 and its two photons
        |
        |  fcsSim      StFcsFastSimulatorMaker
        |              turns g2t_wca_hit into StFcsHit AND records, per hit,
        |              which GEANT track deposited the energy
        v
   DETECTOR LEVEL      StFcsHit -> StFcsCluster -> StFcsPoint
                       exactly the reconstruction that runs on real data
        |
        |  StFcsClusterFeatureMaker
        v
   feat.root           one row per RECONSTRUCTED cluster, carrying the truth
                       that belongs to that cluster
```

A row of `feat.root` is therefore a detector-level object (a reconstructed
cluster: its towers, widths, energy) with truth-level answers attached to it.
That pairing is what makes it trainable. Two independent mechanisms create it:

- **Hit level.** `StFcsHit::getGeantTracks()` — every tower hit knows which
  GEANT tracks put energy in it, because the fast simulator wrote that down.
  Cluster → its hits → the tracks that made them. No matching involved, it is a
  stored link. This gives `trkPid`, `trkE`, `truthNPhoton`.
- **Generator level.** The generated photons are projected from their start
  vertex onto the ECal plane and matched to the cluster centroid within
  `setMcMatchRadius` (11 cm). This is geometric matching, and it is independent
  of how GEANT shared the deposits out. This gives `mcLabel`, `mcSep`, `mcZgg`.

Step 4 is a **different purpose**: an already reconstructed MuDst or picoDst —
in general real data, where no truth exists — and the trained model supplies the
category that truth supplied during training.

```
   step 1   .fzd    -> [ truth + reconstruction ] -> feat.root -> model
   step 4   MuDst   -> [ reconstruction only    ] -> features  -> model -> category
```

The MuDst path works the same way, one level thinner. `StFcsMuMcTruthMaker`
reads the generated photons from `StMuMcTrack` and hands them to the dumper,
which projects and matches them exactly as above — so a MuDst produced from the
same `.fzd`, in your normal BFC pass, also yields `mcLabel`. What it cannot
give you is the hit-level set, since the hit-to-track links are gone.

One ordering trap on that path: `StChain` runs makers in the order they were
constructed, so `StFcsMuMcTruthMaker` must be built **before**
`StFcsClusterFeatureMaker` or the photons arrive an event late. It resolves the
dumper by name in `Init()` so it can be constructed first, and warns in `Init()`
if it finds itself running second.

The picoDst path works the same way again, one level thinner still.
`StPicoMcTrack` and `StPicoMcVertex` are written from the same `g2t_track` table,
so `StFcsPicoFeatureMaker` reads the generated photons there and projects and
matches them exactly as the other two do — `mcLabel` is filled from a picoDst
too. What picoDst additionally loses is the cluster's **tower list**:
`StPicoDstMaker::fillFcsClusters()` copies the cluster scalars and drops
`StMuFcsCluster::hits()`, and the `FcsHits` collection is written flat with no
back-pointer. That association is recovered geometrically — see
`StFcsTowerAssoc.h`, and the numbers in the table below.

| input | truth it carries | tower list | features possible | use it for |
|---|---|---|---|---|
| `.fzd` | hit level **and** generator level | stored | all sets | training, and any hit-level truth study |
| MuDst | generator level only (`StMuMcTrack`) | stored | all sets | **training** or applying |
| picoDst | generator level only (`StPicoMcTrack`) | **reconstructed** | all sets | **training** or applying |

A MuDst or picoDst produced without the MC arrays has no truth at all; the maker
says so once in `Finish()` and `mcLabel` stays at −1.

### The one reconstructed step: towers on picoDst

`StFcsCluster::x(), y()` are the energy-weighted centroid in **cell units**, and
a tower id maps to a cell through `StFcsDb::getRowNumber/getColumnNumber`, whose
centre sits at `(column − 0.5, row − 0.5)`. Both sides are therefore in the same
frame: give every tower to the nearest cluster centroid, then keep only the
`nTowers` the cluster says it has, highest energy first. That second step is what
makes it accurate — nearest-centroid alone gets the count right half the time.

Measured over all 167 ECal clusters of `pi0.e30.vz0.run6.picoDst.root`, by the
same C++ that runs in the maker:

| | |
|---|---|
| recovered `nTowers` exactly right | 92 % (97 % within ±1) |
| recovered energy − stored energy | median **0.0000 GeV** |
| recovered centroid − stored centroid | median 0.086 cells (≈0.5 cm), no bias on either axis |

The residual is entirely towers carrying a negligible share of the energy sitting
between two clusters. **Use the recovered towers for shapes, not for scalars**:
energy, `x`, `y`, `nTowers`, `sigmaMin/Max` and `theta` are stored exactly in the
picoDst and the maker reads them from there. The QA branches `nTowRec`, `eRec`,
`dxRec`, `dyRec` let you repeat this check on your own sample, and you should
before trusting sets 3/13/34 on picoDst.

---

## 0. Build

RCF interactive nodes run Alma 9; STAR code runs in the SL7 container:

```sh
ssh rcas####.rcf.bnl.gov
cd <your working dir>

singularity shell --shell /usr/bin/csh \
  -B /direct -B /star -B /afs -B /gpfs -B /sdcc/lustre02 \
  /cvmfs/star.sdcc.bnl.gov/containers/rhic_sl7.sif csh
```

Inside the container:

```csh
starver dev      # or whichever version you are pinned to
cons
```

Layout `cons` expects, next to your existing `StFcsPi0FinderForEcal`:

```
StRoot/StFcsClusterFeatureMaker/    dumps the training tree from .fzd / MuDst
StRoot/StFcsMLCategoryMaker/        applies the model in a MuDst/StEvent chain
  ├ StFcsClusterFeatures.h          THE definition of the input variables
  └ StFcsTowerAssoc.h               recovers cluster <-> tower on picoDst
StRoot/StFcsPicoFeatureMaker/       dumps the same training tree from picoDst
StRoot/StFcsPicoCategoryMaker/      applies the model to picoDst input
StRoot/StFcsMuMcTruthMaker/         generator-level truth on the MuDst path
```

The two picoDst packages include the two headers from `StFcsMLCategoryMaker`
through `cons`'s `-IStRoot`, so every package shares one definition of the
variables and one association, instead of each carrying a copy. Build one package
while iterating with `cons +StFcsPicoFeatureMaker`.

Two quick checks that need no STAR libraries at all, and no `cons`:

```csh
g++ -DSTANDALONE -std=c++0x -o testFeatures testFeatures.C && ./testFeatures
g++ -DSTANDALONE -std=c++0x -o testAssoc    testAssoc.C    && ./testAssoc
```

The first compiles the feature definitions and asserts their invariants; the
second does the same for the picoDst tower association, which is the one part of
the chain that reconstructs something rather than reading it. Run them after any
edit to `StFcsClusterFeatures.h` or `StFcsTowerAssoc.h`.

---

## 1. Training sample

Either input works. **From a simulated MuDst** you already have — no geometry
tag to get right, and it reuses your normal BFC output:

```csh
root4star -b -q 'runMudst_ml.C("pi0.e30.vz0.all.MuDst.root",-1,-2,".",1,0,0,"","feat_pi0.root")'
```

That is mode 0, and it now also runs `StFcsMuMcTruthMaker`, so `mcLabel` is
filled from `StMuMcTrack`. Check `MuDst->GetListOfBranches()` for
`StMuMcTrack`/`StMuMcVertex` first — a production without the MC arrays gives no
labels, and the maker will say so in `Finish()`.

**From the `.fzd`** if you also want the hit-level branches, or if the MuDst was
made without the MC arrays:

```csh
root4star -b -q 'runFzd_ml.C("pi0.e30.vz0.run1.fzd",-1,"feat_pi0.root","<geom>","<sdt>")'
```

`<geom>` and `<sdt>` have no defaults and the macro refuses to run without them.
Take them from the `.kumac` that produced the `.fzd` — a wrong geometry tag does
not crash, it mismaps GEANT volumes to tower ids, and you would train on
scrambled clusters with nothing to warn you.

The chain is `fzin` (St_geant_Maker, which builds the g2t tables) → `fcsSim`
(the fast simulator, which is what attaches GEANT tracks to hits) →
`fcsCluster` → `fcsPoint` → the feature dumper last, so the fitter's
`chi2Ndf1/2Photon` land in the same tree.

**From a picoDst** if that is what you have on disk — same tree out, no geometry
tag and no database:

```csh
root4star -b -q 'runPicoDst_ml.C("pi0.e30.vz0.all.picoDst.root",-1,0,3,"","feat_pico.root")'
```

That is mode 0. `StFcsDbMaker` runs with `setDbAccess(0)`: the only thing asked
of it is geometry, which it has built in, so there is no `St_db_Maker` and none
of the run-number-1 calibration trouble that bites the MuDst path. The tower list
is reconstructed here rather than read — check `nTowRec` against `nTowers` in the
output before training on sets 3/13/34.

Repeat per species and merge — the trees are identical, so `hadd` is enough:

```csh
hadd feat_all.root feat_gamma.root feat_pi0.root feat_pim.root
```

## 2. Check the truth actually arrived

Before training on anything:

```cpp
root -l feat_pi0.root
clusters->Draw("mcLabel")        // must not be all -1
clusters->Draw("trkPid[0]","e>1")   // 1 = gamma, GEANT3 pid
clusters->Draw("mcSepCell","mcLabel==2")  // the merge transition, in towers
```

All `-1` means the truth never arrived: `fcsSim` missing from the chain, or no
FCS hits in the `.fzd`. Labels come from `mcLabel` (generator level: how many
generated photons project onto the cluster) with `truthNPhoton` (hit level) as
fallback for older files.

## 3. Train

```csh
root4star -b -q 'trainTMVA.C+("feat_all.root","FcsCat",13)'
```

**The trailing `+` matters.** It makes ACLiC compile the macro instead of
letting CINT interpret it — CINT only approximates namespaces and inline
functions, and `StFcsClusterFeatures.h` is built from both, so interpreting it
risks training on features that differ from what the compiled makers compute.

Feature sets — the id names the set, not the variable count:

| id | variables | needs the tower list | on picoDst |
|---|---|---|---|
| 3 | 13 | yes | yes, with the recovered tower list |
| 6 | 6 | no | **yes, with nothing reconstructed** |
| 13 | 13 | yes, **plus `nNeighbor`** | yes, both recovered |
| 34 | 34 | yes | yes, with the recovered tower list |

Set 13 is the only one that uses `nNeighbor`, and picoDst does not store it. It
is recomputed from the recovered tower adjacency — see the next section — so set
13 trains on picoDst like the rest. A feature file dumped before that existed has
`nNeighbor` at −1 on every row, and TMVA kills the job outright on a variable
with no spread:

```
<FATAL> DataSetFactory : Variable nNeighbor is constant. Please remove the variable.
***> abort program execution
```

`trainTMVA.C` now catches that before TMVA does and says which variable and why.

### nNeighbor, and why it is not STAR's number exactly

`StFcsClusterMaker` calls two clusters neighbours when one hit is within 1.01
cells of a tower of each, and records it by pushing into
`StFcsCluster::mNeighbor` — **once per linking hit, with no de-duplication**. So
`StFcsCluster::nNeighbor()` counts linkings, not clusters, and its value depends
on the order hits were consumed in. That order cannot be reconstructed from a
picoDst.

The feature is therefore the number of **distinct** neighbouring clusters, on
both tiers: `StFcsClusterFeatureMaker` de-duplicates `clu->neighbor()`, and
`StFcsTowerAssoc::neighborCounts()` computes the same thing from the tower grid.
The raw STAR count is still in the tree, as `nNeighborRaw`, and is in no feature
set. On the sample that came with this code the distribution is 0: 74, 1: 84,
2: 7, 3: 2 — a real variable, not a placeholder.

One consequence worth knowing: STAR cross-links *every pair* of clusters that a
single hit touches, so three clusters in a row all count as mutual neighbours,
including the two at the ends that do not touch each other. `neighborCounts()`
reproduces that rather than "fixing" it.

Set 3 (the 3×3 tower set) is the recommended starting point. Set 6 is the one
that needs nothing reconstructed anywhere, so it is the fallback if you ever
doubt the picoDst association — train it alongside set 3 and compare the two
models on the same sample; if they disagree much on picoDst and not on MuDst, the
association is what to look at.

Output lands in `weights/FcsCat<set>_BDTG.weights.xml`. Train with the same ROOT
that `root4star` uses — TMVA weight XML is not reliably portable across ROOT
major versions.

The macro prints an account of where every cluster went. If it keeps nothing it
stops before TMVA and names the reason rather than aborting inside it.

**The train/test split is by event, and deterministic.** `trainTMVA.C` no longer
lets TMVA pick the test half at random: even-numbered events train, odd ones
test, with every cluster of an event on the same side (the two photon clusters of
one π⁰ are correlated, and splitting them would leak information). The rule is in
`StFcsTrainTestSplit.h`, together with the class-label rule, and `evalCategory.C`
uses the same header — so it can evaluate on exactly the clusters the model never
saw. A weight file trained *before* this change used TMVA's random split; re-train
before evaluating it.

## 3b. Efficiency, purity, and STAR's categorization beside it

TMVA's summary table ("best signal efficiency times signal purity") tunes a
separate set of cuts for each class and reports one product per class. It is fine
for comparing feature sets, but it is not what the makers deliver, and it hides
efficiency and purity inside one number. `evalCategory.C` gives each cluster the
class with the **highest score** — the rule `StFcsMLCategoryMaker` and
`StFcsPicoCategoryMaker` apply — and measures the result on the held-out half:

```csh
root4star -b -q 'evalCategory.C+("feat_pico_all.root","weights/FcsCat13_BDTG.weights.xml",13)' >& eval13.log
root4star -b -q 'evalCategory.C+("feat_pico_all.root","weights/FcsCat3_BDTG.weights.xml",3)'   >& eval3.log
```

Arguments after the feature set: method (`"BDTG"` or `"MLP"`), sample
(`1` test half — default, `0` training half, `2` everything), output name, `eMin`.
Running `sample=0` next to `sample=1` is the overtraining check: a large gap
between the two means the model has memorised its training clusters.

It prints, for the model and for `catStar` on the same clusters:

- the **confusion matrix** — rows are the true class, columns what was
  assigned, each row as a fraction of its true class;
- per class, **efficiency** (of the true class-k clusters, the fraction called k)
  and **purity** (of the clusters called k, the fraction truly k), with errors;
- a **balanced purity**, as if the three classes were equally common.

Efficiency does not depend on the class mix; purity does. In a single-particle
sample the mix is whatever number of γ, π⁰ and π⁻ events you simulated, so quote
purity only together with the mix — which the macro prints — or use the balanced
one.

`evalFcsCat<set>_<method>.pdf` has four pages: the confusion matrices; efficiency
and purity against cluster energy (model filled, STAR open markers); two-photon
efficiency against the separation of the two photons in towers — the merged-π⁰
transition, and the plot that shows most directly what the model adds; and the
model's three score distributions for each true class. Every histogram behind them
is in the matching `.root` file.

**Per generated particle.** Every feature file now carries `genPid`, `genE` and
`nGen`: the GEANT id and energy of the generated (gun) particle of the event — for
a single-particle sample, which sample the cluster came from. That is lost
otherwise once the γ, π⁰ and π⁻ files are `hadd`-ed together. `genPid` is **not**
the training label: the label stays `mcLabel`, the number of photons inside the
cluster, because a resolved π⁰ photon is physically the same object as a gun
photon and no feature can tell them apart. With `genPid` present,
`evalCategory.C` adds, per particle (γ / π⁰ / π⁻):

- what its clusters truly are, what the model calls them, and what the FCS Cluster
  category calls them — printed, and as page 5 of the PDF;
- every input feature of the chosen set drawn separately for each particle, on the
  pages after it, so the characteristics the model learns from are visible directly.

Feature files dumped before this change have no `genPid`; the macro says so and
skips those pages. Re-dump to get them — training is unaffected.

**STAR's category is not the same three classes.** `catStar` 0 means *ambiguous —
let `StFcsPointMaker` try both fits*, not hadron; STAR has no hadron class. So the
comparison is for one- and two-photon clusters only, and STAR's efficiencies count
only clusters it categorised outright. The fitter resolves its ambiguous ones
later, using a χ² that picoDst does not keep usefully, so STAR's numbers here are a
lower bound on what the full STAR chain achieves.

## 4. Apply

**On MuDst** — the only place the categorization can change the physics, because
the maker sits *before* `StFcsPointMaker` and so steers the 1γ/2γ fit:

```csh
# QA first: changes nothing, fills the STAR-vs-ML migration matrix
root4star -b -q 'runMudst_ml.C("<MuDst>",-1,-2,".",1,0,1,"weights/FcsCat13_BDTG.weights.xml")'
```

Set `mlcat->setMode(0)` for QA-only, `1` to take the model's answer, `2` to
override only above `setConfidence`.

**On picoDst** — evaluation only. Nothing runs downstream of a picoDst, so the
category can only select which clusters enter an analysis, never re-fit one:

```csh
# mode 0: dump features and the STAR category, no model - look first
root4star -b -q 'runPicoDst_ml.C("<picoDst or .list>",-1,0,3,"","feat_pico.root")'
# mode 1: apply. The feature set must match the weight file.
root4star -b -q 'runPicoDst_ml.C("<picoDst or .list>",-1,1,3,"weights/FcsCat3_BDTG.weights.xml")'
# mode 2: both, in one pass
```

## 5. Batch

Submit from the Alma 9 node, **outside** the container, and let the scheduler
enter it per job. In your `submitScheduler/*.xml`:

```xml
<shell>singularity exec -e -B /direct -B /star -B /afs -B /gpfs -B /sdcc/lustre02 /cvmfs/star.sdcc.bnl.gov/containers/rhic_sl7.sif</shell>
```

Submit nodes are `starsub01`–`starsub07`. Ship the weight XML with the job or
give it an absolute path the container mounts — a relative `weights/...`
resolves against the job's scratch directory, not your working directory.

---

## When it goes wrong

Every one of these actually happened while building this; the cause is rarely
what the message says.

| symptom | cause | fix |
|---|---|---|
| `dlopen error: StPicoDstMaker.so: undefined symbol: _ZN7StMuDst16mMuFmsCollectionE` | `StPicoDstMaker` links against `StMuDSTMaker` even when only reading | load `Load.C` and `StMuDSTMaker/COMMON/macros/loadSharedLibraries.C` first — `runPicoDst_ml.C` does |
| `dlopen error: ... undefined symbol: _ZN24StFcsClusterFeatureMaker7kMaxTrkE` | a `static const int` odr-used (`std::min` takes `const&`) with no out-of-class definition; `cons` links it anyway, it only fails at load | define it in the `.cxx`: `const int Class::kConst;` |
| `Error: Too many '}' tmpfile:NN` | CINT mishandling `#if`/`#else` inside a function body | no preprocessor branches in macro bodies; run with `trainTMVA.C+` |
| `<FATAL> DataInputHandler: Encountered empty TTree or TChain` | no labelled clusters — a file whose input had no MC arrays | check `clusters->Draw("mcLabel")`; the macro now diagnoses this before TMVA |
| `StFcsPicoFeatureMaker::Init failed to get StFcsDb` | no `StFcsDbMaker` in the pico chain | add one and call `setDbAccess(0)`; `runPicoDst_ml.C` does |
| every `mcLabel` is −1 from a picoDst | `McTrack`/`McVertex` switched off, or the picoDst was made without them | `picoMaker->SetStatus("McTrack*",1)` and `("McVertex*",1)` |
| `nTowRec` is 0 for every cluster | `FcsHits` switched off in `SetStatus` | `picoMaker->SetStatus("FcsHits*",1)` |
| `<FATAL> Variable <x> is constant. Please remove the variable.` + abort | an input variable the dumper could not fill, so it wrote a placeholder on every row | `trainTMVA.C` now names it and stops first; use a set that does not need it (only set 13 uses `nNeighbor`), or re-dump with a maker that fills it |
| `Error: Symbol <Maker> is not defined in current scope` in a run macro | `gSystem->Load` for it sits below the declaration; CINT parses the whole body first | load every library at the top of the macro |
| `cons` fails at `rootcint`, though the `.cxx` compiled | CINT cannot parse an implementation header pulled into a dictionary header | keep implementation headers in the `.cxx`; forward declare inside `#ifndef __CINT__` |
| clusters land in implausible towers | geometry tag does not match the simulation | use the tag from the `.kumac` that made the `.fzd` |

## What has and has not been tested

`StFcsClusterFeatures.h` and `StFcsTowerAssoc.h` compile clean under
`g++ -Wall -std=c++0x`, and `testFeatures.C` / `testAssoc.C` pass their invariant
checks — including the neighbour-counting rules, where the test caught the
cross-linking behaviour above being different from what I first assumed. The set-6 features were verified against all 130 ECal clusters above
0.5 GeV in a real picoDst, and the generator-level projection was checked against
stub types with a synthetic π⁰ → γγ event.

The picoDst tower association was run — as the compiled C++, not a
reimplementation — over all 167 ECal clusters of `pi0.e30.vz0.run6.picoDst.root`,
giving the numbers in the table above, and all four feature sets came out of it
with no failures and no NaNs. `StFcsPicoFeatureMaker.cxx` and
`StFcsPicoCategoryMaker.cxx` were type-checked against stub headers written from
the real `star-sw` class declarations, so the accessor names and signatures are
right.

Everything else that touches STAR classes — the chain macros, the g2t field
access, the behaviour of `StFcsDbMaker` with `setDbAccess(0)` in a pico chain —
is written from the `star-sw` sources and is only proven by your `cons` build and
first run.
