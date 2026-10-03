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

## 2b. Feature QA for one file

Before training on a new sample, look at every variable the model will see:

```bash
./runQA.sh pi0.e30.vz0.all.MuDst.root        # or a .picoDst.root; [nevt] [eMin] optional
```

It dumps the features (`runMudst_ml.C` as simulation, or `runPicoDst_ml.C`,
chosen from the file name; skipped if `feat_<name>.root` already exists) and runs
`qaFeatures.C`, which computes sets 3 and 13 with the same `StFcsClusterFeatures.h`
the training uses. `qa_<name>.pdf` holds:

- page 1: cluster energy, generated energy, north/south centroid maps, nTowers,
  nNeighbor, the FCS Cluster category, the true class;
- page 2: σ_max vs E with STAR's two category boundaries drawn on;
- per set: every variable split by true class (unit area, all clusters dashed),
  then every variable against cluster energy.

The energy axes end at the gun energy taken from the file name — the `e60` in
`pi0.e60.vz0.all.picoDst.root` (and in the `feat_...` file made from it) gives a
0–60 GeV axis. With no such token the largest energy in the file is used, rounded
up. A fourth argument overrides it: `./runQA.sh file.root -1 0.5 40`.

Each set also gets a **linear correlation matrix**, printed and drawn: the cheap
way to see whether two inputs carry the same information without training
anything. The log flags the strongest pair when it exceeds 95 %. `sigX` against
`sigmaMax` near 100 % is the motivation for feature set 10 (section 3).

`qa_<name>.log` has the per-variable table — non-finite values, min/max/mean/rms,
the share at the single most common value, and the mean per true class. A variable
flagged `CONSTANT` is the one TMVA would abort on. On an existing feature file:
`root4star -b -q 'qaFeatures.C+("feat_pico_all.root","qa_all")'`.

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

### ePIC-style labels: single EM / hadronic / merged π⁰ by generated particle

The default labels count the photons inside each cluster (`mcLabel`). With the
last argument `labelDef=1`, `trainTMVA.C` labels by the **generated particle**
instead, as in the ePIC study (rule in `StFcsTrainTestSplit.h`):

| input category | cluster | trained as |
|---|---|---|
| γ input | γ event, the photon inside | 1 single EM |
| π⁰ 1-cluster input | π⁰ event, **both** photons inside | 2 merged π⁰ |
| π⁰ 2-cluster input | π⁰ event, **one** photon inside (resolved π⁰) | not trained |
| π⁻ input | any cluster of a π⁻ event | 0 hadronic |

Gun fragments with no photon inside are dropped from the γ and π⁰ samples. The
three scores keep their order — `r[0]` P(hadronic), `r[1]` P(single EM), `r[2]`
P(merged π⁰) — so the makers use the model unchanged. The job gets a `gen` suffix:

```csh
root4star -b -q 'trainTMVA.C+("feat_pico_all.root","FcsCat",13,"clusters",0.8,0.5,1)'
root4star -b -q 'evalCategory.C+("feat_pico_all.root","weights/FcsCat13gen_BDTG.weights.xml",13,"BDTG",1,"",0.5,0.8,"clusters",1,1)'
```

Evaluate it with `truthDef=1`, so that the evaluation truth is the training
label. Page 6 of the evaluation PDF is the ePIC score figure: P(Single EM),
P(Hadronic), P(Merged π⁰), each for γ / π⁰ 2-cluster / π⁰ 1-cluster / π⁻ input.
The π⁰ 2-cluster clusters are shown there even though they were not trained on;
a good model gives them a high P(Single EM). Needs `genPid`: re-dump older files.

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

`evalFcsCat<set>_<method>.pdf` pages: the confusion matrices; the **ePIC-style
figure** (below) for the model and for the FCS Cluster category; efficiency
and purity against cluster energy (model filled, STAR open markers); two-photon
efficiency against the separation of the two photons in towers — the merged-π⁰
transition, and the plot that shows most directly what the model adds; and the
model's three score distributions for each true class. Every histogram behind them
is in the matching `.root` file.

**The ePIC-style figure (pages 2 and 3).** Three panels — *Single EM*,
*Hadronic*, *Merged π⁰* — each with efficiency (red) and purity (blue) against
energy, the layout of the ePIC cluster-categorization study; page 2 is the model,
page 3 the FCS Cluster category (whose Hadronic panel is empty: it has no hadron
class). Single EM = class onePhoton, Hadronic = other, Merged π⁰ = twoPhoton. Two
extra arguments, after `treename`, set what "true" means and what the x axis is:

| `truthDef` | a cluster is truly Single EM / Hadronic / Merged π⁰ when … |
|---|---|
| `0` (default) | it holds 1 photon / no photon / 2 photons (`mcLabel`) — what the model is trained on |
| `1` | γ-gun cluster holding the photon / any π⁻ cluster / π⁰-gun cluster holding both photons; gun fragments and resolved π⁰ photons are dropped |
| `2` | it comes from a γ / π⁻ / π⁰ event, whatever it contains — pure sample identity, as in a single-particle study; resolved π⁰s then count against Merged π⁰ |

`energyAxis = 1` plots against the generated (gun) energy instead of the cluster
energy. Both need `genPid`/`genE`. The closest match to the ePIC figure:

```csh
root4star -b -q 'evalCategory.C+("feat_pico_all.root","weights/FcsCat13_BDTG.weights.xml",13,"BDTG",1,"",0.5,0.8,"clusters",1,1)'
```

`truthDef` applies to every page, confusion matrices included. The energy axes
follow the sample, as in `qaFeatures.C`: the `e60` in `feat_pi0.e60...root` gives
0–60 GeV, otherwise the largest energy in the file, rounded up; a last argument
`eAxisMax` overrides it. Bins are 2 GeV wide up to 32 GeV and 4 GeV above. Only
the axes change — the confusion matrices and the printed numbers always use every
cluster.

**Per generated particle.** Every feature file now carries `genPid`, `genE` and
`nGen`: the GEANT id and energy of the generated (gun) particle of the event — for
a single-particle sample, which sample the cluster came from. That is lost
otherwise once the γ, π⁰ and π⁻ files are `hadd`-ed together. `genPid` is **not**
the training label: the label stays `mcLabel`, the number of photons inside the
cluster, because a resolved π⁰ photon is physically the same object as a gun
photon and no feature can tell them apart. With `genPid` present,
`evalCategory.C` adds, per particle (γ / π⁰ / π⁻):

- what its clusters truly are, what the model calls them, and what the FCS Cluster
  category calls them — printed, and as page 8 of the PDF;
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

### Weighted training: flat in cluster energy

`trainTMVA.C`'s last argument, `weightMode=1`, weights each cluster by
1/N(class, cluster-energy bin), so every class has a flat energy spectrum:

```csh
root4star -b -q 'trainTMVA.C+("feat_pico_all.root","FcsCat",13,"clusters",0.8,0.5,1,1)'
#  -> weights/FcsCat13genw_BDTG.weights.xml  (the "w" marks the weighted job)
```

Why: a 60 GeV π⁻ leaves several 1–3 GeV clusters while a γ leaves one cluster
with nearly all its energy, so the low end of the training sample is almost pure
hadron and the high end almost pure photon. Both BDTG and MLP learn that mix as a
prior — single-EM efficiency collapses below 5 GeV, hadron efficiency above
35 GeV. The weights remove the prior; the shower shape then decides.

Thin bins are merged (fewer than 50 clusters), weights are capped at 10× the class
median, and the job prints a per-class table with the weight range and how many
bins were merged or capped. Check that table: a class with a huge weight range is
a class whose energy spectrum barely overlaps the others.

What it cannot do: make the classes equal bin by bin. Each class gets a flat
spectrum and TMVA then equalises class totals, so a class present in fewer energy
bins keeps a larger share where it does live. Where a class is absent entirely —
merged π⁰ below ~10 GeV — nothing is done, and nothing should be: that is
kinematics.

### Set 4: does position help, or does it leak?

Review comment: *"In Set 3, would it be better to add position information, like
pseudorapidity (η) or azimuthal angle (φ)?"*

**Set 4 = set 3 + `x`, `y`** — 15 variables, the first 13 bit-identical to set 3
in value and in order, then the cluster centroid in column and row units. It is
in `StFcsClusterFeatures.h` like every other set, so no feature file has to be
re-dumped: the position is already in the tree.

Why x/y and not η/φ. The ECal is a flat wall ~7.1 m downstream, so for a cluster
in one half, η and φ are smooth monotone functions of the column and row (plus
the vertex z and which half it is in). A boosted tree cuts one variable at a time
and is invariant under any monotone transform of a single input, so `x`, `y` and
the (η, φ) pair derived from them carry the same information to a BDTG; for the
MLP the difference is a reparametrisation that the hidden layer can absorb.
Column and row are also what the detector actually gives us, with no vertex
assumption baked in. If a reviewer wants η/φ on the axis labels, that is a
transform at plot time, not a different input.

The physics that motivates it is real: incidence angle grows with radius and
stretches a shower along the radial direction — an elongation that looks like a
second photon but is not — and a cluster near a plate edge loses part of its
tail, so the same shape means something slightly different at different places.

#### The leakage problem, and the check for it

In a single-particle sample **each gun illuminates its own patch of the
detector**. If the γ, π⁰ and π⁻ runs point even slightly differently, a model
given `x` and `y` can read the class off the position and score beautifully on
simulation while learning nothing about showers. On data, where occupancy is set
by physics and by dead towers, that model collapses.

`qaFeatures.C` now prints the check, on the page immediately after the cluster
overview: the centroid map of each gun species side by side, and above them a
table

```
  cluster centroid per gun particle - the three species must illuminate the
  same region, or a model given x and y (feature set 4) can read the class
  off the position instead of the shower shape
  gun        clusters       <x>     rms x       <y>     rms y
  gamma         41832      11.73      4.86     17.42      7.95
  ...
```

Means agreeing to ≲0.1 cell and visually identical maps ⇒ the position is a
legitimate input. Means that differ ⇒ treat any set-4 gain as the leak until
proven otherwise. The table only appears when the file has a `genPid` branch
(i.e. a simulation dump); on data the page is skipped.

#### Running it

```csh
# QA first - look at the per-gun centroid page before training anything
root4star -b -q 'qaFeatures.C+("feat_pico_mix1.root","qa_mix1")'

# train set 4 and set 3 on the same file, same labels, same weights
root4star -b -q 'trainTMVA.C+("feat_pico_mix1.root","FcsCat",4,"clusters",0.8,0.5,1,1)'
root4star -b -q 'trainTMVA.C+("feat_pico_mix1.root","FcsCat",3,"clusters",0.8,0.5,1,1)'

# evaluate both on the held-out half (sample=1)
root4star -b -q 'evalCategory.C+("feat_pico_mix1.root","weights/FcsCat4genw_BDTG.weights.xml",4,"BDTG",1,"ev4",0.5,0.8,"clusters",1,1)'
root4star -b -q 'evalCategory.C+("feat_pico_mix1.root","weights/FcsCat3genw_BDTG.weights.xml",3,"BDTG",1,"ev3",0.5,0.8,"clusters",1,1)'

root4star -b -q 'compareModels.C+("ev3.root","set 3","ev4.root","set 4")'
```

#### Reading the outcome

| what you see | what it means | what to do |
|---|---|---|
| maps differ, set 4 gains | leakage, most likely | reject; use the derived angle variable instead |
| maps agree, set 4 gains on merged π⁰ | the incidence-angle effect is real | keep set 4, then add the detector id / radial distance |
| maps agree, gain ≲1 point everywhere | position adds nothing at this granularity | stay with set 3 |
| set 4 loses | 2 extra inputs diluting 13 good ones | stay with set 3 |

Also check the TMVA variable ranking in the training log: `x` and `y` near the
top is the signature of leakage, not of good physics.

#### The measured answer: position adds nothing

Run, BDTG, `labelDef=1 weightMode=1`, same merged file, same held-out half:

| | Single EM | Hadronic | Merged π⁰ |
|---|---|---|---|
| **efficiency** set 3 | 0.830 ± 0.003 | 0.897 ± 0.002 | 0.637 ± 0.004 |
| **efficiency** set 4 | 0.828 ± 0.003 | 0.900 ± 0.002 | 0.633 ± 0.004 |
| **purity** set 3 | 0.767 ± 0.003 | 0.812 ± 0.003 | 0.821 ± 0.003 |
| **purity** set 4 | 0.765 ± 0.003 | 0.812 ± 0.003 | 0.820 ± 0.003 |

Differences of 0.2–0.4 % with no consistent sign — set 4 is nominally behind on
two classes and ahead on one. Merged-π⁰ efficiency per photon separation is flat
between the two as well, including the 1.0–2.0 tower bins:

| separation [towers] | set 3 | set 4 | clusters |
|---|---|---|---|
| 0.50 – 1.00 | 0.855 ± 0.004 | 0.850 ± 0.004 | 7805 |
| 1.00 – 1.50 | 0.566 ± 0.007 | 0.560 ± 0.007 | 5448 |
| 1.50 – 2.00 | 0.260 ± 0.008 | 0.257 ± 0.008 | 2926 |

Those bins are where the incidence-angle argument should have bitten hardest: a
separation of one to two towers is the same size as the radial stretch position
would explain away. Nothing moves there, so **position carries no information
the 3×3 tower pattern does not already carry, and set 3 stands.**

The null result also settles the leakage worry in passing: a model exploiting
the gun patches would have *gained*, not tied. Whatever the centroid maps look
like, the BDT is not reading the class off `x` and `y`.

**Answer to the review comment:** position information was added as set 4 and
tested; it changes efficiency and purity by less than half a percent in every
class and does not help the merged-π⁰ bins where it should. Set 3 is kept. Set 4
stays in `StFcsClusterFeatures.h` so the test is reproducible.

One caveat on the comparison itself: the two models were evaluated on the *same*
clusters, so the ± above are the errors on each absolute number, not on their
difference — the difference is better determined than they suggest. That makes
the null stronger, not weaker: a real effect of the quoted size would have shown
up cleanly.

#### The less leak-prone alternative

The quantity the physics argument actually wants is not the position but **the
angle between the shower major axis (`theta`) and the radial direction at the
cluster**. That one variable separates "stretched because it arrived at an angle"
from "stretched because there are two photons", and it cannot encode which gun
fired because every species sees the same geometry. If set 4 helps but the
centroid maps are not identical, that is the variable to add.

Known limitation of set 4 as written: `x` and `y` are column and row *within one
ECal half*, and `ClusterInput` carries no detector id, so the model cannot tell
north from south. The radial distance from the beam axis — the physically
meaningful coordinate — needs the detector id and the half's offset. That is the
refinement to make if set 4 earns its place.

### Set 10: is the detector frame worth its correlation?

Set 13 carries both `sigmaMax`/`sigmaMin`/`theta` and `sigX`/`sigY`/`sigXY`. Those
are the eigenvalues-and-angle and the components of **one** covariance matrix:

```
sigX²  = σmax²·cos²θ + σmin²·sin²θ
sigY²  = σmax²·sin²θ + σmin²·cos²θ
sigXY  = (σmax² − σmin²)·sinθ·cosθ
```

so six numbers describe three degrees of freedom and the extra three add no
information. What they can still buy is a frame: a boosted tree cuts one variable
at a time and cannot rotate, so "wide along the row direction" is one cut on
`sigY` but a joint condition on `sigmaMax` and `theta`. The cost is real too —
correlated inputs mislead the TMVA ranking and gave the MLP its singular Hessian.

**Set 10** is set 13 with those three removed, everything else unchanged and in
the same order. Train both on the same file and let the data decide:

```csh
root4star -b -q 'trainTMVA.C+("feat_pico_all.root","FcsCat",10,"clusters",0.8,0.5,1,1)'
root4star -b -q 'evalCategory.C+("feat_pico_all.root","weights/FcsCat10genw_BDTG.weights.xml",10,"BDTG",1,"",0.5,0.8,"clusters",1,1)'
root4star -b -q 'compareModels.C+("evalFcsCat13genw_BDTG.root","set 13","evalFcsCat10genw_BDTG.root","set 10")'
```

Reading the outcome: a tie means drop the three and keep the smaller model; BDTG
losing while MLP does not means the frame argument is real and set 13 stays for
BDTG; both losing means the moments are doing something we have not understood.
#### What the measured correlations say

Measured on the γ + π⁰ + π⁻ sample (TMVA's own matrix, per class — other /
onePhoton / twoPhoton):

| pair | correlation | |
|---|---|---|
| `sigX` – `sigmaMax` | 0.48 / 0.42 / 0.60 | moderate |
| `sigY` – `sigmaMax` | 0.53 / 0.44 / 0.65 | moderate |
| `sigXY` – `theta` | 0.28 / 0.34 / 0.48 | weak |
| `e1e2Asym` – `e2Frac` | −0.89 / −0.98 / −0.94 | **redundant** |
| `e1e2Asym` – `seedFrac` | 0.80 / 0.96 / 0.90 | **redundant** |
| `sigmaRatio` – `sigmaMin` | 0.92 / 0.94 / 0.86 | **near-redundant** |
| `logE` – `nTowers` | 0.81 / 0.83 / 0.82 | physical |

**The detector-frame moments are not duplicates after all.** The algebra above
assumes both descriptions are built the same way. They are not. Four differences,
in order of importance:

1. **Weighting.** `StFcsClusterMaker::clusterMomentAnalysis()` weights each tower
   by `w = log(E + 1 − 0.1 GeV)`; `sigX`/`sigY`/`sigXY` weight by E itself. Log
   weights follow the shower tail, energy weights follow the core. This is the
   dominant difference and the reason the two are only 0.4–0.6 correlated.
2. **Threshold.** STAR drops towers with `w ≤ 0`, i.e. below 0.1 GeV; we keep
   every tower in the list we are given.
3. **Tower list.** On picoDst the list is not stored, so `StFcsTowerAssoc`
   rebuilds it — 92 % exact tower count, median energy difference zero, but not
   the original. STAR's `sigmaMax`/`sigmaMin` come from the original list, before
   picoDst threw it away. On MuDst our list is exact and this difference vanishes.
4. **Window.** The tree stores an 11×11 tower image around the seed, so a tower
   further than 5 cells from the seed is not in our sum. Rare for real clusters,
   but not impossible for a large hadronic one.

Positions are *not* a difference: `getLocalXYinCell` returns `col − 0.5`,
`row − 0.5` for the ECal — tower centres in cell units, the same geometry we use,
and a constant offset does not change a variance.

So the two are different estimators of the same shape, and set 13 carries more
information than the algebra suggested — consistent with set 13 beating set 3 on
merged π⁰ by 17 points. Point 3 also means the correlation is sample-dependent:
measure it again on a MuDst dump and expect it to rise.

**Set 10 is therefore a control, not an obvious cleanup.** Still worth one
training run: it measures what the energy-weighted moments add once the
log-weighted ones are present. Expect a small loss, not a tie.

**The real duplicates are elsewhere.** `e1e2Asym = (e1−e2)/(e1+e2)` is an exact
function of `seedFrac = e1/E` and `e2Frac = e2/E`, and the matrix shows it at 0.96
and −0.98 for one-photon clusters. `sigmaRatio = sigmaMin/sigmaMax` tracks
`sigmaMin` at 0.94. If the goal is a smaller, better-ranked set, those two are the
ones to drop — a variant worth adding if the set-10 test comes out as expected.

Feature files need no change: the sets are computed from the same stored tree.

### Comparing models

`compareModels.C` reads the `.root` files `evalCategory.C` already wrote and puts
two to four of them on one figure, with the FCS Cluster category from the first
file as a reference:

```csh
root4star -b -q 'compareModels.C+("evalFcsCat13_BDTG.root","BDTG set 13","evalFcsCat13_MLP.root","MLP set 13")'
```

It prints efficiency and purity per class summed over all bins, and the merged-π⁰
efficiency bin by bin in photon separation with the cluster count per bin, then
writes a three-page PDF. The inputs must be evaluations of the same feature file
with the same `truthDef`; nothing checks that.

### Fixed-energy samples, and how to mix them

Single-particle guns at a few fixed energies (10, 20, 30, 40, 50, 60 GeV) are the
efficient way to buy statistics exactly where they are missing — merged π⁰ above
40 GeV, where the photons are less than a tower apart. But they should **not** be
the whole training sample:

- Set 13 uses `logE`, and a boosted tree cannot interpolate. Trained only at six
  energies, it can learn energy-specific rules and behave oddly at 15 or 25 GeV —
  which is what data contains.
- Guns at ≥ 10 GeV produce almost no genuine low-energy photons, and low cluster
  energy is where the model is already weakest.

So keep a **continuous (flat) sample as the backbone** and add fixed-energy sets
on top, weighted toward the high end; add 2 and 5 GeV points if the low-energy
region matters. Keep `weightMode=1` — fixed-energy spikes make the spectrum very
non-uniform, which is exactly what the flat-in-cluster-energy weights correct.

`runTrainAll.sh` runs the whole chain over any list of files:

```bash
./runTrainAll.sh mix1 gamma.e{10,20,30,40,50,60}.vz0.all.picoDst.root \
                      pi0.e{10,20,30,40,50,60}.vz0.all.picoDst.root \
                      pi-.e{10,20,30,40,50,60}.vz0.all.picoDst.root \
                      gamma.e60.vz0.all.picoDst.root pi0.e60.vz0.all.picoDst.root
```

It dumps each file (skipping any already dumped), `hadd`s them into
`feat_pico_<tag>.root`, runs the QA, trains with `labelDef=1 weightMode=1`, evaluates
BDTG and MLP on both halves, and writes the BDTG-vs-MLP and train-vs-test
comparisons. `FEATURESET=10 ./runTrainAll.sh ...` switches the feature set.

`evalCategory.C` prints an extra table for such a sample: efficiency and purity
per class **per energy band**, with the FCS Cluster category on the line beneath.
The bands are found from the gaps in the generated energy — consecutive populated
0.5 GeV slots are one band, a hole wider than 1.5 GeV starts the next — so a
"40 GeV sample" generated as 38–42 GeV is recognised as one point, not as dozens
of distinct energies. A continuous spectrum has no gaps, collapses to a single
band, and is reported rather than tabulated. Evaluate a points-only merge (no
flat sample in it) to get the table.

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

### 4b. On real data, end to end

```bash
./runData.sh st_physics_23045012.picoDst.root weights/FcsCat13genw_MLP.weights.xml 13 MLP
```

Three steps: dump the features of the data clusters and QA them, apply the model,
then draw what changed.

- **`qa_<tag>.pdf`** — the data features. Compare with `./runQA.sh` on a simulated
  sample before believing anything else: a variable that looks different in data
  means the model was trained on something the detector does not produce. The
  true-class panels are empty on data, as expected.
- **`dataResult_<tag>.pdf`** (from `plotDataResult.C`) — the cluster-pair mass with
  STAR's category and with the model's, overlaid, and the migration matrix. The
  entries to look at are STAR "1 onePhoton" → model "merged π⁰" (π⁰s the standard
  chain would fit as one photon) and → "hadronic" (contamination removed).
- **`cat_<tag>.root`** — per cluster, the three scores, the model's class and
  STAR's.

The pair mass here is a diagnostic built from cluster pairs, not from fitted
points. For the physics number, run your own π⁰ finder with the ML category and
compare the yield and peak with the standard chain.

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

Feature set 4 (set 3 + position) is covered by `testFeatures.C`, which checks
that it reports 15 variables, that its first 13 are identical to set 3 in value
and in name, that `x`/`y` equal the centroid, and that it refuses a cluster with
no tower list. The per-gun centroid page added to `qaFeatures.C` syntax-checks
against the ROOT stub headers but has not yet been run on a file — the table and
the three maps come out of the first `qaFeatures.C` run on a mixed-species dump.

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
