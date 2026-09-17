# How to run this

Five steps, in order. Steps 1–3 happen once per training sample; step 4 is what
you repeat.

```
  0. build                cons, inside the SL7 container
  1. training sample      runFzd_ml.C     .fzd   -> feat_*.root
  2. check the truth      three draw commands on feat_*.root
  3. train                trainTMVA.C+    feat   -> weights/*.xml
  4. apply                runMudst_ml.C   (MuDst)  or  runPicoDst_ml.C (picoDst)
```

The one rule that explains the shape of all of it: **train from the `.fzd`,
apply on MuDst or picoDst.** The labels come from GEANT, and GEANT truth does
not survive into a MuDst or a picoDst — `StMuFcsHit` and `StPicoFcsHit` store
detector id, id, adc and energy, and nothing about which track deposited the
energy. A feature file made from a MuDst has features and no labels.

| input | carries | use it for |
|---|---|---|
| `.fzd` | GEANT record: g2t tables, hits, generated particles | **training samples** |
| MuDst | reconstructed clusters + towers, no truth | applying, full feature sets |
| picoDst | cluster summary only, no tower list, no truth | applying, feature set 6 only |

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
StRoot/StFcsClusterFeatureMaker/    dumps the training tree (needs GEANT truth)
StRoot/StFcsMLCategoryMaker/        applies the model in a MuDst/StEvent chain
  └ StFcsClusterFeatures.h          THE definition of the input variables
StRoot/StFcsPicoCategoryMaker/      applies the model to picoDst input
```

`StFcsPicoCategoryMaker` includes `StFcsMLCategoryMaker/StFcsClusterFeatures.h`
through `cons`'s `-IStRoot`, so both packages share one definition of the
variables instead of each carrying a copy. Build one package while iterating
with `cons +StFcsMLCategoryMaker`.

Quick check that needs no STAR libraries at all, and no `cons`:

```csh
g++ -DSTANDALONE -std=c++0x -o testFeatures testFeatures.C && ./testFeatures
```

That compiles the feature definitions and asserts their invariants. Run it after
any edit to `StFcsClusterFeatures.h`.

---

## 1. Training sample, from the `.fzd`

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

| id | variables | needs the tower list | works on picoDst |
|---|---|---|---|
| 3 | 13 | yes | no |
| 6 | 6 | no | **yes** |
| 13 | 13 | yes | no |
| 34 | 34 | yes | no |

Set 3 (the 3×3 tower set) is the recommended starting point. Set 6 is the only
one that can be applied to picoDst input, so train that one too if pico is where
you will run.

Output lands in `weights/FcsCat<set>_BDTG.weights.xml`. Train with the same ROOT
that `root4star` uses — TMVA weight XML is not reliably portable across ROOT
major versions.

The macro prints an account of where every cluster went. If it keeps nothing it
stops before TMVA and names the reason rather than aborting inside it.

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
root4star -b -q 'runPicoDst_ml.C("<picoDst or .list>",-1,0)'   # no model, look first
root4star -b -q 'runPicoDst_ml.C("<picoDst or .list>",-1,1,"weights/FcsCat6_BDTG.weights.xml")'
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
| `<FATAL> DataInputHandler: Encountered empty TTree or TChain` | no labelled clusters — usually a feature file made from a MuDst | rebuild the sample with `runFzd_ml.C`; the macro now diagnoses this before TMVA |
| `cons` fails at `rootcint`, though the `.cxx` compiled | CINT cannot parse an implementation header pulled into a dictionary header | keep implementation headers in the `.cxx`; forward declare inside `#ifndef __CINT__` |
| clusters land in implausible towers | geometry tag does not match the simulation | use the tag from the `.kumac` that made the `.fzd` |

## What has and has not been tested

`StFcsClusterFeatures.h` compiles clean under `g++ -Wall -std=c++0x` and
`testFeatures.C` passes its invariant checks. The set-6 features were verified
against all 130 ECal clusters above 0.5 GeV in a real picoDst, and the
generator-level projection was checked against stub types with a synthetic
π⁰ → γγ event.

Everything that touches STAR classes — the makers, the chain macros, the g2t
field access — is written from the `star-sw` headers and is only proven by your
`cons` build and first run.
