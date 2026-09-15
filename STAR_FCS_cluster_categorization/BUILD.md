# Building and running at RCF (Alma 9 + SL7 container)

Everything here is C++/ROOT. There is nothing to install: TMVA ships inside
ROOT, ROOT ships inside the STAR library stack, and the SL7 container image is
read-only anyway. No pip, no conda, no external ML runtime.

## 1. Put the code in your working directory

Next to your existing `StRoot/StFcsPi0FinderForEcal`:

```
<your working dir>/
  StRoot/StFcsPi0FinderForEcal/       (already there)
  StRoot/StFcsClusterFeatureMaker/    (new)
  StRoot/StFcsMLCategoryMaker/        (new)
  StRoot/StFcsPicoCategoryMaker/      (new, picoDst input)
  runMudst_ml.C
  runPicoDst_ml.C
  trainTMVA.C
  testFeatures.C
```

`StFcsPicoCategoryMaker` includes `StFcsMLCategoryMaker/StFcsClusterFeatures.h`
by that relative path, which `cons` resolves through its `-IStRoot` flag — the
two packages share one definition of the input variables rather than each
carrying a copy.

`StFcsClusterFeatures.h` lives inside `StRoot/StFcsMLCategoryMaker/` and is
included both by the maker and by `trainTMVA.C` — that is deliberate, it is the
one definition of the input variables. `trainTMVA.C` includes it by the relative
path `StRoot/StFcsMLCategoryMaker/StFcsClusterFeatures.h`, so run that macro from
the top of your working directory.

## 2. Get an SL7 shell on an Alma 9 node

RCF interactive nodes run Alma 9; STAR code runs in the SL7 container:

```sh
ssh rcas####.rcf.bnl.gov
cd <your working dir>

singularity shell --shell /usr/bin/csh \
  -B /direct -B /star -B /afs -B /gpfs -B /sdcc/lustre02 \
  /cvmfs/star.sdcc.bnl.gov/containers/rhic_sl7.sif csh
```

`singularity exec ... csh` works too if you just want to run one command. If your
login files still assume AFS, the Alma 9 migration wants:

```csh
setenv USE_NFS4 1
setenv GROUP_DIR /star/nfs4/AFS/star/group
```

## 3. Set the STAR version and build

Inside the container:

```csh
starver dev          # or whichever version your analysis is pinned to
cons
```

`cons` walks `StRoot/`, builds one shared library per subdirectory, and drops
them in `.$STAR_HOST_SYS/lib`. Build one package while iterating:

```csh
cons +StFcsMLCategoryMaker
```

Two things that bite on the first build:

- The directory name, the class name and the `ClassDef`/`ClassImp` name must all
  agree, or `cons` builds a library that will not load.
- `libTMVA` has to be loaded before `StFcsMLCategoryMaker`, which
  `runMudst_ml.C` already does. If you load the maker's library some other way
  and get unresolved TMVA symbols, that is the reason.

## 4. Check the feature definitions compile and behave

Cheap and instant, no STAR libraries needed:

```csh
root -b -q testFeatures.C
```

or outside ROOT entirely:

```sh
g++ -DSTANDALONE -o testFeatures testFeatures.C && ./testFeatures
```

It builds synthetic 1-photon and 2-photon tower patterns and asserts the
invariants (seed fraction higher for one photon, radius larger for two, 5x5
fractions plus `eOut` summing to 1, and so on). Run it after any edit to
`StFcsClusterFeatures.h`.

## 5. The three steps

```csh
# 1. dump features from simulation (mode 0)
root4star -b -q 'runMudst_ml.C("<sim MuDst>",-1,-2,".",1,0,0,"","feat.root")'

# 2. train - inside the container, same ROOT as the chain
root4star -b -q 'trainTMVA.C("feat.root","FcsCat",13)'

# 3. apply (mode 1), starting in QA-only mode
root4star -b -q 'runMudst_ml.C("<MuDst>",-1,-2,".",1,0,1,"weights/FcsCat13_BDTG.weights.xml")'
```

For picoDst input, train with feature set 6 and use the other chain:

```csh
root4star -b -q 'trainTMVA.C("feat.root","FcsCat",6)'
root4star -b -q 'runPicoDst_ml.C("<picoDst or .list>",-1,0)'   # no model, look first
root4star -b -q 'runPicoDst_ml.C("<picoDst or .list>",-1,1,"weights/FcsCat6_BDTG.weights.xml")'
```

Set 6 is not a preference there — picoDst does not store a cluster's tower list,
so the other seven variables do not exist on that input. `PORTING.md` section 7
has the details and the limits that come with it.

Step 2 must run under `root4star` in the container, not under some other ROOT.
TMVA weight XML is not guaranteed to be readable across ROOT major versions, and
a weight file trained elsewhere that silently fails to book is the most common
way this goes wrong.

## 6. Batch jobs

Submit from the Alma 9 node, **outside** the container, and let the scheduler
enter it per job. In your `submitScheduler/*.xml`, add:

```xml
<shell>singularity exec -e -B /direct -B /star -B /afs -B /gpfs -B /sdcc/lustre02 /cvmfs/star.sdcc.bnl.gov/containers/rhic_sl7.sif</shell>
```

Submit nodes on Alma 9 are `starsub01`–`starsub07`. Ship the weight XML with the
job (it is a few hundred kB) or reference it by an absolute path that the
container mount list covers — a relative `weights/...` path resolves against the
job's scratch directory, not your working directory.

## Optional: Python

`optional_python/` holds a numpy/uproot reader and a PyTorch/sklearn weight
exporter for the non-TMVA backend. You do not need either for the TMVA workflow —
they are there only if you ever want to look at the dumped tree outside ROOT.
Those would need `pip install uproot numpy` in a user-space environment on the
Alma 9 side, outside the container.
