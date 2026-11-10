#!/bin/bash
# runTrainAll.sh - the whole chain over a set of single-particle files:
# dump the features of each, merge them, QA the merge, train, evaluate, compare.
#
#   ./runTrainAll.sh <tag> <file1.picoDst.root> [file2.picoDst.root ...]
#
# example, fixed-energy samples plus the flat one you already have:
#   ./runTrainAll.sh e2fix \
#       gamma.e{10,20,30,40,50,60,80}.vz0.all.picoDst.root \
#       pi0.e{10,20,30,40,50,60,80}.vz0.all.picoDst.root \
#       pi-.e{10,20,30,40,50,60,80}.vz0.all.picoDst.root \
#       gamma.e80.vz0.all.picoDst.root pi0.e80.vz0.all.picoDst.root pi-.e80.vz0.all.picoDst.root
#
# WHY THE FLAT SAMPLE STAYS IN THE LIST. Set 13 uses logE, and a tree cannot
# interpolate: trained only at 10, 20, ... 80 GeV it may learn energy-specific
# rules and behave oddly at 15 or 25 GeV, which is what data contains. Fixed
# energies are for STATISTICS where you need them - 40 GeV and up, where merged
# pi0 lives - and for clean evaluation points; the continuum is what keeps the
# model smooth. Mixing both is the intended use.
#
# Outputs, all prefixed with <tag>:
#   feat_pico_<tag>.root   the merged feature tree (hadd of the per-file dumps).
#                          The per-file dumps keep the input's own name,
#                          feat_<input>.picoDst.root, so both say where they
#                          came from - the merge of a short tag like "mix1"
#                          would not otherwise.
#   qa_<tag>.pdf/.log      QA of everything that goes into training
#   FcsCat<set>gen*.weights.xml  the trained models (labelDef=1)
#   eval<tag><rt>_{BDTG,MLP}.* efficiency, purity and the ePIC-style pages
#   cmp_<tag><rt>.pdf      BDTG vs MLP on one figure
#   ov<tag><rt>_*          the training-half evaluations, for the overtraining gap
#
# Set FEATURESET, EMIN, WEIGHTMODE, NORMMODE or LABELDEF in the environment to
# override:
#
#   env WEIGHTMODE=1 ./runTrainAll.sh ...
#
# (env, because tcsh does not take `VAR=value command`.)
#
# WEIGHTMODE DEFAULTS TO 0 - UNWEIGHTED. It used to be hard-coded to 1, the
# flat-in-cluster-energy weighting, and that is what produced the misleading
# high-energy hadronic numbers written up in BUILD.md: weighting asks for equal
# classes in energy bins where the hadronic class barely exists, and a weight
# cannot create clusters. Pass WEIGHTMODE=1 to reproduce the old behaviour.
#
# A non-default mode adds a RUN TAG to every output name - "_w1" for weighted,
# "_eq" for NormMode=EqualNumEvents - so a weighted and an unweighted run of the
# same sample do not overwrite each other's pdfs and logs. The weight files are
# already distinguished by trainTMVA.C's own job name (FcsCat13gen vs
# FcsCat13genw), so two modes can live side by side in weights/ too.
#
# Per-file dumps are kept, so re-running skips what is already there. The merge
# is kept as well: if feat_pico_<tag>.root is already there this script goes
# straight to the QA, so re-running it with a different WEIGHTMODE costs only
# the training and the evaluations.
#
# Because the merge is reused, a re-run only has to name the files whose DUMPS
# might still be missing - on a second pass over a sample already merged, the
# list can be just the newest files, or even one of them. What was merged is
# recorded in feat_pico_<tag>.inputs and checked: naming a file the merge does
# not contain stops the script rather than quietly training on the old sample.

set -e
tag="$1"
shift || true
if [ -z "$tag" ] || [ $# -eq 0 ]; then
   echo "usage: $0 <tag> <file1.picoDst.root> [file2 ...]"
   exit 1
fi

fset="${FEATURESET:-13}"
emin="${EMIN:-0.5}"
wgt="${WEIGHTMODE:-0}"
nrm="${NORMMODE:-0}"
# LABELDEF picks the truth definition, and the evaluation follows it so the
# model is always judged against the labels it was trained on:
#   1 (default) generated particle, cleaned - resolved pi0 photons dropped
#   2           sample-based, nothing dropped (StFcsSampleLabel.h) -> truthDef=3
#   0           mcLabel, the photons inside each cluster           -> truthDef=0
lbl="${LABELDEF:-1}"
case "$lbl" in
   0) tdef=0 ;;
   1) tdef=1 ;;
   2) tdef=3 ;;
   *) echo "LABELDEF must be 0, 1 or 2"; exit 1 ;;
esac
feat="feat_pico_${tag}.root"
# Earlier runs of this script wrote feat_<tag>.root. If that file is there and
# the new name is not, use it rather than spending another hadd on the same
# dumps; rename it if you would rather have the new convention.
if [ ! -f "$feat" ] && [ -f "feat_${tag}.root" ]; then
   echo "== using feat_${tag}.root from an earlier run (new runs write $feat)"
   feat="feat_${tag}.root"
fi
job="FcsCat"
# trainTMVA.C builds its job name as <job><set><gen><w><eq>; mirror it exactly.
wtag="${job}${fset}"
[ "$lbl" = "1" ] && wtag="${wtag}gen"   # labelDef=1 -> "gen"
[ "$lbl" = "2" ] && wtag="${wtag}smp"   # labelDef=2 -> "smp"
[ "$wgt" = "1" ] && wtag="${wtag}w"
[ "$nrm" = "1" ] && wtag="${wtag}eq"
# run tag on the OUTPUT names, so two modes on one sample do not collide
rt=""
[ "$lbl" = "0" ] && rt="${rt}_mc"
[ "$lbl" = "2" ] && rt="${rt}_smp"
[ "$wgt" = "1" ] && rt="${rt}_w1"
[ "$nrm" = "1" ] && rt="${rt}_eq"

# ---- 1. dump each file separately, then merge -------------------------------
dumps=""
for f in "$@"; do
   if [ ! -f "$f" ]; then
      echo "missing input: $f"
      exit 1
   fi
   d="feat_$(basename "$f" .root).root"
   if [ -f "$d" ]; then
      echo "== $d exists, skipping its dump"
   else
      echo "== dumping $f"
      root4star -b -q "runPicoDst_ml.C(\"$f\",-1,0,$fset,\"\",\"$d\")" 2>&1 | tee "dump_$(basename "$f" .root).log"
   fi
   dumps="$dumps $d"
done

# WHAT WENT INTO THE MERGE, recorded beside it. The merge is reused when it is
# already there, which is what makes a second pass cheap - but it also means the
# file list on the command line is NOT necessarily what gets trained on. Naming
# fewer files than the merge holds is fine and deliberate: pass just the new
# 80 GeV files to re-train on an existing 10-80 GeV merge. Naming a file the
# merge does NOT hold is a mistake worth stopping for, because the only reason
# to name it was to include it.
inputs="feat_pico_${tag}.inputs"
if [ -f "$feat" ]; then
   if [ -f "$inputs" ]; then
      missing=""
      for d in $dumps; do
         grep -qxF "$d" "$inputs" || missing="$missing $d"
      done
      if [ -n "$missing" ]; then
         echo "ERROR: $feat was merged WITHOUT:$missing"
         echo "       Training now would silently use the old sample."
         echo "       To rebuild the merge with them in it:"
         echo "          rm $feat $inputs"
         echo "          $0 $tag <the FULL file list, old files included>"
         exit 1
      fi
      echo "== $feat exists and holds every file named ($(wc -l < "$inputs") in all), not re-merging"
   else
      echo "== $feat exists, not re-merging (delete it to rebuild)"
      echo "   NOTE: it predates the input record, so what is actually in it is"
      echo "         not checked against the files named on the command line"
   fi
else
   echo "== merging into $feat"
   hadd -f "$feat" $dumps
   : > "$inputs"
   for d in $dumps; do echo "$d" >> "$inputs"; done
fi

# ---- 2. QA what is about to be trained on -----------------------------------
# The merged file has many gun energies, so the energy axis comes from the data.
echo "== QA of the merged sample"
root4star -b -q "qaFeatures.C+(\"$feat\",\"qa_${tag}\",$emin)" 2>&1 | tee "qa_${tag}.log"

# labelDef=2 ranks the clusters of an event and keeps only some of them, so QA
# what actually reaches each class before training on it.
if [ "$lbl" = "2" ]; then
   echo "== QA of the sample definition (what goes into each class)"
   root4star -b -q "qaSampleLabel.C+(\"$feat\",\"qsmp_${tag}\",$emin,0)" \
      2>&1 | tee "qsmp_${tag}.log"
fi

# ---- 3. train ---------------------------------------------------------------
echo "== training set $fset, labelDef=$lbl, weightMode=$wgt, normMode=$nrm -> $wtag"
root4star -b -q "trainTMVA.C+(\"$feat\",\"$job\",$fset,\"clusters\",0.8,$emin,$lbl,$wgt,$nrm)" \
   2>&1 | tee "train_${tag}${rt}.log"

# ---- 4. evaluate both methods, test half and training half ------------------
for m in BDTG MLP; do
   w="weights/${wtag}_${m}.weights.xml"
   if [ ! -f "$w" ]; then
      echo "no $w - training must have failed, see train_${tag}${rt}.log"
      continue
   fi
   echo "== evaluating $m (test half)"
   root4star -b -q "evalCategory.C+(\"$feat\",\"$w\",$fset,\"$m\",1,\"eval${tag}${rt}_${m}\",$emin,0.8,\"clusters\",$tdef,2)" \
      2>&1 | tee "eval${tag}${rt}_${m}.log"
   echo "== evaluating $m (training half, for the overtraining gap)"
   root4star -b -q "evalCategory.C+(\"$feat\",\"$w\",$fset,\"$m\",0,\"ov${tag}${rt}_${m}\",$emin,0.8,\"clusters\",$tdef,2)" \
      2>&1 | tee "ov${tag}${rt}_${m}.log"
done

# ---- 5. compare ------------------------------------------------------------
if [ -f "eval${tag}${rt}_BDTG.root" ] && [ -f "eval${tag}${rt}_MLP.root" ]; then
   echo "== BDTG vs MLP"
   root4star -b -q "compareModels.C+(\"eval${tag}${rt}_BDTG.root\",\"BDTG\",\"eval${tag}${rt}_MLP.root\",\"MLP\",\"\",\"\",\"\",\"\",\"cmp_${tag}${rt}\")" \
      2>&1 | tee "cmp_${tag}${rt}.log"
fi
for m in BDTG MLP; do
   if [ -f "ov${tag}${rt}_${m}.root" ] && [ -f "eval${tag}${rt}_${m}.root" ]; then
      echo "== overtraining gap, $m"
      root4star -b -q "compareModels.C+(\"ov${tag}${rt}_${m}.root\",\"training half\",\"eval${tag}${rt}_${m}.root\",\"test half\",\"\",\"\",\"\",\"\",\"ovCheck_${tag}${rt}_${m}\")" \
         2>&1 | tee "ovCheck_${tag}${rt}_${m}.log"
   fi
done

echo "== done. Read in this order:"
echo "   qa_${tag}.pdf            do the inputs look sane, and how correlated are they"
echo "   train_${tag}${rt}.log         the weight table and the class counts"
echo "   ovCheck_${tag}${rt}_*.log     train vs test: the overtraining gap"
echo "   cmp_${tag}${rt}.pdf           BDTG vs MLP, and the FCS Cluster reference"
echo "   eval${tag}${rt}_*.pdf         the ePIC-style pages and the per-energy-point table"
