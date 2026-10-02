#!/bin/bash
# runTrainAll.sh - the whole chain over a set of single-particle files:
# dump the features of each, merge them, QA the merge, train, evaluate, compare.
#
#   ./runTrainAll.sh <tag> <file1.picoDst.root> [file2.picoDst.root ...]
#
# example, fixed-energy samples plus the flat one you already have:
#   ./runTrainAll.sh e2fix \
#       gamma.e{10,20,30,40,50,60}.vz0.all.picoDst.root \
#       pi0.e{10,20,30,40,50,60}.vz0.all.picoDst.root \
#       pi-.e{10,20,30,40,50,60}.vz0.all.picoDst.root \
#       gamma.e60.vz0.all.picoDst.root pi0.e60.vz0.all.picoDst.root pi-.e60.vz0.all.picoDst.root
#
# WHY THE FLAT SAMPLE STAYS IN THE LIST. Set 13 uses logE, and a tree cannot
# interpolate: trained only at 10, 20, ... 60 GeV it may learn energy-specific
# rules and behave oddly at 15 or 25 GeV, which is what data contains. Fixed
# energies are for STATISTICS where you need them - 40 to 60 GeV, where merged
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
#   FcsCat13genw*.weights.xml  the trained models (labelDef=1, weightMode=1)
#   eval<tag>_{BDTG,MLP}.* efficiency, purity and the ePIC-style pages
#   cmp_<tag>.pdf          BDTG vs MLP on one figure
#   ov<tag>_*              the training-half evaluations, for the overtraining gap
#
# Set FEATURESET, or EMIN, in the environment to override.
# Per-file dumps are kept, so re-running skips what is already there.

set -e
tag="$1"
shift || true
if [ -z "$tag" ] || [ $# -eq 0 ]; then
   echo "usage: $0 <tag> <file1.picoDst.root> [file2 ...]"
   exit 1
fi

fset="${FEATURESET:-13}"
emin="${EMIN:-0.5}"
feat="feat_pico_${tag}.root"
# Earlier runs of this script wrote feat_<tag>.root. If that file is there and
# the new name is not, use it rather than spending another hadd on the same
# dumps; rename it if you would rather have the new convention.
if [ ! -f "$feat" ] && [ -f "feat_${tag}.root" ]; then
   echo "== using feat_${tag}.root from an earlier run (new runs write $feat)"
   feat="feat_${tag}.root"
fi
job="FcsCat"
wtag="${job}${fset}genw"   # labelDef=1 + weightMode=1 -> "gen" + "w"

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

if [ -f "$feat" ]; then
   echo "== $feat exists, not re-merging (delete it to rebuild)"
else
   echo "== merging into $feat"
   hadd -f "$feat" $dumps
fi

# ---- 2. QA what is about to be trained on -----------------------------------
# The merged file has many gun energies, so the energy axis comes from the data.
echo "== QA of the merged sample"
root4star -b -q "qaFeatures.C+(\"$feat\",\"qa_${tag}\",$emin)" 2>&1 | tee "qa_${tag}.log"

# ---- 3. train: generated-particle labels, flat-in-cluster-energy weights -----
echo "== training set $fset, labelDef=1, weightMode=1"
root4star -b -q "trainTMVA.C+(\"$feat\",\"$job\",$fset,\"clusters\",0.8,$emin,1,1)" 2>&1 | tee "train_${tag}.log"

# ---- 4. evaluate both methods, test half and training half ------------------
for m in BDTG MLP; do
   w="weights/${wtag}_${m}.weights.xml"
   if [ ! -f "$w" ]; then
      echo "no $w - training must have failed, see train_${tag}.log"
      continue
   fi
   echo "== evaluating $m (test half)"
   root4star -b -q "evalCategory.C+(\"$feat\",\"$w\",$fset,\"$m\",1,\"eval${tag}_${m}\",$emin,0.8,\"clusters\",1,1)" \
      2>&1 | tee "eval${tag}_${m}.log"
   echo "== evaluating $m (training half, for the overtraining gap)"
   root4star -b -q "evalCategory.C+(\"$feat\",\"$w\",$fset,\"$m\",0,\"ov${tag}_${m}\",$emin,0.8,\"clusters\",1,1)" \
      2>&1 | tee "ov${tag}_${m}.log"
done

# ---- 5. compare ------------------------------------------------------------
if [ -f "eval${tag}_BDTG.root" ] && [ -f "eval${tag}_MLP.root" ]; then
   echo "== BDTG vs MLP"
   root4star -b -q "compareModels.C+(\"eval${tag}_BDTG.root\",\"BDTG\",\"eval${tag}_MLP.root\",\"MLP\",\"\",\"\",\"\",\"\",\"cmp_${tag}\")" \
      2>&1 | tee "cmp_${tag}.log"
fi
for m in BDTG MLP; do
   if [ -f "ov${tag}_${m}.root" ] && [ -f "eval${tag}_${m}.root" ]; then
      echo "== overtraining gap, $m"
      root4star -b -q "compareModels.C+(\"ov${tag}_${m}.root\",\"training half\",\"eval${tag}_${m}.root\",\"test half\",\"\",\"\",\"\",\"\",\"ovCheck_${tag}_${m}\")" \
         2>&1 | tee "ovCheck_${tag}_${m}.log"
   fi
done

echo "== done. Read in this order:"
echo "   qa_${tag}.pdf            do the inputs look sane, and how correlated are they"
echo "   train_${tag}.log         the weight table and the class counts"
echo "   ovCheck_${tag}_*.log     train vs test: the overtraining gap"
echo "   cmp_${tag}.pdf           BDTG vs MLP, and the FCS Cluster reference"
echo "   eval${tag}_*.pdf         the ePIC-style pages and the per-energy-point table"
