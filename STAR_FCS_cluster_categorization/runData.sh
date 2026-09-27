#!/bin/bash
# runData.sh - apply a trained model to REAL DATA and look at what changed.
#
#   ./runData.sh <data.picoDst.root> <weights.xml> [featureSet] [method] [nevt]
#
#     featureSet  must match the weight file (default 13)
#     method      BDTG or MLP, must match the weight file (default BDTG)
#     nevt        events to read, -1 = all (default)
#
# example:
#   ./runData.sh st_physics_23045012.picoDst.root weights/FcsCat13genw_MLP.weights.xml 13 MLP
#
# Three steps, three outputs:
#   1. feat_<tag>.root / qa_<tag>.pdf   the features of the DATA clusters, and
#      the QA plots for them. Compare these with the QA of a simulated sample
#      (./runQA.sh on a gamma/pi0/pi- file): if a variable looks different in
#      data, the model was trained on something the detector does not produce,
#      and every number downstream is suspect. This is the step people skip and
#      then spend a month wondering why data and simulation disagree.
#      The true-class panels will be empty - data has no MC truth. That is
#      expected; "no truth" is the only populated class.
#   2. cat_<tag>.root                   the category maker output: per cluster
#      the model's three scores and its class, beside STAR's own category, plus
#      cluster-pair mass spectra built with each.
#   3. dataResult_<tag>.pdf             the comparison: the two mass spectra
#      overlaid, and the migration matrix saying which clusters changed class.
#
# WHAT TO LOOK AT, in order:
#   - the feature QA, data vs simulation (step 1);
#   - the migration matrix: how many clusters STAR calls one photon does the
#     model call merged pi0, and how many does it call hadronic;
#   - the pi0 peak in the two mass spectra. A better category should sharpen
#     the peak or raise it above the background, not merely move counts around.
# Then run your own pi0 finder with the ML category and compare the yield with
# the standard chain - this script is the quick look, not the analysis.
#
# Run it from the directory that holds StRoot/ and the macros, after cons.

set -e
in="$1"
wts="$2"
set13="${3:-13}"
method="${4:-BDTG}"
nevt="${5:--1}"

if [ -z "$in" ] || [ ! -f "$in" ]; then
   echo "usage: $0 <data.picoDst.root> <weights.xml> [featureSet] [method] [nevt]"
   exit 1
fi
if [ -z "$wts" ] || [ ! -f "$wts" ]; then
   echo "weight file '$wts' not found"
   exit 1
fi
case "$in" in
   *picoDst*) ;;
   *) echo "this script expects a picoDst; for a MuDst use runMudst_ml.C with isSim=0"; exit 1 ;;
esac

tag=$(basename "$in" .root)
feat="feat_${tag}.root"
cat="cat_${tag}.root"
qa="qa_${tag}"
res="dataResult_${tag}"

# ---- 1. features + QA of the data clusters ----
if [ -f "$feat" ]; then
   echo "== $feat exists, skipping the dump"
else
   echo "== dumping features from $in"
   root4star -b -q "runPicoDst_ml.C(\"$in\",$nevt,0,$set13,\"\",\"$feat\")" 2>&1 | tee "dump_${tag}.log"
fi
echo "== QA of the data features"
root4star -b -q "qaFeatures.C+(\"$feat\",\"$qa\")" 2>&1 | tee "${qa}.log"

# ---- 2. apply the model ----
echo "== applying $method from $wts"
root4star -b -q "runPicoDst_ml.C(\"$in\",$nevt,1,$set13,\"$wts\",\"$cat\",\"$method\")" 2>&1 | tee "apply_${tag}.log"

# ---- 3. what changed ----
echo "== comparing the two categories"
root4star -b -q "plotDataResult.C+(\"$cat\",\"$res\")" 2>&1 | tee "${res}.log"

echo "== done:"
echo "   ${qa}.pdf          feature QA of the data (compare with simulation)"
echo "   ${res}.pdf         mass spectra and category migration"
echo "   $cat   per-cluster scores and categories"
