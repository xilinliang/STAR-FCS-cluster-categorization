#!/bin/bash
# runQA.sh - feature QA for one picoDst or MuDst file, in one command:
#   1. dump the cluster features  (runPicoDst_ml.C or runMudst_ml.C, mode 0)
#   2. plot every variable of feature sets 3 and 13  (qaFeatures.C)
#
# usage:  ./runQA.sh <file.picoDst.root | file.MuDst.root> [nevt] [eMin] [eMax]
#   nevt  events to read, -1 = all (default)
#   eMin  minimum cluster energy in GeV (default 0.5, as in training)
#   eMax  upper end of the energy axes in GeV; 0 (default) reads the gun energy
#         from the file name - pi0.e60.vz0.all.picoDst.root gives 60 GeV - and
#         falls back to the largest energy in the file when the name has none
#
# example:
#   ./runQA.sh pi0.e30.vz0.all.MuDst.root
#     -> feat_pi0.e30.vz0.all.MuDst.root   the feature tree
#        qa_pi0.e30.vz0.all.MuDst.pdf      the plots
#        qa_pi0.e30.vz0.all.MuDst.root     the histograms
#        qa_pi0.e30.vz0.all.MuDst.log      the printed per-variable table
#
# Run it from the directory that holds StRoot/ and the macros, after cons.
# A MuDst is treated as SIMULATION (isSim=1: energies from dE, no gain
# tables); for real-data MuDst run runMudst_ml.C with isSim=0 yourself and
# then qaFeatures.C on its output.
# If the feature file already exists, step 1 is skipped - delete it to re-dump.

set -e
in="$1"
nevt="${2:--1}"
emin="${3:-0.5}"
emax="${4:-0}"
if [ -z "$in" ] || [ ! -f "$in" ]; then
   echo "usage: $0 <file.picoDst.root | file.MuDst.root> [nevt] [eMin] [eMax]"
   exit 1
fi

tag=$(basename "$in" .root)
feat="feat_${tag}.root"
qa="qa_${tag}"

if [ -f "$feat" ]; then
   echo "== $feat exists, skipping the dump"
else
   case "$in" in
      *picoDst*)
         echo "== dumping features from picoDst $in"
         root4star -b -q "runPicoDst_ml.C(\"$in\",$nevt,0,3,\"\",\"$feat\",\"BDTG\",$emin)" 2>&1 | tee "dump_${tag}.log"
         ;;
      *MuDst*)
         echo "== dumping features from MuDst $in (simulation)"
         # runMudst_ml.C reads everything with nevt = -2
         mnevt=$nevt
         [ "$mnevt" -lt 0 ] && mnevt=-2
         root4star -b -q "runMudst_ml.C(\"$in\",-1,$mnevt,\".\",1,0,0,\"\",\"$feat\",1)" 2>&1 | tee "dump_${tag}.log"
         ;;
      *)
         echo "cannot tell whether $in is a picoDst or a MuDst from its name"
         exit 1
         ;;
   esac
fi

if [ ! -f "$feat" ]; then
   echo "no $feat was written - see dump_${tag}.log"
   exit 1
fi

echo "== QA plots for feature sets 3 and 13"
root4star -b -q "qaFeatures.C+(\"$feat\",\"$qa\",$emin,$emax)" 2>&1 | tee "${qa}.log"
echo "== done: ${qa}.pdf"
