#!/bin/bash
# runCrossCheck.sh - does the difference between two samples come from the
# DATA or from the TRAINING?
#
#   ./runCrossCheck.sh <featA.root> <nameA> <featB.root> <nameB>
#
#   ./runCrossCheck.sh feat_pico_flat.root flat feat_pico_mix1.root mix
#
# Two feature files, two models, four evaluations:
#
#                        evaluated on A        evaluated on B
#   model trained on A    A/A  (in-sample)      A/B  (transfer)
#   model trained on B    B/A  (transfer)       B/B  (in-sample)
#
# Read it like this:
#   numbers follow the ROW   -> the MODEL is different (training, weights,
#                               class prior) and carries its behaviour with it
#   numbers follow the COLUMN-> the DATA is different and both models are
#                               reacting the same way to the same clusters
#   neither                  -> the two interact; look at the per-energy pages
#
# WHY IT RETRAINS BOTH. Comparing an old weight file against a new one confuses
# "different sample" with "different trainTMVA arguments" - labelDef, weightMode
# and the feature set all change the model and none of them is recorded in the
# evaluation PDF. Both models here are trained from scratch, in this script,
# with identical settings, so the only thing that differs is the input file.
#
# THE FCS CLUSTER ROW IS THE CONTROL. catStar is a fixed cut and cannot learn,
# so any change in ITS numbers between the two files is pure sample composition.
# Subtract that and what is left is the model. Every evaluation prints it.
#
# OVERLAPPING SAMPLES. If B is a hadd that CONTAINS A (a mix built on top of the
# flat sample), the split is by event ordinal in tree order, so A's test half is
# not guaranteed to be B's test half - the parity depends on how many events sit
# in front of A's block in B. The script therefore also evaluates each model on
# the TRAINING half of each file and prints both: if the two halves agree, no
# leak is biasing anything (the measured train/test gap on these samples is
# <= 0.006, so this is expected, but it is checked rather than assumed). To
# remove the question entirely, build B from files A does not contain.
#
# Environment overrides:
#   FEATURESET=3   feature set            METHOD=MLP   method to table
#   EMIN=0.5       cluster energy cut     NORMMODE=0   1 = EqualNumEvents
#   WEIGHTMODE=1   0 turns OFF the flat-in-cluster-energy training weights
#   SKIPTRAIN=1    reuse existing weight files instead of retraining
#
# WEIGHTMODE=0 is worth one run of its own. The flat-in-energy weight is
# 1/N(class, E bin), capped at 10x the class median. On a sample built from
# fixed-energy guns the CLUSTER energy spectrum is spiky for gamma and merged
# pi0 (they keep nearly all the gun energy) but smooth for pi- (it deposits a
# random fraction), so the sparse bins BETWEEN the spikes hold few gamma
# clusters and those get pushed to the cap. An MLP trained by back-propagation
# feels that directly - a 10x cluster moves the weights 10x per step - while a
# BDT only sees weights as bin sums. Same file, MLP breaks and BDTG does not,
# is the signature.

set -e
fA="$1"; nA="$2"; fB="$3"; nB="$4"
if [ -z "$fB" ]; then
   echo "usage: $0 <featA.root> <nameA> <featB.root> <nameB>"
   echo "   eg: $0 feat_pico_flat.root flat feat_pico_mix1.root mix"
   exit 1
fi
for f in "$fA" "$fB"; do
   [ -f "$f" ] || { echo "missing input: $f"; exit 1; }
done

fset="${FEATURESET:-3}"
emin="${EMIN:-0.5}"
meth="${METHOD:-MLP}"
nrm="${NORMMODE:-0}"
wgt="${WEIGHTMODE:-1}"
sfx="gen"; [ "$wgt" = "1" ] && sfx="genw"
[ "$nrm" = "1" ] && sfx="${sfx}eq"
jA="X${nA}"; jB="X${nB}"
wA="weights/${jA}${fset}${sfx}_${meth}.weights.xml"
wB="weights/${jB}${fset}${sfx}_${meth}.weights.xml"

echo "== cross check: $nA ($fA) vs $nB ($fB)"
echo "   set $fset, method $meth, eMin $emin, weightMode $wgt, normMode $nrm"

# ---- 1. train one model per sample, identical settings ----------------------
if [ "${SKIPTRAIN:-0}" = "1" ]; then
   echo "== SKIPTRAIN=1, reusing existing weight files"
else
   for p in "$jA:$fA" "$jB:$fB"; do
      j="${p%%:*}"; f="${p#*:}"
      echo "== training $j on $f"
      root4star -b -q "trainTMVA.C+(\"$f\",\"$j\",$fset,\"clusters\",0.8,$emin,1,$wgt,$nrm)" \
         2>&1 | tee "xtrain_${j}.log"
   done
fi
for w in "$wA" "$wB"; do
   [ -f "$w" ] || { echo "no $w - see the xtrain_*.log above"; exit 1; }
done

# ---- 2. four evaluations, test half; plus the training half as a leak check --
#   tag = x_<weights>W_<data>D[_tr]
for wp in "$nA:$wA" "$nB:$wB"; do
   wn="${wp%%:*}"; w="${wp#*:}"
   for dp in "$nA:$fA" "$nB:$fB"; do
      dn="${dp%%:*}"; d="${dp#*:}"
      for half in 1 0; do
         sfx2=""; [ "$half" = "0" ] && sfx2="_tr"
         out="x_${wn}W_${dn}D${sfx2}"
         echo "== $meth trained on $wn, evaluated on $dn ($([ $half = 1 ] && echo test || echo training) half)"
         root4star -b -q \
            "evalCategory.C+(\"$d\",\"$w\",$fset,\"$meth\",$half,\"$out\",$emin,0.8,\"clusters\",1,1)" \
            2>&1 | tee "${out}.log"
      done
   done
done

# ---- 3. the 2x2 table, scraped from the logs --------------------------------
# evalCategory prints, after the model's confusion matrix:
#   class          efficiency            purity   purity(balanced)
#   other       0.690 +- 0.003    0.812 +- 0.003    0.845
# and the sample's class mix. purity(balanced) is the one to compare across
# samples: plain purity moves when the gamma : pi0 : pi- mixture moves, even
# with an identical model, while efficiency and purity(balanced) do not.
scrape() {   # $1 = log, $2 = class row, $3 = field (2 eff, 5 pur, 8 purbal)
   awk -v cls="$2" -v fld="${3:-2}" '
      /^--- '"$meth"' ---/      { inml = 1 }
      /catStar branch/          { inml = 0 }
      inml && $1 == cls && NF >= 8 { print $fld; exit }
   ' "$1" 2>/dev/null
}
starEff() {  # $1 = log, $2 = class row
   awk -v cls="$2" '
      /catStar branch/            { inst = 1 }
      inst && $1 == cls && NF >= 8 { print $2; exit }
   ' "$1" 2>/dev/null
}

echo ""
echo "================= $meth, set $fset, test half, efficiency ================="
printf "  %-22s %12s %12s %12s\n" "model \\ data" "other(had)" "onePhoton" "twoPhoton"
for wn in "$nA" "$nB"; do
   for dn in "$nA" "$nB"; do
      L="x_${wn}W_${dn}D.log"
      printf "  %-22s %12s %12s %12s\n" "trained $wn -> $dn" \
         "$(scrape $L other)" "$(scrape $L onePhoton)" "$(scrape $L twoPhoton)"
   done
done
echo ""
echo "  --- FCS Cluster on the same clusters (the control: it cannot learn) ---"
for dn in "$nA" "$nB"; do
   L="x_${nA}W_${dn}D.log"
   printf "  %-22s %12s %12s %12s\n" "$dn data" "-" \
      "$(starEff $L onePhoton)" "$(starEff $L twoPhoton)"
done
echo ""
echo "================= same, purity(balanced) ================="
printf "  %-22s %12s %12s %12s\n" "model \\ data" "other(had)" "onePhoton" "twoPhoton"
for wn in "$nA" "$nB"; do
   for dn in "$nA" "$nB"; do
      L="x_${wn}W_${dn}D.log"
      printf "  %-22s %12s %12s %12s\n" "trained $wn -> $dn" \
         "$(scrape $L other 8)" "$(scrape $L onePhoton 8)" "$(scrape $L twoPhoton 8)"
   done
done
echo ""
echo "  --- test half vs training half (leak / overtraining check) ---"
for wn in "$nA" "$nB"; do
   for dn in "$nA" "$nB"; do
      printf "  %-22s test %-8s train %-8s\n" "trained $wn -> $dn" \
         "$(scrape x_${wn}W_${dn}D.log other)" "$(scrape x_${wn}W_${dn}D_tr.log other)"
   done
done
echo ""
echo "  --- class mix of each file (what plain purity is sensitive to) ---"
grep -h "class mix of this sample" x_${nA}W_${nA}D.log x_${nA}W_${nB}D.log | sed 's/^/  /'

# ---- 4. the figures: two models on one plot, once per dataset ---------------
for dn in "$nA" "$nB"; do
   a="x_${nA}W_${dn}D.root"; b="x_${nB}W_${dn}D.root"
   if [ -f "$a" ] && [ -f "$b" ]; then
      echo "== figures: both models on the $dn sample"
      root4star -b -q "compareModels.C+(\"$a\",\"trained on $nA\",\"$b\",\"trained on $nB\",\"\",\"\",\"\",\"\",\"xcmp_on_${dn}\")" \
         2>&1 | tee "xcmp_on_${dn}.log"
   fi
done

echo ""
echo "== done."
echo "   the 2x2 above        rows move -> the model; columns move -> the data"
echo "   xcmp_on_${nA}.pdf / xcmp_on_${nB}.pdf   the same split, per energy bin"
echo "   xtrain_*.log         class counts and the flat-in-energy weight table"
echo "   compare the class counts in the two xtrain logs first - with"
echo "   NormMode=NumEvents they ARE the class prior the model learned"
