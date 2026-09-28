// trainTMVA.C - train the FCS ECal cluster category classifier with TMVA.
//
// Run it with the SAME ROOT that root4star uses, i.e. inside the SL7 container
// after starver. Compile it with ACLiC - note the trailing '+':
//
//     root4star -b -q 'trainTMVA.C+("fcsEcalClusterFeatures.root","FcsCat",13)'
//
// The '+' matters. Interpreted, this macro goes through CINT, which parses
// namespaces and inline functions in StFcsClusterFeatures.h only
// approximately; ACLiC hands it to the real compiler instead - the same one
// cons uses - so the feature code that runs here is the code that runs in the
// makers. Without the '+' you may get a bare
//     Error: Too many '}' tmpfile:NN
// which is CINT giving up on the parse, not a problem with your input file.
// TMVA weight XML is not guaranteed to be readable across ROOT major versions,
// and a weight file trained in some other ROOT is the classic way to lose a week.
//
// Nothing to install: TMVA ships inside ROOT, which ships inside the STAR
// library stack.
//
// Input : the `clusters` tree written by StFcsClusterFeatureMaker (.fzd or
//         MuDst) or StFcsPicoFeatureMaker (picoDst) - one schema, either way,
//         run on simulation so the truth branches are filled
// Output: weights/<jobname><set>_BDTG.weights.xml , ..._MLP.weights.xml
//         and <jobname><set>.root for the TMVA GUI
//
// The input variables come from StFcsClusterFeatures.h - the same header the
// inference maker uses - so training and application cannot drift apart.
//
// Feature sets: 3, 6, 10, 13 or 34. Set 3 is the 3x3 one - the id names the
// window, not the variable count, and it has 13 variables. It is the good
// starting point: raw tower energies plus E, sigmaMax, sigmaMin and E1/E, with
// none of the correlated derived triplets of set 13.
//
// Set 10 is set 13 without sigX, sigY and sigXY - the three detector-frame
// moments that are algebraically the same information as sigmaMax, sigmaMin
// and theta. Train 10 and 13 on the same file and compare them with
// compareModels.C: if they tie, prefer 10, because correlated inputs make the
// TMVA variable ranking misleading and give the MLP a singular Hessian.
//
// Set 6 is the tower-free one: it is computable from any input without
// reconstructing anything, which used to be the only way to work on picoDst.
// StPicoFcsCluster still stores no tower list, but StFcsTowerAssoc.h now
// recovers it geometrically (92 % exact tower count, median energy difference
// zero - see BUILD.md), so sets 3, 13 and 34 work on picoDst too. Set 6 costs
// you the tower-level shape detail, so expect it to separate 1-photon from
// 2-photon clusters less sharply; train both on the same sample and compare.
//
// The input file can come from a .fzd, a MuDst or a picoDst - all three write
// this same tree, and all three carry the generator-level label mcLabel.
//
// LABELS (labelDef, the last argument)
//   0 (default) by the photons inside the cluster (mcLabel): other /
//     onePhoton / twoPhoton. Every cluster of every sample is used.
//   1 ePIC-style, by the generated particle (genPid) of a single-particle
//     sample, cleaned with mcLabel - see StFcsTrainTestSplit.h:
//       class 0 hadronic   = every cluster of a pi- event
//       class 1 single EM  = gamma-event cluster holding the photon
//       class 2 merged pi0 = pi0-event cluster holding BOTH photons
//     Resolved pi0 photons ("pi0 2-cluster input") and gun fragments are not
//     trained on; evalCategory.C still shows their scores. Needs genPid, so
//     re-dump older feature files. The job gets a "gen" suffix so it does not
//     overwrite the mcLabel model:
//       root4star -b -q 'trainTMVA.C+("feat_pico_all.root","FcsCat",13,"clusters",0.8,0.5,1)'
//       -> weights/FcsCat13gen_BDTG.weights.xml
//     The class order - and so the order of the three scores - is the same
//     for both: r[0] hadronic/other, r[1] single EM, r[2] merged pi0.
//
// TRAINING WEIGHTS (weightMode, the argument after labelDef)
//   0 (default) every cluster counts once.
//   1 FLAT IN CLUSTER ENERGY. Each cluster is weighted by 1/N(class, energy
//     bin), so within any energy bin the three classes are equally common.
//
//     Why: the classes are not mixed evenly along the energy axis. A 60 GeV
//     pi- leaves several clusters of 1-3 GeV, while a gamma leaves one cluster
//     carrying nearly all of its energy - so the low-energy end of the sample
//     is almost pure hadron and the high-energy end almost pure photon. The
//     model sees the energy (logE in set 13, e in set 3) and learns that mix
//     as a prior: single-EM efficiency collapses below 5 GeV and hadron
//     efficiency collapses above 35 GeV, in both BDTG and MLP. Weighting
//     removes the prior and leaves the shower shape to decide.
//
//     The binning is in CLUSTER energy - the quantity the model is given, not
//     the generated energy. A bin with fewer than kMinBin clusters of a class
//     is merged into its neighbour, and weights are capped at kWCap times the
//     class median so a handful of clusters cannot dominate the loss. The
//     printout lists, per class, the weight range and how many bins were
//     merged or capped; look at it before trusting a model.
//
//     WHAT IT DOES AND DOES NOT DO. Each class ends up with a FLAT energy
//     spectrum, and TMVA then scales the three classes to equal totals. So the
//     brutal part of the prior goes away - a 9:1 hadron majority in the lowest
//     bin becomes roughly 1.5:1 - but the classes are not made exactly equal
//     bin by bin: a class present in fewer energy bins keeps a larger share in
//     the bins where it does live. Nothing can be done where a class is simply
//     absent (merged pi0 below ~10 GeV), and nothing should be: that is
//     kinematics, not a sampling artefact.
//
//     The job name gets a "w" suffix, so a weighted model never overwrites an
//     unweighted one:
//       trainTMVA.C+("feat_all.root","FcsCat",13,"clusters",0.8,0.5,1,1)
//       -> weights/FcsCat13genw_BDTG.weights.xml
//
// author: generated for Xilin Liang

#include <algorithm>
#include <vector>

#include "TCut.h"
#include "TFile.h"
#include "TString.h"
#include "TSystem.h"
#include "TTree.h"
#include "TMVA/Factory.h"
#include "TMVA/Tools.h"

#include "StRoot/StFcsMLCategoryMaker/StFcsClusterFeatures.h"
#include "StRoot/StFcsMLCategoryMaker/StFcsTrainTestSplit.h"

void trainTMVA(const char* infile = "fcsEcalClusterFeatures.root",
               const char* jobname = "FcsCat",
               int featureSet = 13,          // 3, 6, 10, 13 or 34 - see the note above
               const char* treename = "clusters",
               float purityCut = 0.8,
               float eMin = 0.5,
               int labelDef = 0,             // 0 mcLabel, 1 generated particle (ePIC-style)
               int weightMode = 0) {         // 0 unweighted, 1 flat in cluster energy
   using namespace StFcsClusterFeatures;
   gSystem->Load("libTMVA");
   TMVA::Tools::Instance();

   const int NVAR = nVar(featureSet);
   const char** varname = varNames(featureSet);
   if (labelDef != 1) labelDef = 0;
   if (weightMode != 1) weightMode = 0;
   const TString job = Form("%s%d%s%s", jobname, featureSet, labelDef == 1 ? "gen" : "",
                            weightMode == 1 ? "w" : "");
   printf("training feature set %d (%d variables), job %s\n", featureSet, NVAR, job.Data());

   // ---------------------------------------------------------------- input
   TFile* fin = TFile::Open(infile);
   if (!fin || fin->IsZombie()) { printf("cannot open %s\n", infile); return; }
   TTree* in = (TTree*)fin->Get(treename);
   if (!in) { printf("no tree %s in %s\n", treename, infile); return; }

   const int NW = 11;  // StFcsClusterFeatureMaker::kNW
   Float_t e, x, y, sigmaMin, sigmaMax, theta, xw, yw, truthPurity;
   Float_t img[NW * NW], mask[NW * NW];
   Int_t nTowers, nNeighbor, seedRow, seedCol, catStar, truthNPhoton, mcLabel;
   Int_t run = 0, event = 0;
   in->SetBranchAddress("run", &run);      // for the train/test split, see below
   in->SetBranchAddress("event", &event);
   in->SetBranchAddress("e", &e);
   in->SetBranchAddress("x", &x);
   in->SetBranchAddress("y", &y);
   in->SetBranchAddress("sigmaMin", &sigmaMin);
   in->SetBranchAddress("sigmaMax", &sigmaMax);
   in->SetBranchAddress("theta", &theta);
   in->SetBranchAddress("xw", &xw);
   in->SetBranchAddress("yw", &yw);
   in->SetBranchAddress("img", img);
   in->SetBranchAddress("mask", mask);
   in->SetBranchAddress("nTowers", &nTowers);
   in->SetBranchAddress("nNeighbor", &nNeighbor);
   in->SetBranchAddress("seedRow", &seedRow);
   in->SetBranchAddress("seedCol", &seedCol);
   in->SetBranchAddress("catStar", &catStar);
   in->SetBranchAddress("truthNPhoton", &truthNPhoton);
   // generator-level label; -1 on feature files written before it existed
   mcLabel = -1;
   if (in->GetBranch("mcLabel")) in->SetBranchAddress("mcLabel", &mcLabel);
   in->SetBranchAddress("truthPurity", &truthPurity);
   Int_t genPid = 0;
   if (in->GetBranch("genPid")) {
      in->SetBranchAddress("genPid", &genPid);
   } else if (labelDef == 1) {
      printf("labelDef=1 labels by the generated particle, but %s has no genPid branch.\n"
             "Re-dump it with the current feature maker, or use labelDef=0.\n", infile);
      return;
   }

   // ------------------------------------------------- flat training trees
   TFile* ftmp = new TFile(job + "_train.root", "RECREATE");
   Float_t v[kNVarMax];
   TTree* t[3];
   const char* clsname[3] = {"other", "onePhoton", "twoPhoton"};
   // TRAIN/TEST SPLIT. Not TMVA's SplitMode=Random: that picks the test half
   // with an internal seed nobody outside TMVA can reproduce, so a later
   // efficiency/purity study (evalCategory.C) could not avoid the training
   // clusters. The split is made here instead, by whole event, with the rule
   // in StFcsTrainTestSplit.h that evalCategory.C applies too.
   Float_t isTest = 0;
   // ---------------------------------------------------- training weights
   // Bin edges in CLUSTER energy: fine where the class mix changes fastest,
   // coarse where it does not. The last bin is open-ended.
   const int kNWB = 14;
   const double wEdge[kNWB + 1] = {0.0, 1, 2, 3, 5,  7,  10, 15,
                                   20,  25, 30, 40, 50, 60, 1e9};
   const int kMinBin = 50;    // fewer clusters than this: merge into the neighbour
   const double kWCap = 10.0; // cap a weight at this times the class median
   double wCount[3][kNWB];
   for (int c = 0; c < 3; c++)
      for (int b = 0; b < kNWB; b++) wCount[c][b] = 0;
   double wVal[3][kNWB];  // the weight each (class, bin) gets; filled after pass 0
   for (int c = 0; c < 3; c++)
      for (int b = 0; b < kNWB; b++) wVal[c][b] = 1.0;
   Float_t wgt = 1.0;

   for (int c = 0; c < 3; c++) {
      t[c] = new TTree(clsname[c], clsname[c]);
      for (int i = 0; i < NVAR; i++)
         t[c]->Branch(varname[i], &v[i], Form("%s/F", varname[i]));
      t[c]->Branch("isTest", &isTest, "isTest/F");
      t[c]->Branch("w", &wgt, "w/F");
   }

   const int half = NW / 2;
   float te[NW * NW];
   int trow[NW * NW], tcol[NW * NW];
   Long64_t n = in->GetEntries();
   Long64_t kept[3] = {0, 0, 0};
   Long64_t keptTest[3] = {0, 0, 0};

   // accounting, so that an empty training sample explains itself
   const bool haveMcBranch = (in->GetBranch("mcLabel") != 0);
   Long64_t nBelowE = 0, nNoTruth = 0, nImpure = 0, nNoTowers = 0, nNotSample = 0;

   // range of every input variable over the kept clusters. TMVA aborts the
   // whole job on a constant one - see the check after the loop.
   Float_t vmin[kNVarMax], vmax[kNVarMax];
   for (int i = 0; i < NVAR; i++) {
      vmin[i] = 1e30;
      vmax[i] = -1e30;
   }

   // With weightMode = 1 the clusters are read TWICE: pass 0 only counts them
   // per (class, energy bin), pass 1 fills the trees with the weights that
   // count implies. Everything else - cuts, labels, features - is identical in
   // the two passes, so the weights describe exactly the clusters that are
   // trained on. Unweighted, pass 1 runs alone.
   long nEventsSeen = 0;
   for (int pass = (weightMode == 1 ? 0 : 1); pass < 2; pass++) {
   const bool counting = (pass == 0);
   StFcsTrainTestSplit::EventSplitter splitter;
   nBelowE = nNoTruth = nImpure = nNoTowers = nNotSample = 0;
   for (int c = 0; c < 3; c++) kept[c] = keptTest[c] = 0;

   for (Long64_t i = 0; i < n; i++) {
      in->GetEntry(i);
      // before ANY selection, so the event count - and with it the split -
      // does not depend on the cuts
      isTest = (Float_t)splitter.isTest(run, event);
      if (e < eMin) {
         nBelowE++;
         continue;
      }

      // LABEL SOURCE. mcLabel is generator level: it counts the GENERATED
      // photons projecting onto this cluster, so it does not care how GEANT
      // shared the energy deposits between primary and shower tracks. Prefer it
      // whenever it is present, and note that it needs no purity cut - which
      // matters for hadronic clusters, whose energy is spread over many tracks
      // so their leading-track purity is naturally low and the old rule threw
      // most of them away, leaving class 0 nearly empty.
      //
      // truthNPhoton (hit level) is the fallback for feature files made before
      // the generator-level branches existed.
      // the rule itself lives in StFcsTrainTestSplit.h, so that evalCategory.C
      // judges the model against exactly the labels it was trained on
      int cls = StFcsTrainTestSplit::trainingLabel(mcLabel, truthNPhoton, truthPurity, purityCut);
      if (labelDef == 1 && cls >= 0) {
         // ePIC-style: the generated particle decides; resolved pi0 photons
         // and gun fragments are left out (StFcsTrainTestSplit.h)
         cls = StFcsTrainTestSplit::sampleLabel(genPid, mcLabel);
         if (cls < 0) {
            nNotSample++;
            continue;
         }
      }
      if (cls < 0) {
         if (mcLabel < 0 && truthNPhoton < 0)
            nNoTruth++;  // no truth at all on this cluster
         else
            nImpure++;   // truth present but the purity cut rejected it
         continue;
      }

      // rebuild the cluster's towers from the stored image and mask
      int nTow = 0;
      for (int dr = -half; dr <= half; dr++) {
         for (int dc = -half; dc <= half; dc++) {
            const int pix = (dr + half) * NW + (dc + half);
            if (mask[pix] <= 0 || img[pix] <= 0) continue;
            te[nTow] = img[pix];
            trow[nTow] = seedRow + dr;
            tcol[nTow] = seedCol + dc;
            nTow++;
         }
      }

      ClusterInput c;
      c.e = e;
      c.x = x;
      c.y = y;
      c.sigmaMin = sigmaMin;
      c.sigmaMax = sigmaMax;
      c.theta = theta;
      c.nTowers = nTowers;
      c.nNeighbor = nNeighbor;
      c.xw = xw;
      c.yw = yw;
      c.nTow = nTow;
      c.towerE = te;
      c.towerRow = trow;
      c.towerCol = tcol;

      if (compute(featureSet, c, v) != NVAR) {
         nNoTowers++;
         continue;
      }
      // which energy bin this cluster belongs to
      int eb = 0;
      while (eb < kNWB - 1 && e >= wEdge[eb + 1]) eb++;
      if (counting) {
         wCount[cls][eb] += 1;
         continue;
      }
      for (int i = 0; i < NVAR; i++) {
         if (v[i] < vmin[i]) vmin[i] = v[i];
         if (v[i] > vmax[i]) vmax[i] = v[i];
      }
      wgt = (weightMode == 1) ? (Float_t)wVal[cls][eb] : 1.0f;
      t[cls]->Fill();
      kept[cls]++;
      if (isTest > 0.5) keptTest[cls]++;
   }
   nEventsSeen = splitter.nEvents();

   // ---- end of the counting pass: turn the counts into weights ----
   if (counting) {
      printf("\n--- flat-in-energy training weights (bins of CLUSTER energy) ---\n");
      printf("  %-11s %8s %9s %9s %9s %7s %7s\n", "class", "clusters", "bins used", "min w", "max w",
             "merged", "capped");
      for (int c = 0; c < 3; c++) {
         // merge a bin with too few clusters into the next one up, so a
         // handful of clusters cannot become a huge weight
         double nEff[kNWB];
         int owner[kNWB];  // which bin a cluster of this bin is counted in
         for (int b = 0; b < kNWB; b++) {
            nEff[b] = wCount[c][b];
            owner[b] = b;
         }
         int nMerged = 0;
         for (int b = 0; b < kNWB - 1; b++) {
            if (nEff[b] <= 0 || nEff[b] >= kMinBin) continue;
            nEff[b + 1] += nEff[b];
            nEff[b] = 0;
            for (int q = 0; q <= b; q++)
               if (owner[q] == b) owner[q] = b + 1;
            nMerged++;
         }
         // the last bin cannot merge upward: fold it downward instead
         if (nEff[kNWB - 1] > 0 && nEff[kNWB - 1] < kMinBin) {
            for (int b = kNWB - 2; b >= 0; b--) {
               if (nEff[b] <= 0) continue;
               nEff[b] += nEff[kNWB - 1];
               nEff[kNWB - 1] = 0;
               for (int q = 0; q < kNWB; q++)
                  if (owner[q] == kNWB - 1) owner[q] = b;
               nMerged++;
               break;
            }
         }
         double total = 0;
         int nUsed = 0;
         for (int b = 0; b < kNWB; b++) {
            total += nEff[b];
            if (nEff[b] > 0) nUsed++;
         }
         if (total <= 0 || nUsed == 0) continue;
         // equal total weight in every populated bin, mean weight 1
         std::vector<double> ws;
         for (int b = 0; b < kNWB; b++) {
            const double nb = nEff[owner[b]];
            wVal[c][b] = (nb > 0) ? (total / nUsed) / nb : 1.0;
            if (wCount[c][b] > 0) ws.push_back(wVal[c][b]);
         }
         std::sort(ws.begin(), ws.end());
         const double med = ws.empty() ? 1.0 : ws[ws.size() / 2];
         int nCap = 0;
         double lo = 1e30, hi = 0;
         for (int b = 0; b < kNWB; b++) {
            if (wCount[c][b] <= 0) continue;
            if (wVal[c][b] > kWCap * med) {
               wVal[c][b] = kWCap * med;
               nCap++;
            }
            if (wVal[c][b] < lo) lo = wVal[c][b];
            if (wVal[c][b] > hi) hi = wVal[c][b];
         }
         // renormalise so the average weight of this class is 1, which keeps
         // TMVA's own NormMode doing what it did before
         double sw = 0;
         for (int b = 0; b < kNWB; b++) sw += wCount[c][b] * wVal[c][b];
         const double scale = (sw > 0) ? (total / sw) : 1.0;
         for (int b = 0; b < kNWB; b++) wVal[c][b] *= scale;
         printf("  %-11s %8.0f %9d %9.3g %9.3g %7d %7d\n", clsname[c], total, nUsed, lo * scale, hi * scale,
                nMerged, nCap);
      }
      printf("  a class whose clusters all sit in one energy bin gets weight 1 everywhere\n");
   }
   }  // pass
   // ------------------------------------------------------------ accounting
   printf("\n--- where the %lld clusters in %s went ---\n", n, infile);
   printf("  below eMin = %.2f GeV        : %lld\n", eMin, nBelowE);
   printf("  no truth on the cluster      : %lld\n", nNoTruth);
   printf("  truth present, cut by purity : %lld\n", nImpure);
   printf("  no usable tower list         : %lld\n", nNoTowers);
   if (labelDef == 1)
      printf("  not a training class (labelDef=1: resolved pi0 photon, fragment, other gun): %lld\n", nNotSample);
   printf("  KEPT  other=%lld  1photon=%lld  2photon=%lld\n", kept[0], kept[1], kept[2]);
   printf("  split by event (%ld events): train %lld / %lld / %lld, test %lld / %lld / %lld\n",
          nEventsSeen, kept[0] - keptTest[0], kept[1] - keptTest[1], kept[2] - keptTest[2],
          keptTest[0], keptTest[1], keptTest[2]);
   if (weightMode == 1) printf("  training weights: flat in cluster energy (see the table above)\n");
   printf("  label source: %s\n", labelDef == 1 ? "generated particle (genPid), cleaned with mcLabel"
                                 : haveMcBranch  ? "mcLabel (generator level)"
                                                 : "truthNPhoton (hit level); no mcLabel branch");
   if (labelDef == 1)
      printf("  classes: other = hadronic (pi-), onePhoton = single EM (gamma), twoPhoton = merged pi0\n");

   if (kept[0] + kept[1] + kept[2] == 0) {
      printf("\nNothing to train on. Reading the numbers above:\n");
      if (n == 0) {
         printf("  The tree is empty. The feature dumper ran but wrote no clusters -\n"
                "  check the ECal cluster count in the job that produced it.\n");
      } else if (nNoTruth == n - nBelowE) {
         printf("  Every cluster has no truth, so the input carried no MC arrays.\n"
                "  All three inputs CAN be labelled - the generator-level truth survives\n"
                "  into every tier - so check, in order:\n"
                "    .fzd     : is fcsSim in the chain? no fast simulator, no hits\n"
                "    MuDst    : does it have StMuMcTrack/StMuMcVertex branches, and was\n"
                "               StFcsMuMcTruthMaker constructed BEFORE the dumper?\n"
                "    picoDst  : SetStatus(\"McTrack*\",1) and (\"McVertex*\",1), and was the\n"
                "               picoDst produced from a MuDst that had the MC arrays?\n"
                "  Only the HIT-level branches (trkPid, truthNPhoton) are exclusive to the\n"
                "  .fzd; mcLabel, which is what this macro trains on, is not.\n");
      } else if (nBelowE == n) {
         printf("  Every cluster is below eMin = %.2f GeV. Lower it, or check the energy\n"
                "  scale in the dumper.\n", eMin);
      } else if (nImpure > 0) {
         printf("  Truth is present but the purity cut rejected everything. Lower\n"
                "  purityCut, or re-dump with the generator-level branches so mcLabel\n"
                "  is used instead - it needs no purity cut.\n");
      }
      printf("\nStopping before TMVA, which would abort on empty trees.\n");
      ftmp->Close();
      return;
   }
   if (kept[1] < 100 || kept[2] < 100)
      printf("  WARNING: thin classes - the model will not be worth much yet\n");
   printf("\n");

   // --------------------------------------------------------- split check
   //
   // The split counts events by watching (run, event) change between
   // consecutive tree entries. If a dumper ever wrote the same run and event
   // number on every cluster, that would be ONE event, everything would land on
   // the training side, and TMVA would have nothing to test on.
   for (int c = 0; c < 3; c++) {
      if (kept[c] == 0) continue;
      if (keptTest[c] == 0 || keptTest[c] == kept[c]) {
         printf("\nThe train/test split put every '%s' cluster on one side.\n", clsname[c]);
         printf("It splits by event, using the run and event branches, and found only\n");
         printf("%ld event(s) in %lld clusters - those branches are probably not filled.\n",
                nEventsSeen, n);
         printf("Stopping before TMVA.\n");
         ftmp->Close();
         return;
      }
   }

   // ------------------------------------------------- constant-variable check
   //
   // TMVA does not skip a variable that never changes, it kills the job:
   //   <FATAL> DataSetFactory : Variable nNeighbor is constant. Please remove
   //                            the variable.
   //   ***> abort program execution
   // and by then it has already printed two screens of setup, so the cause is
   // easy to miss. Catch it here, where the reason can be explained.
   //
   // Dropping the offending variable automatically would be worse than
   // stopping: the weight file would then hold NVAR-1 variables while
   // StFcsClusterFeatures::compute() still produces NVAR in a fixed order, and
   // TMVA::Reader would refuse the model at application time - or, worse,
   // silently pair up the wrong ones. Training and inference share one
   // definition of the inputs in this package, and that is worth keeping.
   {
      int nConst = 0;
      for (int i = 0; i < NVAR; i++) {
         if (vmax[i] > vmin[i]) continue;
         if (nConst == 0)
            printf("--- constant input variables (TMVA would abort on these) ---\n");
         printf("  %-12s is always %g\n", varname[i], vmin[i]);
         nConst++;
      }
      if (nConst > 0) {
         printf("\nA variable with no spread carries no information, and TMVA refuses it.\n");
         printf("Usual cause: the input tier does not store that quantity, so the dumper\n");
         printf("wrote a placeholder on every cluster. nNeighbor and nPoints are the two\n");
         printf("that picoDst does not carry directly.\n\n");
         printf("Options, in order of preference:\n");
         printf("  - use a feature set that does not need it. Only set 13 uses nNeighbor;\n");
         printf("    sets 3, 6 and 34 do not:\n");
         printf("      trainTMVA.C+(\"%s\",\"%s\",3)\n", infile, jobname);
         printf("  - re-dump with a maker that fills it. StFcsPicoFeatureMaker computes\n");
         printf("    nNeighbor from the recovered tower adjacency; a feature file made\n");
         printf("    before that existed has it at -1 everywhere.\n");
         printf("  - if the sample itself is the reason (a single-particle gun where no\n");
         printf("    cluster ever has a neighbour), train on a mixed sample, or use set 3.\n");
         printf("\nStopping before TMVA.\n");
         ftmp->Close();
         return;
      }
   }

   // --------------------------------------------------------------- TMVA
   TFile* fout = TFile::Open(job + ".root", "RECREATE");
   TMVA::Factory* factory = new TMVA::Factory(
       job, fout,
       "!V:!Silent:Color:DrawProgressBar:Transformations=I;N:AnalysisType=multiclass");

   // ROOT 5 TMVA API: the Factory owns the variables and the trees directly.
   // There is deliberately no #if ROOT_VERSION here - CINT mishandles
   // preprocessor branches inside a function body and reports
   //   Error: Too many '}' tmpfile:NN
   // which is a parse failure, not a problem with your input file. STAR DEV is
   // ROOT 5.34.38. On ROOT 6.08+ the same calls move to a TMVA::DataLoader:
   //   TMVA::DataLoader* dl = new TMVA::DataLoader("dataset");
   //   dl->AddVariable(...); dl->AddTree(...); dl->PrepareTrainingAndTestTree(...);
   //   factory->BookMethod(dl, TMVA::Types::kBDT, "BDTG", "...");
   for (int i = 0; i < NVAR; i++) factory->AddVariable(varname[i], 'F');
   // Each class tree goes in twice, once per side of the split, so TMVA trains
   // and tests on exactly the clusters StFcsTrainTestSplit assigned.
   for (int c = 0; c < 3; c++) {
      factory->AddTree(t[c], clsname[c], 1.0, TCut("isTest<0.5"), TMVA::Types::kTraining);
      factory->AddTree(t[c], clsname[c], 1.0, TCut("isTest>0.5"), TMVA::Types::kTesting);
   }
   if (weightMode == 1)
      for (int c = 0; c < 3; c++) factory->SetWeightExpression("w", clsname[c]);
   factory->PrepareTrainingAndTestTree("", "NormMode=NumEvents:!V");
   factory->BookMethod(TMVA::Types::kBDT, "BDTG",
                       "!H:!V:NTrees=600:MaxDepth=4:BoostType=Grad:Shrinkage=0.10:"
                       "UseBaggedBoost:BaggedSampleFraction=0.5:nCuts=40:"
                       "NegWeightTreatment=IgnoreNegWeightsInTraining");
   // No UseRegulator. TMVA's Bayesian regulator inverts the Hessian of the
   // network every few epochs, and the feature sets here contain inputs that
   // are exact functions of other inputs - set 13 has sigmaRatio =
   // sigmaMin/sigmaMax and e1e2Asym = (e1-e2)/(e1+e2), set 3 has seedFrac equal
   // to the central tower fraction t11. Each is a direction the Hessian cannot
   // see, so it is singular and ROOT prints
   //   Error in <TDecompLU::InvertLU>: matrix is singular, 2 diag elements <
   //   tolerance of 2.2204e-16
   // mid-progress-bar. Training carries on without that regulator update, so
   // the result survives, but the message looks like a failure and the
   // regulator is not doing its job anyway. Overtraining is watched instead
   // through the train/test error TMVA prints at the end - they agreed to
   // 0.087 vs 0.087 on the first real sample.
   factory->BookMethod(TMVA::Types::kMLP, "MLP",
                       "!H:!V:NeuronType=tanh:NCycles=600:HiddenLayers=N+5,N:"
                       "TestRate=5:EstimatorType=CE:VarTransform=Norm");

   factory->TrainAllMethods();
   factory->TestAllMethods();
   factory->EvaluateAllMethods();

   fout->Close();
   ftmp->Close();
   printf("\nweights are in weights/%s_BDTG.weights.xml (and _MLP)\n", job.Data());
   printf("point StFcsMLCategoryMaker at one of them:\n");
   printf("  mlcat->setFeatureSet(%d);\n", featureSet);
   printf("  mlcat->setTMVAMethod(\"BDTG\");\n");
   printf("  mlcat->setWeightFile(\"weights/%s_BDTG.weights.xml\");\n", job.Data());
   delete factory;
}
