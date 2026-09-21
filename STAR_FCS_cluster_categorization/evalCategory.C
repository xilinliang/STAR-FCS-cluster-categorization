// evalCategory.C - efficiency, purity and confusion matrix of a trained model,
// with STAR's own category beside it on the same clusters.
//
//   root4star -b -q 'evalCategory.C+("feat_pico_all.root","weights/FcsCat13_BDTG.weights.xml",13)'
//
// Compile with ACLiC (the trailing '+'), as for trainTMVA.C: the features are
// computed by StFcsClusterFeatures.h, which CINT only parses approximately.
//
// Output
//   printed : confusion matrix, and per class the efficiency and purity with
//             binomial errors, for the model and for STAR's catStar
//   <out>.pdf  : four pages - confusion matrices; efficiency and purity vs
//                cluster energy; two-photon efficiency vs photon separation;
//                the model's score distributions per true class
//   <out>.root : every histogram and TEfficiency behind the pages
//
// WHAT THE NUMBERS ARE - read this before quoting one
//
// Every cluster is given ONE category: the class with the highest of the
// model's three scores. That is exactly the rule StFcsMLCategoryMaker and
// StFcsPicoCategoryMaker apply (setMode(1)), so these are the numbers the
// makers will deliver, unlike the "best efficiency x purity" in TMVA's own
// summary, which tunes a separate set of cuts for each class.
//
//   efficiency(k) = of the clusters that truly are class k,
//                   the fraction the model calls k
//   purity(k)     = of the clusters the model calls k,
//                   the fraction that truly are k
//
// Efficiency does not depend on how many clusters of each class the sample
// happens to contain. Purity does: it is diluted by whichever classes are
// abundant, and in a single-particle sample those proportions are set by how
// many gamma, pi0 and pi- events were simulated, not by physics. So a second
// purity is printed, "balanced", computed as if all three classes were equally
// common. Quote efficiencies freely; quote a purity only together with the
// class mix it assumes.
//
// HELD-OUT CLUSTERS ONLY, BY DEFAULT. sample = 1 evaluates on the test half,
// selected by the same event-level rule trainTMVA.C used to train
// (StFcsTrainTestSplit.h). That is only clean for a weight file produced by a
// trainTMVA.C that already split this way - an older one used TMVA's internal
// random split, and its "test half" here would include training clusters. If
// in doubt, re-train; it takes three minutes.
//
// STAR'S CATEGORY IS NOT THE SAME THREE CLASSES. catStar 0 means "ambiguous -
// let StFcsPointMaker try both fits", not "hadron". STAR has no hadron class
// at all. So the model and STAR are compared on classes 1 and 2 only, and the
// STAR matrix is printed with its own column meaning. STAR's ambiguous
// clusters are resolved later by the fitter's chi2, which picoDst does not
// keep usefully, so STAR's one- and two-photon efficiencies here count only
// the clusters it categorised outright: a lower bound on what the full STAR
// chain achieves.
//
// author: generated for Xilin Liang

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "TCanvas.h"
#include "TEfficiency.h"
#include "TFile.h"
#include "TGraphAsymmErrors.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TLegend.h"
#include "TLine.h"
#include "TROOT.h"
#include "TString.h"
#include "TStyle.h"
#include "TSystem.h"
#include "TTree.h"
#include "TMVA/Reader.h"

#include "StRoot/StFcsMLCategoryMaker/StFcsClusterFeatures.h"
#include "StRoot/StFcsMLCategoryMaker/StFcsTrainTestSplit.h"

namespace {

const char* kClsName[3] = {"other", "onePhoton", "twoPhoton"};
const char* kStarName[3] = {"0 ambiguous", "1 onePhoton", "2 twoPhoton"};
// How STAR's own category (the catStar branch, set by StFcsClusterMaker) is
// labelled on every figure. Change it here and all four pages follow.
const char* kRefName = "FCS Cluster";
const int kClsColor[3] = {kGray + 2, kAzure + 1, kOrange + 7};

// binomial error on a fraction pass/total
double binomErr(double pass, double total) {
   if (total <= 0) return 0.0;
   const double p = pass / total;
   return sqrt(p * (1.0 - p) / total);
}

// One confusion matrix: counts[true][assigned].
struct Confusion {
   double c[3][3];
   Confusion() {
      for (int i = 0; i < 3; i++)
         for (int j = 0; j < 3; j++) c[i][j] = 0;
   }
   double rowSum(int i) const { return c[i][0] + c[i][1] + c[i][2]; }
   double colSum(int j) const { return c[0][j] + c[1][j] + c[2][j]; }
   double eff(int k) const { return rowSum(k) > 0 ? c[k][k] / rowSum(k) : 0.0; }
   double pur(int k) const { return colSum(k) > 0 ? c[k][k] / colSum(k) : 0.0; }
   // purity as if every true class had the same number of clusters: weight
   // each row by 1/rowSum, i.e. use the row-normalised matrix
   double purBalanced(int k) const {
      double num = 0, den = 0;
      for (int t = 0; t < 3; t++) {
         if (rowSum(t) <= 0) continue;
         const double r = c[t][k] / rowSum(t);
         den += r;
         if (t == k) num = r;
      }
      return den > 0 ? num / den : 0.0;
   }
};

void printMatrix(const Confusion& m, const char* title, const char** colNames) {
   printf("\n%s\n", title);
   printf("  rows = TRUE class, columns = ASSIGNED; each row as a fraction of that true class\n");
   printf("  %-11s", "");
   for (int j = 0; j < 3; j++) printf(" %13s", colNames[j]);
   printf(" %10s\n", "clusters");
   for (int i = 0; i < 3; i++) {
      printf("  %-11s", kClsName[i]);
      const double n = m.rowSum(i);
      for (int j = 0; j < 3; j++) printf(" %13.3f", n > 0 ? m.c[i][j] / n : 0.0);
      printf(" %10.0f\n", n);
   }
}

TEfficiency* makeEff(const char* name, const char* title, int nb, const double* bins) {
   TEfficiency* e = new TEfficiency(name, title, nb, bins);
   e->SetStatisticOption(TEfficiency::kFCP);  // Clopper-Pearson, as the PDG recommends
   return e;
}

// TEfficiency::Draw in ROOT 5.34 either clears the pad or redraws an axis, so
// it cannot be overlaid cleanly. Draw its graph on a frame instead.
TGraphAsymmErrors* drawEff(TEfficiency* e, int color, int marker, const char* opt = "P") {
   TGraphAsymmErrors* g = e->CreateGraph();
   if (!g) return 0;
   g->SetLineColor(color);
   g->SetMarkerColor(color);
   g->SetMarkerStyle(marker);
   g->SetMarkerSize(1.0);
   g->Draw(opt);
   return g;
}

}  // namespace

void evalCategory(const char* infile = "feat_pico_all.root",
                  const char* weightFile = "weights/FcsCat13_BDTG.weights.xml",
                  int featureSet = 13,       // must match the weight file
                  const char* method = "BDTG",
                  int sample = 1,            // 1 = test half (held out), 0 = training half, 2 = all
                  const char* outName = "",  // default: evalFcsCat<set>_<method>
                  float eMin = 0.5,
                  float purityCut = 0.8,
                  const char* treename = "clusters") {
   using namespace StFcsClusterFeatures;
   gSystem->Load("libTMVA");
   gROOT->SetBatch(kTRUE);
   gStyle->SetOptStat(0);
   gStyle->SetPaintTextFormat("4.2f");
   // PDF page in the canvas's own 2.2:1 shape. The default paper is 20x26 cm
   // portrait, which leaves a wide canvas squeezed into one corner of the page.
   gStyle->SetPaperSize(28.0, 12.7);

   const TString out = strlen(outName) ? TString(outName) : Form("evalFcsCat%d_%s", featureSet, method);
   const char* sampleName[3] = {"TRAINING half", "TEST half (held out)", "ALL clusters"};
   if (sample < 0 || sample > 2) sample = 1;

   const int NVAR = nVar(featureSet);
   const char** names = varNames(featureSet);

   // ---------------------------------------------------------------- model
   Float_t var[kNVarMax];
   for (int i = 0; i < kNVarMax; i++) var[i] = 0;
   TMVA::Reader* reader = new TMVA::Reader("!Color:Silent");
   for (int i = 0; i < NVAR; i++) reader->AddVariable(names[i], &var[i]);
   if (!reader->BookMVA(method, weightFile)) {
      printf("could not book %s from %s - is it feature set %d?\n", method, weightFile, featureSet);
      return;
   }

   // ---------------------------------------------------------------- input
   TFile* fin = TFile::Open(infile);
   if (!fin || fin->IsZombie()) {
      printf("cannot open %s\n", infile);
      return;
   }
   TTree* in = (TTree*)fin->Get(treename);
   if (!in) {
      printf("no tree %s in %s\n", treename, infile);
      return;
   }

   const int NW = 11;
   Float_t e, x, y, sigmaMin, sigmaMax, theta, xw, yw, truthPurity, mcSepCell = -1;
   Float_t img[NW * NW], mask[NW * NW];
   Int_t run = 0, event = 0, nTowers, nNeighbor = 0, seedRow, seedCol, catStar, truthNPhoton, mcLabel = -1;
   in->SetBranchAddress("run", &run);
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
   in->SetBranchAddress("truthPurity", &truthPurity);
   if (in->GetBranch("mcLabel")) in->SetBranchAddress("mcLabel", &mcLabel);
   const bool haveSep = (in->GetBranch("mcSepCell") != 0);
   if (haveSep) in->SetBranchAddress("mcSepCell", &mcSepCell);

   // ------------------------------------------------------------ histograms
   const int nEB = 10;
   const double eBins[nEB + 1] = {0.5, 1, 2, 3, 5, 7, 10, 14, 18, 24, 32};
   const int nSB = 12;
   const double sBins[nSB + 1] = {0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 11.0};

   TEfficiency *effE_ml[3], *purE_ml[3], *effE_st[3], *purE_st[3];
   for (int k = 0; k < 3; k++) {
      effE_ml[k] = makeEff(Form("effE_ml_%s", kClsName[k]),
                           Form("model efficiency, %s;cluster E [GeV];efficiency", kClsName[k]), nEB, eBins);
      purE_ml[k] = makeEff(Form("purE_ml_%s", kClsName[k]),
                           Form("model purity, %s;cluster E [GeV];purity", kClsName[k]), nEB, eBins);
      effE_st[k] = makeEff(Form("effE_star_%s", kClsName[k]),
                           Form("%s efficiency, %s;cluster E [GeV];efficiency", kRefName, kClsName[k]), nEB, eBins);
      purE_st[k] = makeEff(Form("purE_star_%s", kClsName[k]),
                           Form("%s purity, %s;cluster E [GeV];purity", kRefName, kClsName[k]), nEB, eBins);
   }
   TEfficiency* effSep_ml = makeEff("effSep_ml", "two-photon efficiency vs separation, model;"
                                    "photon separation [towers];efficiency", nSB, sBins);
   TEfficiency* effSep_st = makeEff("effSep_star", Form("two-photon efficiency vs separation, %s;"
                                    "photon separation [towers];efficiency", kRefName), nSB, sBins);

   // model score k, split by true class
   TH1F* hScore[3][3];
   for (int k = 0; k < 3; k++)
      for (int t = 0; t < 3; t++) {
         hScore[k][t] = new TH1F(Form("hScore_%s_true_%s", kClsName[k], kClsName[t]),
                                 Form("%s score;%s score;fraction of clusters", kClsName[k], kClsName[k]),
                                 50, 0, 1);
         hScore[k][t]->SetLineColor(kClsColor[t]);
         hScore[k][t]->SetLineWidth(2);
      }

   // ------------------------------------------------------------ event loop
   Confusion ml, st;
   StFcsTrainTestSplit::EventSplitter splitter;
   const int half = NW / 2;
   float te[NW * NW];
   int trow[NW * NW], tcol[NW * NW];
   Long64_t nUsed = 0, nNoLabel = 0, nNoFeat = 0, nBadScore = 0;

   const Long64_t n = in->GetEntries();
   for (Long64_t i = 0; i < n; i++) {
      in->GetEntry(i);
      // before any selection, exactly as in trainTMVA.C, or the split differs
      const int isTest = splitter.isTest(run, event);
      if (sample == 1 && !isTest) continue;
      if (sample == 0 && isTest) continue;
      if (e < eMin) continue;

      const int truth = StFcsTrainTestSplit::trainingLabel(mcLabel, truthNPhoton, truthPurity, purityCut);
      if (truth < 0) {
         nNoLabel++;
         continue;
      }

      // rebuild the cluster's towers from image and mask, as trainTMVA.C does
      int nTow = 0;
      for (int dr = -half; dr <= half; dr++)
         for (int dc = -half; dc <= half; dc++) {
            const int pix = (dr + half) * NW + (dc + half);
            if (mask[pix] <= 0 || img[pix] <= 0) continue;
            te[nTow] = img[pix];
            trow[nTow] = seedRow + dr;
            tcol[nTow] = seedCol + dc;
            nTow++;
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
      if (compute(featureSet, c, var) != NVAR) {
         nNoFeat++;
         continue;
      }

      const std::vector<Float_t>& r = reader->EvaluateMulticlass(method);
      if (r.size() != 3) {
         nBadScore++;
         continue;
      }
      int pred = 0;
      for (int k = 1; k < 3; k++)
         if (r[k] > r[pred]) pred = k;
      const int star = (catStar >= 0 && catStar <= 2) ? catStar : 0;

      nUsed++;
      ml.c[truth][pred] += 1;
      st.c[truth][star] += 1;
      for (int k = 0; k < 3; k++) hScore[k][truth]->Fill(r[k]);

      for (int k = 0; k < 3; k++) {
         if (truth == k) effE_ml[k]->Fill(pred == k, e);
         if (pred == k) purE_ml[k]->Fill(truth == k, e);
         if (k == 0) continue;  // STAR has no "other" class
         if (truth == k) effE_st[k]->Fill(star == k, e);
         if (star == k) purE_st[k]->Fill(truth == k, e);
      }
      if (haveSep && truth == 2 && mcSepCell >= 0) {
         effSep_ml->Fill(pred == 2, mcSepCell);
         effSep_st->Fill(star == 2, mcSepCell);
      }
   }

   // ----------------------------------------------------------------- print
   printf("\n=== %s, feature set %d, %s of %s ===\n", method, featureSet, sampleName[sample], infile);
   printf("  %lld clusters evaluated (%ld events read); skipped: %lld no label, %lld no features",
          nUsed, splitter.nEvents(), nNoLabel, nNoFeat);
   if (nBadScore) printf(", %lld with a non-multiclass response", nBadScore);
   printf("\n  each cluster gets the class with the HIGHEST score - the rule the makers use\n");
   if (nUsed == 0) {
      printf("\nNothing evaluated. With sample=1 this means no test-half clusters: the run and\n"
             "event branches may be unfilled, putting every cluster in one event.\n");
      return;
   }

   printMatrix(ml, Form("--- %s ---", method), kClsName);
   printf("\n  %-11s %17s %17s %17s\n", "class", "efficiency", "purity", "purity(balanced)");
   for (int k = 0; k < 3; k++)
      printf("  %-11s    %.3f +- %.3f    %.3f +- %.3f    %.3f\n", kClsName[k], ml.eff(k),
             binomErr(ml.c[k][k], ml.rowSum(k)), ml.pur(k), binomErr(ml.c[k][k], ml.colSum(k)),
             ml.purBalanced(k));

   printMatrix(st, Form("--- %s category (catStar branch) on the same clusters ---", kRefName), kStarName);
   printf("\n  %-11s %17s %17s %17s\n", "class", "efficiency", "purity", "purity(balanced)");
   for (int k = 1; k < 3; k++)
      printf("  %-11s    %.3f +- %.3f    %.3f +- %.3f    %.3f\n", kClsName[k], st.eff(k),
             binomErr(st.c[k][k], st.rowSum(k)), st.pur(k), binomErr(st.c[k][k], st.colSum(k)),
             st.purBalanced(k));
   printf("  (STAR column 0 is 'ambiguous', resolved later by the point fitter; its efficiencies\n"
          "   count only clusters it categorised outright, so they are a lower bound)\n");

   printf("\n  class mix of this sample: other %.1f%%  onePhoton %.1f%%  twoPhoton %.1f%%\n",
          100.0 * ml.rowSum(0) / nUsed, 100.0 * ml.rowSum(1) / nUsed, 100.0 * ml.rowSum(2) / nUsed);
   printf("  purity depends on that mix; efficiency and purity(balanced) do not\n");

   // ----------------------------------------------------------------- plots
   TFile* fout = new TFile(out + ".root", "RECREATE");
   const TString pdf = out + ".pdf";
   TCanvas* cv = new TCanvas("cv", "evalCategory", 1100, 500);

   // page 1: confusion matrices, row-normalised
   TH2F* hConf[2];
   const Confusion* mats[2] = {&ml, &st};
   // TString, not Form(): Form() writes into a circular buffer that later
   // Form() calls overwrite, so its result must not be kept around
   const TString matTitle[2] = {TString::Format("%s (set %d)", method, featureSet), kRefName};
   cv->Divide(2, 1);
   for (int m = 0; m < 2; m++) {
      hConf[m] = new TH2F(Form("hConf_%d", m),
                          Form("%s: fraction of each TRUE class;assigned;true", matTitle[m].Data()), 3, 0, 3, 3, 0, 3);
      for (int t = 0; t < 3; t++) {
         const double rs = mats[m]->rowSum(t);
         for (int a = 0; a < 3; a++) hConf[m]->SetBinContent(a + 1, t + 1, rs > 0 ? mats[m]->c[t][a] / rs : 0);
         hConf[m]->GetYaxis()->SetBinLabel(t + 1, kClsName[t]);
         hConf[m]->GetXaxis()->SetBinLabel(t + 1, m == 0 ? kClsName[t] : kStarName[t]);
      }
      hConf[m]->SetMinimum(0);
      hConf[m]->SetMaximum(1);
      hConf[m]->SetMarkerSize(1.8);
      cv->cd(m + 1);
      gPad->SetLeftMargin(0.16);
      gPad->SetRightMargin(0.13);
      hConf[m]->Draw("colz text");
      hConf[m]->Write();
   }
   cv->Print(pdf + "(");

   // page 2: efficiency and purity vs energy
   cv->Clear();
   cv->Divide(2, 1);
   TLegend* leg[2];
   TEfficiency** effs[2][2] = {{effE_ml, effE_st}, {purE_ml, purE_st}};
   const char* ytitle[2] = {"efficiency", "purity"};
   for (int p = 0; p < 2; p++) {
      cv->cd(p + 1);
      // The legend lives in a band ABOVE the frame, not inside it: these
      // points cover the whole 0-1 range somewhere across the energy axis, so
      // no corner of the frame is reliably empty. The frame title would sit in
      // that band too, so the y-axis title names the panel instead.
      gPad->SetTopMargin(0.26);
      gPad->SetGridy();
      TH1F* fr = gPad->DrawFrame(eBins[0], 0, eBins[nEB], 1.05,
                                 Form(";cluster E [GeV];%s", ytitle[p]));
      (void)fr;
      leg[p] = new TLegend(0.10, 0.755, 0.95, 0.985);
      leg[p]->SetBorderSize(0);
      leg[p]->SetFillStyle(0);
      leg[p]->SetTextSize(0.042);
      leg[p]->SetNColumns(2);  // left column the model, right column FCS Cluster
      for (int k = 0; k < 3; k++) {
         TGraphAsymmErrors* g = drawEff(effs[p][0][k], kClsColor[k], 20);
         leg[p]->AddEntry(g, Form("%s, %s", kClsName[k], method), "lp");
         if (k == 0) {
            leg[p]->AddEntry((TObject*)0, "", "");  // FCS Cluster has no "other" class
            continue;
         }
         TGraphAsymmErrors* s = drawEff(effs[p][1][k], kClsColor[k], 24);
         leg[p]->AddEntry(s, Form("%s, %s", kClsName[k], kRefName), "lp");
      }
      leg[p]->Draw();
   }
   cv->Print(pdf);

   // page 3: two-photon efficiency vs photon separation - the merge transition
   cv->Clear();
   cv->cd();
   gPad->SetGridy();
   gPad->DrawFrame(sBins[0], 0, sBins[nSB], 1.05,
                   "Merged #pi^{0} clusters: fraction identified as two-photon;"
                   "separation of the two photons at the ECal [towers];efficiency");
   TLegend* lsep = new TLegend(0.55, 0.15, 0.88, 0.32);
   lsep->SetBorderSize(0);
   lsep->SetFillStyle(0);
   TGraphAsymmErrors* gs1 = drawEff(effSep_ml, kClsColor[2], 20);
   TGraphAsymmErrors* gs2 = drawEff(effSep_st, kClsColor[2], 24);
   if (gs1) lsep->AddEntry(gs1, method, "lp");
   if (gs2) lsep->AddEntry(gs2, Form("%s (category 2)", kRefName), "lp");
   lsep->Draw();
   if (!haveSep) printf("  (no mcSepCell branch in %s - page 3 is empty)\n", infile);
   cv->Print(pdf);

   // page 4: score distributions per true class
   cv->Clear();
   cv->Divide(3, 1);
   for (int k = 0; k < 3; k++) {
      cv->cd(k + 1);
      gPad->SetLogy();
      double ymax = 0;
      for (int t = 0; t < 3; t++) {
         const double s = hScore[k][t]->Integral();
         if (s > 0) hScore[k][t]->Scale(1.0 / s);
         if (hScore[k][t]->GetMaximum() > ymax) ymax = hScore[k][t]->GetMaximum();
      }
      for (int t = 0; t < 3; t++) {
         hScore[k][t]->SetMaximum(ymax * 3);
         hScore[k][t]->SetMinimum(1e-4);
         hScore[k][t]->Draw(t == 0 ? "hist" : "hist same");
      }
      if (k == 0) {
         TLegend* ls = new TLegend(0.2, 0.72, 0.6, 0.88);
         ls->SetBorderSize(0);
         ls->SetFillStyle(0);
         for (int t = 0; t < 3; t++) ls->AddEntry(hScore[k][t], Form("true %s", kClsName[t]), "l");
         ls->Draw();
      }
   }
   cv->Print(pdf + ")");

   // everything behind the plots
   fout->cd();
   for (int k = 0; k < 3; k++) {
      effE_ml[k]->Write();
      purE_ml[k]->Write();
      if (k > 0) {
         effE_st[k]->Write();
         purE_st[k]->Write();
      }
      for (int t = 0; t < 3; t++) hScore[k][t]->Write();
   }
   effSep_ml->Write();
   effSep_st->Write();
   fout->Close();
   printf("\nwrote %s and %s.root\n", pdf.Data(), out.Data());
}
