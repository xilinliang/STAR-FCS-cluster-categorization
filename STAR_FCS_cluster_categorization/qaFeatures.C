// qaFeatures.C - quality-assurance plots for every input variable of feature
// sets 3 and 13, from ONE feature file (the `clusters` tree written by
// runPicoDst_ml.C or runMudst_ml.C in mode 0).
//
//   root4star -b -q 'qaFeatures.C+("feat_pi0.e30.root","qa_pi0.e30")'
//
// or let runQA.sh do both steps, straight from a picoDst or MuDst:
//
//   ./runQA.sh pi0.e30.vz0.all.MuDst.root
//
// Compile with ACLiC (the trailing '+'), as for trainTMVA.C and
// evalCategory.C. The variables are computed by StFcsClusterFeatures.h - the
// same code the training and the category makers use - from the tree's image
// and mask, exactly as trainTMVA.C does. So what you see here is what TMVA is
// fed, not a separate re-implementation.
//
// Output
//   <out>.pdf
//     page 1        cluster overview: energy, generated energy, centroid maps
//                   (north, south), nTowers, nNeighbor, FCS Cluster category,
//                   true class
//     gun pages     everything split by gun particle - gamma, pi0, pi-.
//                   First page: generated energy, one entry per CLUSTER (top
//                   row, the spectrum the training saw) and one per EVENT
//                   (bottom row, the spectrum as thrown); the difference
//                   between the rows is cluster multiplicity.
//                   Second page: CLUSTER energy, and below it the containment
//                   E_cluster / E_gen - near 1 for gamma, far below for pi-,
//                   which is why a spiky generated spectrum does not make a
//                   spiky spectrum of the variable the model actually sees.
//                   Third page: the three species overlaid, all three
//                   normalisations side by side.
//     leak page     cluster centroid per gun particle, the check that must pass
//                   before feature set 4 (position) can be believed
//     next page     sigmaMax vs energy with STAR's category boundaries drawn on
//     set 3 pages   the linear correlation matrix of the inputs, then every
//                   variable split by true class (unit area), then every
//                   variable against cluster energy (all clusters)
//     set 13 pages  the same for set 13
//   <out>.root  every histogram
//   printed     the per-gun-particle tables (clusters, events, clusters per
//               event, mean generated energy; and the centroid means and RMS),
//               then per set, the correlation matrix (with the strongest pair
//               flagged when it exceeds 95 %), and a table per variable:
//               entries, non-finite values,
//               min / max / mean / rms, the fraction sitting at the most
//               common single value (a constant or near-constant variable
//               is what makes TMVA abort with "Variable ... is constant"),
//               and the mean per true class
//
// True class = mcLabel, the same label trainTMVA.C trains on: other (no
// photon inside the cluster), onePhoton, twoPhoton. Clusters without MC truth
// (real data) are shown as "no truth"; every panel also has all clusters in
// black dashes, so the macro is equally useful on data.
//
// ENERGY AXIS. Every energy axis ends at the gun energy of the sample, taken
// from the file name: pi0.e60.vz0.all.picoDst.root, or the feature file made
// from it, gives 60 GeV. With no such token in the name the largest energy in
// the file is used instead, rounded up. Pass eMax to set it by hand.
//
// author: generated for Xilin Liang

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <vector>

#include "TCanvas.h"
#include "TFile.h"
#include "TGraph.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TLatex.h"
#include "TLegend.h"
#include "TROOT.h"
#include "TString.h"
#include "TStyle.h"
#include "TTree.h"

#include "StRoot/StFcsMLCategoryMaker/StFcsClusterFeatures.h"
#include "StRoot/StFcsMLCategoryMaker/StFcsTrainTestSplit.h"

namespace {

// class slots: 0 other, 1 onePhoton, 2 twoPhoton, 3 no truth
const int kNCls = 4;
const char* kQaClsName[kNCls] = {"other", "onePhoton", "twoPhoton", "no truth"};
const int kQaClsColor[kNCls] = {kGray + 2, kAzure + 1, kOrange + 7, kMagenta + 1};
// gun particles: gamma, pi0, pi-
const int kQaGunColor[3] = {kAzure + 1, kOrange + 7, kGray + 2};

bool bad(float v) { return v != v || fabs(v) > 1e30; }

// The gun energy of a single-particle file, read from its name: the token
// ".e<number>" or "_e<number>", as in pi0.e60.vz0.all.picoDst.root -> 60.
// Returns 0 when the name has no such token (real data, a mixed sample, or a
// name written some other way), and then the energy axis is set from the file
// itself instead.
double gunEnergyFromName(const char* name) {
   if (!name) return 0;
   const char* s = name;
   for (const char* p = name; *p; p++) {
      if ((*p != '.' && *p != '_' && *p != '/') || (p[1] != 'e' && p[1] != 'E')) continue;
      const char* d = p + 2;
      if (*d < '0' || *d > '9') continue;
      const double v = atof(d);
      // the token has to END here, so that ".eta05" or ".energy" is not read
      // as a number
      while (*d >= '0' && *d <= '9') d++;
      if (*d == '.' || *d == '_' || *d == 0) {
         if (v > 0 && v < 1e4) return v;
      }
   }
   (void)s;
   return 0;
}

// round up to a round number, so the axis ends at 30 or 60 rather than 58.7
double niceCeil(double v) {
   if (v <= 0) return 1;
   const double step = (v <= 20) ? 2 : (v <= 50) ? 5 : 10;
   return step * ceil(v / step);
}

// 0.5 - 99.5 % quantile range of the finite values, padded by 3 %, so one
// outlier cannot squeeze the distribution into a single bin
void quantileRange(const std::vector<float>& v, double& lo, double& hi) {
   std::vector<float> s;
   s.reserve(v.size());
   for (size_t i = 0; i < v.size(); i++)
      if (!bad(v[i])) s.push_back(v[i]);
   if (s.empty()) {
      lo = 0;
      hi = 1;
      return;
   }
   std::sort(s.begin(), s.end());
   lo = s[(size_t)(0.005 * (s.size() - 1))];
   hi = s[(size_t)(0.995 * (s.size() - 1))];
   if (hi <= lo) {  // (near) constant: open a window around it
      lo -= 0.5;
      hi += 0.5;
   }
   const double pad = 0.03 * (hi - lo);
   lo -= pad;
   hi += pad;
}

// unit-area overlay of h[0..n-1]; returns nothing, draws on the current pad
void drawOverlay(TH1F** h, int n) {
   double ymax = 0;
   for (int i = 0; i < n; i++) {
      if (!h[i]) continue;
      const double s = h[i]->Integral();
      if (s > 0) h[i]->Scale(1.0 / s);
      if (h[i]->GetMaximum() > ymax) ymax = h[i]->GetMaximum();
   }
   bool first = true;
   for (int i = 0; i < n; i++) {
      if (!h[i]) continue;
      h[i]->SetMinimum(0);
      h[i]->SetMaximum(1.25 * ymax);
      h[i]->Draw(first ? "hist" : "hist same");
      first = false;
   }
}

}  // namespace

void qaFeatures(const char* infile = "feat_pico.root",
                const char* outName = "",    // default: qa_<input file name without .root>
                float eMin = 0.5,            // same cut as trainTMVA.C / evalCategory.C
                float eMax = 0,              // upper end of every energy axis; 0 = automatic
                const char* treename = "clusters") {
   using namespace StFcsClusterFeatures;
   gROOT->SetBatch(kTRUE);
   gStyle->SetOptStat(0);
   gStyle->SetPaperSize(28.0, 12.7);  // PDF page in the canvas's own shape

   TString out(outName);
   if (out.Length() == 0) {
      out = infile;
      out.ReplaceAll(".root", "");
      const Ssiz_t slash = out.Last('/');
      if (slash >= 0) out.Remove(0, slash + 1);
      out.Prepend("qa_");
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
   Float_t e, x, y, sigmaMin, sigmaMax, theta, xw, yw, truthPurity = 0;
   Float_t img[NW * NW], mask[NW * NW];
   Int_t det = 0, nTowers, nNeighbor = 0, seedRow, seedCol, catStar = 0, truthNPhoton = -1, mcLabel = -1;
   Float_t genE = 0;
   Int_t genPid = 0;
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
   if (in->GetBranch("det")) in->SetBranchAddress("det", &det);
   if (in->GetBranch("catStar")) in->SetBranchAddress("catStar", &catStar);
   if (in->GetBranch("truthNPhoton")) in->SetBranchAddress("truthNPhoton", &truthNPhoton);
   if (in->GetBranch("truthPurity")) in->SetBranchAddress("truthPurity", &truthPurity);
   if (in->GetBranch("mcLabel")) in->SetBranchAddress("mcLabel", &mcLabel);
   const bool haveGen = (in->GetBranch("genE") != 0);
   if (haveGen) in->SetBranchAddress("genE", &genE);
   const bool havePid = (in->GetBranch("genPid") != 0);
   if (havePid) in->SetBranchAddress("genPid", &genPid);
   // run and event are needed only to count each event once for the per-EVENT
   // generated-energy page; the per-cluster pages do not use them.
   Int_t run = 0, event = 0;
   const bool haveEvt = (in->GetBranch("run") != 0 && in->GetBranch("event") != 0);
   if (haveEvt) {
      in->SetBranchAddress("run", &run);
      in->SetBranchAddress("event", &event);
   }

   // ------------------------------------------------------- energy axis
   //
   // A 30 GeV gun drawn on a 0-60 axis wastes half the page, and a 60 GeV gun
   // on a 0-32 axis loses its top half altogether. So the upper end follows the
   // sample: the gun energy in the file name when it has one (pi0.e60... -> 60),
   // otherwise the largest energy actually in the file, rounded up.
   double eTop = eMax;
   const char* eTopFrom = "the eMax argument";
   if (eTop <= 0) {
      eTop = gunEnergyFromName(infile);
      eTopFrom = "the gun energy in the file name";
   }
   if (eTop <= 0) {
      double m = in->GetMaximum("e");
      if (haveGen) {
         const double g = in->GetMaximum("genE");
         if (g > m) m = g;
      }
      eTop = niceCeil(1.05 * m);
      eTopFrom = "the largest energy in the file";
   }
   if (eTop <= eMin) eTop = niceCeil(2 * eMin);
   printf("energy axes run 0 - %.3g GeV, from %s\n", eTop, eTopFrom);
   const int nEbin = 64;

   // ---------------------------------------------------- the two feature sets
   const int nSet = 2;
   const int sets[nSet] = {3, 13};
   std::vector<float> val[nSet][kNVarMax];  // [set][variable][cluster]
   std::vector<int> cls;                    // class slot of each stored cluster
   std::vector<float> eClu;                 // cluster energy of each stored cluster

   // ------------------------------------------------------ overview histos
   TH1F* hE = new TH1F("hE", "cluster energy;E [GeV];clusters", nEbin, 0, eTop);
   TH1F* hGenE = new TH1F("hGenE", "generated (gun) energy, per cluster;E_{gen} [GeV];clusters", nEbin, 0, eTop);
   TH2F* hXY[2];
   for (int d = 0; d < 2; d++)
      hXY[d] = new TH2F(Form("hXY_det%d", d), Form("centroid, det %d (%s);x [column];y [row]", d, d == 0 ? "north" : "south"),
                        24, 0, 24, 36, 0, 36);
   // Centroid per GUN PARTICLE. Feature set 4 adds the cluster position to the
   // inputs, which is only legitimate if every species illuminates the same
   // part of the detector: if the gamma, pi0 and pi- guns point at slightly
   // different patches, a model given x and y can read the class off the
   // position instead of the shower shape, and the gain evaporates on data.
   // These three maps, and the table printed with them, are that check.
   const int kNG = 3;  // gamma, pi0, pi-
   const char* gName[kNG] = {"gamma", "pi0", "pi-"};
   TH2F* hXYg[kNG];
   for (int g = 0; g < kNG; g++)
      hXYg[g] = new TH2F(Form("hXY_%s", gName[g]), Form("centroid, %s gun;x [column];y [row]", gName[g]),
                         24, 0, 24, 36, 0, 36);
   double gN[kNG], gX[kNG], gY[kNG], gXX[kNG], gYY[kNG];
   for (int g = 0; g < kNG; g++) gN[g] = gX[g] = gY[g] = gXX[g] = gYY[g] = 0;

   // Generated (gun) energy split by gun particle, twice over.
   //
   //   hGenEg[g]  one entry per CLUSTER. genE is an event quantity copied onto
   //              every cluster of the event, so an event making two clusters
   //              counts twice. This is the spectrum the TRAINING sees, because
   //              training is per cluster - which is why it is the one to look
   //              at when asking what the model was fed.
   //   hGenEvt[g] one entry per EVENT. The generated spectrum as thrown, with
   //              the cluster-multiplicity weighting removed. Dividing the two
   //              gives clusters per event against energy.
   //
   // The two differ by more than a constant: pi- makes 1.6 clusters per event
   // and gamma 1.0, and the ratio itself grows with energy, so the per-cluster
   // spectrum is tilted towards pi- and towards high energy relative to what
   // was actually generated.
   TH1F* hGenEg[kNG];
   TH1F* hGenEvt[kNG];
   for (int g = 0; g < kNG; g++) {
      hGenEg[g] = new TH1F(Form("hGenEg_%s", gName[g]),
                           Form("generated energy, %s gun;E_{gen} [GeV];clusters", gName[g]),
                           nEbin, 0, eTop);
      hGenEvt[g] = new TH1F(Form("hGenEvt_%s", gName[g]),
                            Form("generated energy, %s gun;E_{gen} [GeV];events", gName[g]),
                            nEbin, 0, eTop);
      hGenEg[g]->SetLineColor(kQaGunColor[g]);
      hGenEvt[g]->SetLineColor(kQaGunColor[g]);
   }
   long gEvt[kNG];
   for (int g = 0; g < kNG; g++) gEvt[g] = 0;

   // CLUSTER energy split by gun particle, and the containment fraction
   // E_cluster / E_gen beside it. The fraction is the reason the two spectra
   // do not look alike: a gamma puts nearly all of its energy into one cluster
   // and keeps the gun's shape, while a pi- deposits a random fraction and
   // smears a fixed gun energy across the whole range. So a spiky GENERATED
   // spectrum does not imply a spiky spectrum of the variable the model
   // actually sees.
   TH1F* hEg[kNG];
   TH1F* hFrac[kNG];
   for (int g = 0; g < kNG; g++) {
      hEg[g] = new TH1F(Form("hEg_%s", gName[g]),
                        Form("cluster energy, %s gun;E [GeV];clusters", gName[g]),
                        nEbin, 0, eTop);
      hFrac[g] = new TH1F(Form("hFrac_%s", gName[g]),
                          Form("E_{cluster} / E_{gen}, %s gun;E / E_{gen};clusters", gName[g]),
                          60, 0, 1.5);
      hEg[g]->SetLineColor(kQaGunColor[g]);
      hFrac[g]->SetLineColor(kQaGunColor[g]);
   }

   TH1F* hNTow = new TH1F("hNTow", "towers per cluster;nTowers;clusters", 40, 0.5, 40.5);
   TH1F* hNNb = new TH1F("hNNb", "neighbouring clusters;nNeighbor;clusters", 6, -0.5, 5.5);
   TH1F* hCat = new TH1F("hCat", "FCS Cluster category;category;clusters", 3, -0.5, 2.5);
   TH1F* hCls = new TH1F("hCls", "true class (mcLabel);;clusters", kNCls, -0.5, kNCls - 0.5);
   TH2F* hSigE = new TH2F("hSigE", "sigmaMax vs E, nTowers #geq 5;E [GeV];#sigma_{max} [cells]", nEbin, 0, eTop, 60, 0, 3);

   // ---------------------------------------------------------- event loop
   const int half = NW / 2;
   float te[NW * NW], var[kNVarMax];
   int trow[NW * NW], tcol[NW * NW];
   Long64_t nRead = 0, nLowE = 0, nNoFeat[nSet] = {0, 0};
   const Long64_t n = in->GetEntries();
   int prevRun = -1, prevEvent = -1;
   bool firstEntry = true;
   for (Long64_t i = 0; i < n; i++) {
      in->GetEntry(i);
      nRead++;

      // One entry per EVENT, counted before any cluster cut, so this page shows
      // the generated spectrum of the sample rather than the spectrum of the
      // clusters that survived. Clusters of one event are consecutive in the
      // tree (the dumpers fill event by event), so a change of (run, event) is
      // a new event. Events that produced NO cluster entry at all never reach
      // the tree and cannot be counted here.
      if (haveGen && havePid && haveEvt) {
         if (firstEntry || run != prevRun || event != prevEvent) {
            const int ge = (genPid == 1) ? 0 : (genPid == 7) ? 1 : (genPid == 9) ? 2 : -1;
            if (ge >= 0) {
               hGenEvt[ge]->Fill(genE);
               gEvt[ge]++;
            }
            prevRun = run;
            prevEvent = event;
            firstEntry = false;
         }
      }

      if (e < eMin) {
         nLowE++;
         continue;
      }
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
      StFcsClusterFeatures::ClusterInput c;
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

      // both sets or neither, so the two sets are drawn from the same clusters
      float v2[nSet][kNVarMax];
      bool ok = true;
      for (int s = 0; s < nSet; s++) {
         if (compute(sets[s], c, var) != nVar(sets[s])) {
            nNoFeat[s]++;
            ok = false;
            continue;
         }
         for (int k = 0; k < nVar(sets[s]); k++) v2[s][k] = var[k];
      }
      if (!ok) continue;

      int t = StFcsTrainTestSplit::trainingLabel(mcLabel, truthNPhoton, truthPurity, 0.8f);
      if (t < 0) t = 3;
      cls.push_back(t);
      eClu.push_back(e);
      for (int s = 0; s < nSet; s++)
         for (int k = 0; k < nVar(sets[s]); k++) val[s][k].push_back(v2[s][k]);

      hE->Fill(e);
      if (haveGen) hGenE->Fill(genE);
      if (det == 0 || det == 1) hXY[det]->Fill(x, y);
      if (havePid) {
         const int g = (genPid == 1) ? 0 : (genPid == 7) ? 1 : (genPid == 9) ? 2 : -1;
         if (g >= 0) {
            hEg[g]->Fill(e);
            if (haveGen) {
               hGenEg[g]->Fill(genE);
               if (genE > 0) hFrac[g]->Fill(e / genE);
            }
            hXYg[g]->Fill(x, y);
            gN[g] += 1;
            gX[g] += x;
            gY[g] += y;
            gXX[g] += (double)x * x;
            gYY[g] += (double)y * y;
         }
      }
      hNTow->Fill(nTowers);
      hNNb->Fill(nNeighbor);
      hCat->Fill(catStar);
      hCls->Fill(t);
      if (nTowers >= 5) hSigE->Fill(e, sigmaMax);
   }
   const int nClu = (int)cls.size();
   printf("\n=== qaFeatures: %s ===\n", infile);
   printf("  %lld clusters read, %lld below %.2f GeV, %d with both feature sets computed", nRead, nLowE, eMin, nClu);
   for (int s = 0; s < nSet; s++)
      if (nNoFeat[s]) printf(", %lld without set %d", nNoFeat[s], sets[s]);
   printf("\n");
   double nCls[kNCls] = {0, 0, 0, 0};
   for (int i = 0; i < nClu; i++) nCls[cls[i]] += 1;
   printf("  true class:");
   for (int k = 0; k < kNCls; k++) printf("  %s %.0f", kQaClsName[k], nCls[k]);
   printf("\n");
   if (nClu == 0) {
      printf("  nothing to plot\n");
      return;
   }

   TFile* fout = new TFile(out + ".root", "RECREATE");
   const TString pdf = out + ".pdf";
   TCanvas* cv = new TCanvas("cv", "qaFeatures", 1100, 500);
   std::vector<TH1*> keep;

   // ------------------------------------------------ page 1: overview
   cv->Divide(4, 2);
   TH1* ov[8] = {hE, hGenE, hXY[0], hXY[1], hNTow, hNNb, hCat, hCls};
   for (int k = 0; k < kNCls; k++) hCls->GetXaxis()->SetBinLabel(k + 1, kQaClsName[k]);
   hCat->GetXaxis()->SetBinLabel(1, "0 ambiguous");
   hCat->GetXaxis()->SetBinLabel(2, "1 onePhoton");
   hCat->GetXaxis()->SetBinLabel(3, "2 twoPhoton");
   for (int p = 0; p < 8; p++) {
      cv->cd(p + 1);
      gPad->SetLeftMargin(0.15);
      if (p == 2 || p == 3) {
         gPad->SetRightMargin(0.14);
         ov[p]->Draw("colz");
      } else {
         ov[p]->SetLineWidth(2);
         ov[p]->Draw("hist");
      }
      keep.push_back(ov[p]);
   }
   if (!haveGen) {
      cv->cd(2);
      TLatex nt;
      nt.SetNDC();
      nt.SetTextSize(0.07);
      nt.DrawLatexNDC(0.25, 0.5, "no genE branch in this file");
   }
   cv->Print(pdf + "(");

   // ---- generated energy per gun particle, per cluster and per event ------
   //
   // Two rows of three. Top: one entry per cluster, which is the spectrum the
   // training actually saw. Bottom: one entry per event, the spectrum as
   // thrown. The printed table gives the ratio, which is clusters per event.
   if (haveGen && havePid && (gN[0] + gN[1] + gN[2]) > 0) {
      printf("\n  generated (gun) energy per gun particle\n");
      printf("  %-8s %10s %9s %9s %12s %12s %11s %10s\n", "gun", "clusters", "events",
             "clu/event", "<Egen> clu", "<Egen> evt", "<E clu>", "<E/Egen>");
      for (int g = 0; g < kNG; g++) {
         if (gN[g] <= 0) continue;
         const double ne = (double)gEvt[g];
         char cpe[32], mev[32];
         if (ne > 0) {
            snprintf(cpe, sizeof(cpe), "%.3f", gN[g] / ne);
            snprintf(mev, sizeof(mev), "%.2f", hGenEvt[g]->GetMean());
         } else {
            snprintf(cpe, sizeof(cpe), "-");
            snprintf(mev, sizeof(mev), "-");
         }
         printf("  %-8s %10.0f %9ld %9s %12.2f %12s %11.2f %10.3f\n", gName[g], gN[g], gEvt[g],
                cpe, hGenEg[g]->GetMean(), mev, hEg[g]->GetMean(), hFrac[g]->GetMean());
      }
      if (!haveEvt)
         printf("  (no run/event branch in this file - the per-event row is empty)\n");
      printf("  clu/event above 1 is why the per-cluster spectrum is not the generated one:\n"
             "  a pi- event contributes more entries than a gamma event of the same energy.\n"
             "  <E/Egen> is the containment: near 1 for gamma, far below for pi-, which is why\n"
             "  a spiky GENERATED spectrum does not make a spiky CLUSTER spectrum for hadrons\n");

      cv->Clear();
      cv->Divide(3, 2);
      for (int g = 0; g < kNG; g++) {
         cv->cd(g + 1);
         gPad->SetLeftMargin(0.15);
         hGenEg[g]->SetLineWidth(2);
         hGenEg[g]->Draw("hist");
         keep.push_back(hGenEg[g]);

         cv->cd(g + 4);
         gPad->SetLeftMargin(0.15);
         hGenEvt[g]->SetLineWidth(2);
         hGenEvt[g]->Draw("hist");
         keep.push_back(hGenEvt[g]);
         if (!haveEvt) {
            TLatex nb;
            nb.SetNDC();
            nb.SetTextSize(0.055);
            nb.DrawLatexNDC(0.18, 0.5, "no run/event branch");
         }
      }
      cv->Print(pdf);

      // cluster energy per gun particle, with the containment fraction below it
      cv->Clear();
      cv->Divide(3, 2);
      for (int g = 0; g < kNG; g++) {
         cv->cd(g + 1);
         gPad->SetLeftMargin(0.15);
         hEg[g]->SetLineWidth(2);
         hEg[g]->Draw("hist");
         keep.push_back(hEg[g]);

         cv->cd(g + 4);
         gPad->SetLeftMargin(0.15);
         hFrac[g]->SetLineWidth(2);
         hFrac[g]->Draw("hist");
         keep.push_back(hFrac[g]);
      }
      cv->Print(pdf);

      // the three species over one another, all three normalisations
      cv->Clear();
      cv->Divide(3, 1);
      for (int pane = 0; pane < 3; pane++) {
         cv->cd(pane + 1);
         gPad->SetLeftMargin(0.13);
         TLegend* lg = new TLegend(0.62, 0.70, 0.90, 0.89);
         lg->SetBorderSize(0);
         lg->SetFillStyle(0);
         const char* ovTitle[3] = {"cluster energy;E [GeV];clusters",
                                   "generated energy, per cluster;E_{gen} [GeV];clusters",
                                   "generated energy, per event;E_{gen} [GeV];events"};
         double top = 0;
         for (int g = 0; g < kNG; g++) {
            TH1F* h = (pane == 0) ? hEg[g] : (pane == 1) ? hGenEg[g] : hGenEvt[g];
            if (h->GetMaximum() > top) top = h->GetMaximum();
         }
         for (int g = 0; g < kNG; g++) {
            TH1F* h = (pane == 0) ? hEg[g] : (pane == 1) ? hGenEg[g] : hGenEvt[g];
            h->SetTitle(ovTitle[pane]);
            h->SetMaximum(1.15 * top);
            h->SetLineWidth(2);
            h->Draw(g == 0 ? "hist" : "hist same");
            lg->AddEntry(h, gName[g], "l");
         }
         lg->Draw();
      }
      cv->Print(pdf);
   }

   // ---- centroid per gun particle: the leakage check for feature set 4 ----
   if (havePid && (gN[0] + gN[1] + gN[2]) > 0) {
      printf("\n  cluster centroid per gun particle - the three species must illuminate the\n"
             "  same region, or a model given x and y (feature set 4) can read the class\n"
             "  off the position instead of the shower shape\n");
      printf("  %-8s %10s %9s %9s %9s %9s\n", "gun", "clusters", "<x>", "rms x", "<y>", "rms y");
      for (int g = 0; g < kNG; g++) {
         if (gN[g] <= 0) continue;
         const double mx = gX[g] / gN[g], my = gY[g] / gN[g];
         printf("  %-8s %10.0f %9.2f %9.2f %9.2f %9.2f\n", gName[g], gN[g], mx,
                sqrt(fabs(gXX[g] / gN[g] - mx * mx)), my, sqrt(fabs(gYY[g] / gN[g] - my * my)));
      }
      printf("  means differing by more than ~0.1 cell are a warning, not a verdict:\n"
             "  look at the three maps on the next page before adding position as an input\n");
      cv->Clear();
      cv->Divide(3, 1);
      for (int g = 0; g < kNG; g++) {
         cv->cd(g + 1);
         gPad->SetLeftMargin(0.15);
         gPad->SetRightMargin(0.14);
         gPad->SetLogz(0);
         hXYg[g]->Draw("colz");
         keep.push_back(hXYg[g]);
      }
      cv->Print(pdf);
   }

   // ------------------------------------------ page 2: sigmaMax vs E + STAR cuts
   cv->Clear();
   cv->cd();
   gPad->SetRightMargin(0.12);
   gPad->SetLogz();
   hSigE->Draw("colz");
   keep.push_back(hSigE);
   // StFcsClusterMaker::categorization(), nTowers >= 5:
   //   sigmaMax > 1/2.5 + 0.003 E + 7/E  -> 2 (two photons)
   //   sigmaMax < 1/2.1 - 0.001 E + 2/E  -> 1 (one photon), otherwise 0
   const int nPt = 60;
   double ex[nPt], cut2[nPt], cut1[nPt];
   for (int j = 0; j < nPt; j++) {
      ex[j] = 1.0 + (eTop - 1.0) * j / (nPt - 1);
      cut2[j] = 1.0 / 2.5 + 0.003 * ex[j] + 7.0 / ex[j];
      cut1[j] = 1.0 / 2.1 - 0.001 * ex[j] + 2.0 / ex[j];
   }
   TGraph* g2 = new TGraph(nPt, ex, cut2);
   TGraph* g1 = new TGraph(nPt, ex, cut1);
   g2->SetLineColor(kOrange + 7);
   g1->SetLineColor(kAzure + 1);
   g2->SetLineWidth(3);
   g1->SetLineWidth(3);
   g2->Draw("L");
   g1->Draw("L");
   TLegend* lc = new TLegend(0.45, 0.72, 0.87, 0.88);
   lc->SetFillStyle(0);
   lc->SetBorderSize(0);
   lc->AddEntry(g2, "above: FCS Cluster category 2", "l");
   lc->AddEntry(g1, "below: FCS Cluster category 1", "l");
   lc->Draw();
   cv->Print(pdf);

   // ------------------------------------------------------- per set
   for (int s = 0; s < nSet; s++) {
      const int set = sets[s];
      const int nv = nVar(set);
      const char** names = varNames(set);

      // ---- printed table ----
      printf("\n--- feature set %d: %d variables, %d clusters ---\n", set, nv, nClu);
      printf("  %-11s %8s %10s %10s %10s %10s %8s |", "variable", "nonfin", "min", "max", "mean", "rms", "top1%");
      for (int k = 0; k < kNCls; k++)
         if (nCls[k] > 0) printf(" %10s", kQaClsName[k]);
      printf("   (mean per true class)\n");
      for (int v = 0; v < nv; v++) {
         const std::vector<float>& a = val[s][v];
         long nBad = 0;
         double mn = 1e300, mx = -1e300, sum = 0, sum2 = 0, sumC[kNCls] = {0, 0, 0, 0}, nC[kNCls] = {0, 0, 0, 0};
         std::map<float, long> freq;
         for (int i = 0; i < nClu; i++) {
            if (bad(a[i])) {
               nBad++;
               continue;
            }
            mn = std::min(mn, (double)a[i]);
            mx = std::max(mx, (double)a[i]);
            sum += a[i];
            sum2 += a[i] * a[i];
            sumC[cls[i]] += a[i];
            nC[cls[i]] += 1;
            freq[a[i]]++;
         }
         const double ng = nClu - nBad;
         const double mean = ng > 0 ? sum / ng : 0;
         const double rms = ng > 0 ? sqrt(std::max(0.0, sum2 / ng - mean * mean)) : 0;
         long top = 0;
         for (std::map<float, long>::const_iterator it = freq.begin(); it != freq.end(); ++it)
            if (it->second > top) top = it->second;
         const double topFrac = ng > 0 ? 100.0 * top / ng : 0;
         printf("  %-11s %8ld %10.4g %10.4g %10.4g %10.4g %7.1f%% |", names[v], nBad, ng > 0 ? mn : 0, ng > 0 ? mx : 0, mean,
                rms, topFrac);
         for (int k = 0; k < kNCls; k++)
            if (nCls[k] > 0) printf(" %10.4g", nC[k] > 0 ? sumC[k] / nC[k] : 0.0);
         if (nBad > 0) printf("  <- NON-FINITE VALUES");
         if (rms == 0) printf("  <- CONSTANT: TMVA will abort");
         else if (topFrac > 95) printf("  <- nearly constant");
         printf("\n");
      }

      // ---- linear correlation between the inputs ----
      //
      // Two inputs correlated above ~0.95 carry one piece of information twice.
      // That is not fatal - a BDT copes - but it makes the TMVA variable
      // ranking meaningless and it is what gives an MLP a singular Hessian. In
      // set 13, sigX/sigY/sigXY are the same covariance matrix as
      // sigmaMax/sigmaMin/theta in another frame, so this table is the evidence
      // for or against training set 10 instead. TMVA prints the same matrix
      // when it trains; this is the cheap way to see it without training.
      {
         std::vector<double> mean(nv, 0.0), sd(nv, 0.0);
         std::vector<long> ngood(nv, 0);
         for (int a = 0; a < nv; a++) {
            double s1 = 0, s2 = 0;
            long ng = 0;
            for (int i = 0; i < nClu; i++) {
               const float x1 = val[s][a][i];
               if (bad(x1)) continue;
               s1 += x1;
               s2 += (double)x1 * x1;
               ng++;
            }
            ngood[a] = ng;
            mean[a] = ng > 0 ? s1 / ng : 0;
            sd[a] = ng > 0 ? sqrt(std::max(0.0, s2 / ng - mean[a] * mean[a])) : 0;
         }
         TH2F* hc = new TH2F(Form("hCorr_set%d", set), Form("set %d: linear correlation [%%];;", set), nv, 0,
                             nv, nv, 0, nv);
         printf("\n  linear correlation between the set %d inputs [%%]\n  %-11s", set, "");
         for (int a = 0; a < nv; a++) printf(" %6.6s", names[a]);
         printf("\n");
         double worst = 0;
         int wa = 0, wb = 0;
         for (int a = 0; a < nv; a++) {
            printf("  %-11s", names[a]);
            for (int b2 = 0; b2 < nv; b2++) {
               double cov = 0;
               long ng = 0;
               for (int i = 0; i < nClu; i++) {
                  const float x1 = val[s][a][i], x2 = val[s][b2][i];
                  if (bad(x1) || bad(x2)) continue;
                  cov += ((double)x1 - mean[a]) * ((double)x2 - mean[b2]);
                  ng++;
               }
               const double den = (ng > 0 && sd[a] > 0 && sd[b2] > 0) ? (ng * sd[a] * sd[b2]) : 0;
               const double r = (den > 0) ? cov / den : (a == b2 ? 1.0 : 0.0);
               printf(" %6.0f", 100 * r);
               hc->SetBinContent(a + 1, b2 + 1, 100 * r);
               if (a != b2 && fabs(r) > worst) {
                  worst = fabs(r);
                  wa = a;
                  wb = b2;
               }
            }
            printf("\n");
            hc->GetXaxis()->SetBinLabel(a + 1, names[a]);
            hc->GetYaxis()->SetBinLabel(a + 1, names[a]);
         }
         printf("  strongest pair: %s / %s at %.0f %%", names[wa], names[wb], 100 * worst);
         if (worst > 0.95)
            printf("  <- one piece of information twice\n");
         else
            printf("\n");
         hc->SetMinimum(-100);
         hc->SetMaximum(100);
         hc->SetMarkerSize(1.2);
         hc->GetXaxis()->SetLabelSize(0.035);
         hc->GetYaxis()->SetLabelSize(0.035);
         cv->Clear();
         cv->cd();
         // The canvas still carries SetLogz from the sigmaMax-vs-E page, and a
         // log z axis silently drops every NEGATIVE correlation - which is half
         // of what this matrix is for. Turn it off explicitly, and print the
         // numbers as integers rather than with six digits.
         gPad->SetLogz(0);
         gStyle->SetPaintTextFormat("4.0f");
         gPad->SetLeftMargin(0.17);
         gPad->SetRightMargin(0.13);
         gPad->SetBottomMargin(0.16);
         hc->Draw("colz text");
         keep.push_back(hc);
         cv->Print(pdf);
         gStyle->SetPaintTextFormat("g");
      }

      // ---- distributions per true class, 8 variables per page ----
      const int nPage = (nv + 7) / 8;
      for (int pg = 0; pg < nPage; pg++) {
         cv->Clear();
         cv->Divide(4, 2);
         for (int j = 0; j < 8; j++) {
            const int v = pg * 8 + j;
            if (v >= nv) break;
            double lo, hi;
            quantileRange(val[s][v], lo, hi);
            TH1F* h[kNCls + 1];
            for (int k = 0; k <= kNCls; k++) {
               h[k] = 0;
               if (k < kNCls && nCls[k] <= 0) continue;
               const char* tag = k < kNCls ? kQaClsName[k] : "all";
               h[k] = new TH1F(Form("hS%d_%s_%s", set, names[v], tag),
                               Form("set %d: %s;%s;fraction of clusters", set, names[v], names[v]), 50, lo, hi);
               h[k]->SetLineColor(k < kNCls ? kQaClsColor[k] : kBlack);
               h[k]->SetLineWidth(2);
               if (k == kNCls) h[k]->SetLineStyle(2);
               keep.push_back(h[k]);
            }
            for (int i = 0; i < nClu; i++) {
               const float a = val[s][v][i];
               if (bad(a)) continue;
               h[cls[i]]->Fill(a);
               h[kNCls]->Fill(a);
            }
            cv->cd(j + 1);
            gPad->SetLeftMargin(0.15);
            drawOverlay(h, kNCls + 1);
            if (j == 0) {
               TLegend* lg = new TLegend(0.55, 0.60, 0.95, 0.89);
               lg->SetBorderSize(0);
               lg->SetFillStyle(0);
               for (int k = 0; k <= kNCls; k++)
                  if (h[k]) lg->AddEntry(h[k], k < kNCls ? kQaClsName[k] : "all clusters", "l");
               lg->Draw();
            }
         }
         cv->Print(pdf);
      }

      // ---- each variable against cluster energy, all clusters ----
      for (int pg = 0; pg < nPage; pg++) {
         cv->Clear();
         cv->Divide(4, 2);
         for (int j = 0; j < 8; j++) {
            const int v = pg * 8 + j;
            if (v >= nv) break;
            double lo, hi;
            quantileRange(val[s][v], lo, hi);
            TH2F* h2 = new TH2F(Form("hS%d_%s_vsE", set, names[v]),
                                Form("set %d: %s vs E;cluster E [GeV];%s", set, names[v], names[v]), 32, 0, eTop, 40,
                                lo, hi);
            for (int i = 0; i < nClu; i++)
               if (!bad(val[s][v][i])) h2->Fill(eClu[i], val[s][v][i]);
            cv->cd(j + 1);
            gPad->SetLeftMargin(0.15);
            gPad->SetRightMargin(0.13);
            gPad->SetLogz();
            h2->Draw("colz");
            keep.push_back(h2);
         }
         const bool last = (s == nSet - 1) && (pg == nPage - 1);
         cv->Print(last ? (pdf + ")").Data() : pdf.Data());
      }
   }

   fout->cd();
   for (size_t i = 0; i < keep.size(); i++) keep[i]->Write();
   fout->Close();
   printf("\nwrote %s and %s.root\n", pdf.Data(), out.Data());
}
