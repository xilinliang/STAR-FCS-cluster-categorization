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
//     page 2        sigmaMax vs energy with STAR's category boundaries drawn on
//     set 3 pages   every variable, split by true class (unit area), then
//                   every variable against cluster energy (all clusters)
//     set 13 pages  the same for set 13
//   <out>.root  every histogram
//   printed     per set, a table per variable: entries, non-finite values,
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
   for (Long64_t i = 0; i < n; i++) {
      in->GetEntry(i);
      nRead++;
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
