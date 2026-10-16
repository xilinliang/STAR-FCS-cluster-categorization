// qaWeights.C - what the flat-in-cluster-energy training weight does to the
// sample the model actually sees.
//
//   root4star -b -q 'qaWeights.C+("feat_pico_mix1.root","qaw_mix1")'
//   root4star -b -q 'qaWeights.C+("feat_pico_mix1.root","qaw_mix1",3,0.5,1)'
//        (infile, out, featureSet, eMin, labelDef, treename, purityCut, eMax)
//
// EVERY distribution is drawn TWICE on the same pad: solid = weighted, dashed
// = unweighted, both over the same clusters. The pair is the point - the gap
// between the two curves IS the weighting.
//
// The weights come from StFcsEnergyWeights.h, the same header trainTMVA.C
// applies, so this macro cannot drift from what was trained. Only the TRAINING
// half is used (the split of StFcsTrainTestSplit.h), because that is the
// sample the weights act on; the test half is never weighted.
//
// Output
//   <out>.pdf
//     page 1   the weight itself against cluster energy, one line per class,
//              with the cap marked
//     page 2   cluster energy per class, weighted over unweighted. The
//              weighted curves are flat by construction - that is the recipe
//              working, and the check that it did
//     page 3   class composition in each energy bin, unweighted then weighted.
//              THE plot: if cluster energy carries class information, the
//              unweighted pad shows it and the weighted pad should not
//     page 4   effective statistics, (sum w)^2 / sum w^2 over the raw count,
//              per class per bin: where the weighting spends its evidence
//     then     every input variable of the set, one page per class, each pad
//              weighted over unweighted at unit area
//   <out>.root  every histogram
//   printed     the weight table, and per class the raw and effective counts
//
// author: generated for Xilin Liang

#include <cmath>
#include <cstdio>
#include <vector>

#include "TCanvas.h"
#include "TFile.h"
#include "TGraph.h"
#include "TH1F.h"
#include "TLatex.h"
#include "TLegend.h"
#include "TLine.h"
#include "TROOT.h"
#include "TString.h"
#include "TStyle.h"
#include "TTree.h"

#include "StRoot/StFcsMLCategoryMaker/StFcsClusterFeatures.h"
#include "StRoot/StFcsMLCategoryMaker/StFcsEnergyWeights.h"
#include "StRoot/StFcsMLCategoryMaker/StFcsTrainTestSplit.h"

namespace {
const int kNC = 3;
const char* kCName[kNC] = {"hadronic (pi-)", "single EM (gamma)", "merged pi0"};
const char* kCShort[kNC] = {"hadronic", "singleEM", "mergedPi0"};
const int kCColor[kNC] = {kGray + 2, kAzure + 1, kOrange + 7};

// solid = weighted, dashed = unweighted, everywhere
void styleW(TH1F* h, int col) {
   h->SetLineColor(col);
   h->SetLineWidth(3);
   h->SetLineStyle(1);
   h->SetStats(0);
}
void styleU(TH1F* h, int col) {
   h->SetLineColor(col);
   h->SetLineWidth(2);
   h->SetLineStyle(2);
   h->SetStats(0);
}
void unitArea(TH1F* h) {
   const double a = h->Integral();
   if (a > 0) h->Scale(1.0 / a);
}
// draw a pair on the current pad with a common vertical range
void drawPair(TH1F* u, TH1F* w) {
   double top = u->GetMaximum();
   if (w->GetMaximum() > top) top = w->GetMaximum();
   u->SetMaximum(1.25 * (top > 0 ? top : 1));
   u->SetMinimum(0);
   u->Draw("hist");
   w->Draw("hist same");
}
}  // namespace

void qaWeights(const char* infile = "feat_pico_mix1.root", const char* outName = "qaWeights",
               int featureSet = 3, float eMin = 0.5, int labelDef = 1,
               const char* treename = "clusters", float purityCut = 0.8, float eMax = 0) {
   using namespace StFcsClusterFeatures;
   namespace W = StFcsEnergyWeights;

   gROOT->SetBatch(kTRUE);
   gStyle->SetOptStat(0);
   gStyle->SetOptTitle(1);
   gStyle->SetPaperSize(28.0, 12.7);

   const TString out = strlen(outName) ? TString(outName) : TString("qaWeights");
   const int NVAR = nVar(featureSet);
   const char** vname = varNames(featureSet);

   TFile* fin = TFile::Open(infile);
   if (!fin || fin->IsZombie()) { printf("cannot open %s\n", infile); return; }
   TTree* in = (TTree*)fin->Get(treename);
   if (!in) { printf("no tree %s in %s\n", treename, infile); return; }

   const int NW = 11;
   Float_t e, x, y, sigmaMin, sigmaMax, theta, xw, yw, truthPurity = 0;
   Float_t img[NW * NW], mask[NW * NW];
   Int_t det = 0, nTowers, nNeighbor = 0, seedRow, seedCol, truthNPhoton = -1, mcLabel = -1;
   Int_t genPid = 0, run = 0, event = 0;
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
   if (in->GetBranch("truthNPhoton")) in->SetBranchAddress("truthNPhoton", &truthNPhoton);
   if (in->GetBranch("truthPurity")) in->SetBranchAddress("truthPurity", &truthPurity);
   if (in->GetBranch("mcLabel")) in->SetBranchAddress("mcLabel", &mcLabel);
   const bool havePid = (in->GetBranch("genPid") != 0);
   if (havePid) in->SetBranchAddress("genPid", &genPid);
   in->SetBranchAddress("run", &run);
   in->SetBranchAddress("event", &event);
   if (labelDef == 1 && !havePid) {
      printf("labelDef=1 needs genPid, which %s does not have - using mcLabel\n", infile);
      labelDef = 0;
   }

   double eTop = eMax;
   if (eTop <= 0) {
      eTop = in->GetMaximum("e");
      eTop = 10.0 * ceil(1.05 * eTop / 10.0);
   }
   const int nEbin = 60;
   printf("cluster-energy axis 0 - %.3g GeV, feature set %d (%d variables)\n", eTop, featureSet, NVAR);

   // ---------------------------------------------------------------- pass 0
   // count the TRAINING half per (class, weight bin), exactly as trainTMVA.C
   double wCount[kNC][W::kNWB];
   for (int c = 0; c < kNC; c++)
      for (int b = 0; b < W::kNWB; b++) wCount[c][b] = 0;

   const int half = NW / 2;
   float te[NW * NW], v[kNVarMax];
   int trow[NW * NW], tcol[NW * NW];
   const Long64_t n = in->GetEntries();

   double wVal[kNC][W::kNWB];
   TH1F* hEu[kNC];
   TH1F* hEw[kNC];
   TH1F* hVu[kNC][kNVarMax];
   TH1F* hVw[kNC][kNVarMax];
   double sumW[kNC][W::kNWB], sumW2[kNC][W::kNWB];
   for (int c = 0; c < kNC; c++) {
      hEu[c] = hEw[c] = 0;
      for (int k = 0; k < kNVarMax; k++) hVu[c][k] = hVw[c][k] = 0;
      for (int b = 0; b < W::kNWB; b++) { wVal[c][b] = 1.0; sumW[c][b] = sumW2[c][b] = 0; }
   }
   // Pass 0 also measures each variable's range. A plain min-max is hostage to
   // one outlier (seedFrac has been seen above 5 on picoDst, see BUILD.md), so
   // the axis stops at mean + 5 rms when that is tighter than the maximum.
   double vN[kNVarMax], vS[kNVarMax], vS2[kNVarMax];
   float vlo[kNVarMax], vhi[kNVarMax];
   for (int k = 0; k < kNVarMax; k++) {
      vN[k] = vS[k] = vS2[k] = 0;
      vlo[k] = 1e30;
      vhi[k] = -1e30;
   }

   for (int pass = 0; pass < 2; pass++) {
      const bool counting = (pass == 0);
      StFcsTrainTestSplit::EventSplitter splitter;

      if (!counting) {
         W::Report rep[kNC];
         W::build(wCount, wVal, rep);
         W::printTable(rep, kCShort);
         for (int c = 0; c < kNC; c++) {
            hEu[c] = new TH1F(Form("hEu_%s", kCShort[c]),
                              Form("cluster energy, %s;E [GeV];clusters", kCName[c]), nEbin, 0, eTop);
            hEw[c] = new TH1F(Form("hEw_%s", kCShort[c]),
                              Form("cluster energy, %s;E [GeV];clusters", kCName[c]), nEbin, 0, eTop);
            for (int k = 0; k < NVAR; k++) {
               double a = vlo[k], b2 = vhi[k];
               if (vN[k] > 1) {
                  const double m = vS[k] / vN[k];
                  const double rms = sqrt(fabs(vS2[k] / vN[k] - m * m));
                  if (rms > 0 && m + 5 * rms < b2) b2 = m + 5 * rms;
                  if (rms > 0 && m - 5 * rms > a) a = m - 5 * rms;
               }
               if (!(a < b2)) { a = 0; b2 = 1; }
               const double pad = 0.05 * (b2 - a);
               hVu[c][k] = new TH1F(Form("hVu_%s_%s", kCShort[c], vname[k]),
                                    Form("%s, %s;%s;unit area", vname[k], kCName[c], vname[k]),
                                    50, a - pad, b2 + pad);
               hVw[c][k] = new TH1F(Form("hVw_%s_%s", kCShort[c], vname[k]),
                                    Form("%s, %s;%s;unit area", vname[k], kCName[c], vname[k]),
                                    50, a - pad, b2 + pad);
            }
         }
      }

      for (Long64_t i = 0; i < n; i++) {
         in->GetEntry(i);
         const int isTest = splitter.isTest(run, event);
         if (isTest) continue;           // the weights act on the training half only
         if (e < eMin) continue;

         int cls = StFcsTrainTestSplit::trainingLabel(mcLabel, truthNPhoton, truthPurity, purityCut);
         if (labelDef == 1 && cls >= 0) cls = StFcsTrainTestSplit::sampleLabel(genPid, mcLabel);
         if (cls < 0 || cls >= kNC) continue;

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
         c.e = e; c.x = x; c.y = y;
         c.sigmaMin = sigmaMin; c.sigmaMax = sigmaMax; c.theta = theta;
         c.nTowers = nTowers; c.nNeighbor = nNeighbor;
         c.xw = xw; c.yw = yw;
         c.nTow = nTow; c.towerE = te; c.towerRow = trow; c.towerCol = tcol;
         if (compute(featureSet, c, v) != NVAR) continue;

         const int eb = W::bin(e);
         if (counting) {
            wCount[cls][eb] += 1;
            for (int k = 0; k < NVAR; k++) {
               if (v[k] < vlo[k]) vlo[k] = v[k];
               if (v[k] > vhi[k]) vhi[k] = v[k];
               vN[k] += 1;
               vS[k] += v[k];
               vS2[k] += (double)v[k] * v[k];
            }
            continue;
         }

         const double wt = wVal[cls][eb];
         hEu[cls]->Fill(e);
         hEw[cls]->Fill(e, wt);
         sumW[cls][eb] += wt;
         sumW2[cls][eb] += wt * wt;
         for (int k = 0; k < NVAR; k++) {
            hVu[cls][k]->Fill(v[k]);
            hVw[cls][k]->Fill(v[k], wt);
         }
      }

      if (counting) continue;

      // ------------------------------------------------------------ output
      TFile* fout = new TFile(out + ".root", "RECREATE");
      const TString pdf = out + ".pdf";
      TCanvas* cv = new TCanvas("cv", "qaWeights", 1100, 500);
      std::vector<TH1*> keep;

      printf("\n  raw and effective statistics of the TRAINING half\n");
      printf("  %-11s %10s %12s %10s\n", "class", "clusters", "effective", "ratio");
      for (int c = 0; c < kNC; c++) {
         double sw = 0, sw2 = 0;
         for (int b = 0; b < W::kNWB; b++) { sw += sumW[c][b]; sw2 += sumW2[c][b]; }
         const double neff = (sw2 > 0) ? sw * sw / sw2 : 0;
         printf("  %-11s %10.0f %12.0f %9.2f\n", kCShort[c], hEu[c]->GetEntries(), neff,
                hEu[c]->GetEntries() > 0 ? neff / hEu[c]->GetEntries() : 0);
      }
      printf("  the ratio is how much independent evidence survives the weighting\n");

      // ---- page 1: the weight against cluster energy ----
      cv->Clear();
      cv->cd();
      gPad->SetLeftMargin(0.10);
      gPad->SetLogy();
      TH1F* fr = new TH1F("frW", "training weight per cluster;cluster energy [GeV];weight",
                          nEbin, 0, eTop);
      fr->SetMinimum(0.1);
      fr->SetMaximum(40);
      fr->SetStats(0);
      fr->Draw();
      keep.push_back(fr);
      TLegend* lg1 = new TLegend(0.68, 0.70, 0.90, 0.89);
      lg1->SetBorderSize(0);
      lg1->SetFillStyle(0);
      for (int c = 0; c < kNC; c++) {
         TH1F* hw = new TH1F(Form("hwStep_%s", kCShort[c]), "", nEbin, 0, eTop);
         for (int bb = 1; bb <= nEbin; bb++) {
            const double ec = hw->GetXaxis()->GetBinCenter(bb);
            hw->SetBinContent(bb, wVal[c][W::bin(ec)]);
         }
         styleW(hw, kCColor[c]);
         hw->Draw("hist same");
         lg1->AddEntry(hw, kCShort[c], "l");
         keep.push_back(hw);
      }
      TLine* one = new TLine(0, 1, eTop, 1);
      one->SetLineStyle(3);
      one->Draw();
      lg1->Draw();
      TLatex nt;
      nt.SetNDC();
      nt.SetTextSize(0.035);
      nt.DrawLatexNDC(0.12, 0.84, "flat weight 1 = unweighted; above it the bin is upweighted");
      cv->Print(pdf + "(");

      // ---- page 2: cluster energy per class, weighted over unweighted ----
      cv->Clear();
      cv->Divide(3, 1);
      for (int c = 0; c < kNC; c++) {
         cv->cd(c + 1);
         gPad->SetLeftMargin(0.15);
         styleU(hEu[c], kCColor[c]);
         styleW(hEw[c], kCColor[c]);
         drawPair(hEu[c], hEw[c]);
         TLegend* lg = new TLegend(0.45, 0.76, 0.90, 0.89);
         lg->SetBorderSize(0);
         lg->SetFillStyle(0);
         lg->AddEntry(hEu[c], "unweighted", "l");
         lg->AddEntry(hEw[c], "weighted", "l");
         lg->Draw();
         keep.push_back(hEu[c]);
         keep.push_back(hEw[c]);
      }
      cv->Print(pdf);

      // ---- page 3: class composition per energy bin ----
      cv->Clear();
      cv->Divide(2, 1);
      const char* ttl[2] = {"class share per energy bin, unweighted;cluster energy [GeV];share",
                            "class share per energy bin, weighted;cluster energy [GeV];share"};
      for (int pane = 0; pane < 2; pane++) {
         cv->cd(pane + 1);
         gPad->SetLeftMargin(0.13);
         TH1F* frf = new TH1F(Form("frFrac%d", pane), ttl[pane], W::kNWB, 0, (double)W::kNWB);
         frf->SetMinimum(0);
         frf->SetMaximum(1.15);
         frf->SetStats(0);
         for (int b = 0; b < W::kNWB; b++)
            frf->GetXaxis()->SetBinLabel(b + 1, Form("%g", W::kEdge[b]));
         frf->GetXaxis()->SetTitle("cluster energy bin, lower edge [GeV]");
         frf->Draw();
         keep.push_back(frf);
         TLegend* lg = new TLegend(0.62, 0.72, 0.90, 0.89);
         lg->SetBorderSize(0);
         lg->SetFillStyle(0);
         for (int c = 0; c < kNC; c++) {
            TH1F* hf = new TH1F(Form("hFrac%d_%s", pane, kCShort[c]), "", W::kNWB, 0, (double)W::kNWB);
            for (int b = 0; b < W::kNWB; b++) {
               double tot = 0, mine = 0;
               for (int q = 0; q < kNC; q++) {
                  const double val = (pane == 0) ? wCount[q][b] : sumW[q][b];
                  tot += val;
                  if (q == c) mine = val;
               }
               hf->SetBinContent(b + 1, tot > 0 ? mine / tot : 0);
            }
            styleW(hf, kCColor[c]);
            if (pane == 0) hf->SetLineStyle(2);
            hf->Draw("hist same");
            lg->AddEntry(hf, kCShort[c], "l");
            keep.push_back(hf);
         }
         TLine* third = new TLine(0, 1.0 / 3, (double)W::kNWB, 1.0 / 3);
         third->SetLineStyle(3);
         third->Draw();
         lg->Draw();
      }
      cv->Print(pdf);

      // ---- page 4: effective statistics per bin ----
      cv->Clear();
      cv->cd();
      gPad->SetLeftMargin(0.10);
      TH1F* fre = new TH1F("frEff", "effective clusters / raw clusters, per energy bin;"
                                    "cluster energy bin, lower edge [GeV];N_{eff} / N",
                           W::kNWB, 0, (double)W::kNWB);
      fre->SetMinimum(0);
      fre->SetMaximum(1.3);
      fre->SetStats(0);
      for (int b = 0; b < W::kNWB; b++) fre->GetXaxis()->SetBinLabel(b + 1, Form("%g", W::kEdge[b]));
      fre->Draw();
      keep.push_back(fre);
      TLegend* lg4 = new TLegend(0.68, 0.72, 0.90, 0.89);
      lg4->SetBorderSize(0);
      lg4->SetFillStyle(0);
      for (int c = 0; c < kNC; c++) {
         TH1F* he = new TH1F(Form("hEff_%s", kCShort[c]), "", W::kNWB, 0, (double)W::kNWB);
         for (int b = 0; b < W::kNWB; b++) {
            const double neff = (sumW2[c][b] > 0) ? sumW[c][b] * sumW[c][b] / sumW2[c][b] : 0;
            he->SetBinContent(b + 1, wCount[c][b] > 0 ? neff / wCount[c][b] : 0);
         }
         styleW(he, kCColor[c]);
         he->Draw("hist same");
         lg4->AddEntry(he, kCShort[c], "l");
         keep.push_back(he);
      }
      lg4->Draw();
      nt.DrawLatexNDC(0.12, 0.84, "1 = the weighting costs nothing in that bin");
      cv->Print(pdf);

      // ---- the input variables, one page per class ----
      for (int c = 0; c < kNC; c++) {
         const int nPad = NVAR;
         int nx = 5, ny = (nPad + 4) / 5;
         if (nPad <= 8) { nx = 4; ny = 2; }
         cv->Clear();
         cv->Divide(nx, ny);
         for (int k = 0; k < nPad; k++) {
            cv->cd(k + 1);
            gPad->SetLeftMargin(0.16);
            unitArea(hVu[c][k]);
            unitArea(hVw[c][k]);
            styleU(hVu[c][k], kCColor[c]);
            styleW(hVw[c][k], kCColor[c]);
            drawPair(hVu[c][k], hVw[c][k]);
            keep.push_back(hVu[c][k]);
            keep.push_back(hVw[c][k]);
            if (k == 0) {
               TLegend* lg = new TLegend(0.40, 0.74, 0.92, 0.89);
               lg->SetBorderSize(0);
               lg->SetFillStyle(0);
               lg->AddEntry(hVu[c][k], "unweighted", "l");
               lg->AddEntry(hVw[c][k], "weighted", "l");
               lg->Draw();
            }
         }
         cv->Print(c == kNC - 1 ? pdf + ")" : pdf);
      }

      for (size_t q = 0; q < keep.size(); q++) keep[q]->Write();
      fout->Close();
      printf("\nwrote %s.pdf and %s.root\n", out.Data(), out.Data());
   }
}
