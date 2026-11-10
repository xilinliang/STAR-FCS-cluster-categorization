// qaSampleLabel.C - what goes into each training class under the SAMPLE
// definition (StFcsSampleLabel.h: trainTMVA.C labelDef=2, evalCategory.C
// truthDef=3).
//
//   root4star -b -q 'qaSampleLabel.C+("feat_pico_80GeV.root","qsmp_80GeV")'
//        (infile, out, eMin, sample, treename, eMax)
//
// The sample definition RANKS the clusters of an event and keeps only some of
// them, so two questions matter before any model is trained on it: what is in
// each class, and what was thrown away to get there. This macro answers both
// from the same pass.
//
//   gamma event  the leading cluster            -> 1EM
//   pi- event    every cluster                  -> Hadron
//   pi0 event    merged (mcLabel >= 2): that cluster alone -> MergedPi0
//                resolved: the two leading clusters       -> 1EM
//   everything else above eMin in those events  -> dropped, trained on by
//                                                  nothing
//
// Output
//   <out>.pdf
//     page 1  ECal cluster energy per class: three on one log axis, then one
//             pad each. THE plot - it is the spectrum each class is trained
//             on, and where two classes overlap is where the model has to
//             work from shape alone
//     page 2  the same three normalised to unit area, so the shapes compare
//             without the class sizes in the way
//     page 3  what each gun contributed: kept against dropped, in cluster
//             energy, one pad per gun. A dropped population that looks like
//             a kept one is a warning - it means the ranking is cutting into
//             real objects rather than junk
//     page 4  class share in each cluster-energy bin: the class mix the model
//             sees against energy, which is what sets purity bin by bin
//     page 5  generated (gun) energy per class, for the same clusters
//   <out>.root  every histogram
//
// Only the TRAINING half is used by default (sample=0), because that is what
// labelDef=2 trains on. sample=1 is the held-out half, 2 is everything.
//
// author: generated for Xilin Liang

#include <cmath>
#include <cstdio>
#include <vector>

#include "TCanvas.h"
#include "TFile.h"
#include "TH1F.h"
#include "TLatex.h"
#include "TLegend.h"
#include "TLine.h"
#include "TROOT.h"
#include "TString.h"
#include "TStyle.h"
#include "TTree.h"

#include "StRoot/StFcsMLCategoryMaker/StFcsSampleLabel.h"
#include "StRoot/StFcsMLCategoryMaker/StFcsTrainTestSplit.h"

namespace {

const int kNC = 3;
const char* kCName[kNC] = {"Hadron (pi-)", "1EM (gamma / resolved pi0)", "MergedPi0"};
const char* kCShort[kNC] = {"Hadron", "1EM", "MergedPi0"};
const int kCColor[kNC] = {kGray + 2, kAzure + 2, kOrange + 7};

const int kNG = 3;
const char* kGName[kNG] = {"gamma gun", "pi0 gun", "pi- gun"};
const char* kGShort[kNG] = {"gamma", "pi0", "pim"};
const int kGColor[kNG] = {kAzure + 2, kOrange + 7, kGray + 2};

void style(TH1F* h, int col, int width = 3, int lstyle = 1) {
   h->SetLineColor(col);
   h->SetLineWidth(width);
   h->SetLineStyle(lstyle);
   h->SetStats(0);
}

// a log pad needs a positive minimum, and an empty histogram must not blank it
void drawLog(TH1F* h, const char* opt) {
   double mx = h->GetMaximum();
   if (mx <= 0) mx = 1;
   h->SetMaximum(mx * 3);
   h->SetMinimum(0.5);
   h->Draw(opt);
}

}  // namespace

void qaSampleLabel(const char* infile = "feat_pico_all.root", const char* outName = "qaSample",
                   float eMin = 0.5, int sample = 0, const char* treename = "clusters",
                   float eMax = 0) {
   gROOT->SetBatch(kTRUE);
   gStyle->SetOptStat(0);
   gStyle->SetPalette(1);

   TFile* fin = TFile::Open(infile, "READ");
   if (!fin || fin->IsZombie()) {
      printf("cannot open %s\n", infile);
      return;
   }
   TTree* in = (TTree*)fin->Get(treename);
   if (!in) {
      printf("no tree %s in %s\n", treename, infile);
      return;
   }

   // ------------------------------------------------- the sample definition
   std::vector<signed char> lab;
   StFcsSampleLabel::Report srep;
   if (StFcsSampleLabel::build(infile, treename, lab, eMin, &srep) < 0) {
      printf("%s has no genPid/mcLabel - the sample definition needs both.\n"
             "Re-dump it with the current feature maker.\n", infile);
      return;
   }
   StFcsSampleLabel::printReport(srep);

   Float_t e = 0, genE = -1;
   Int_t run = 0, event = 0, genPid = 0, mcLabel = -1;
   in->SetBranchAddress("run", &run);
   in->SetBranchAddress("event", &event);
   in->SetBranchAddress("e", &e);
   const bool haveGen = (in->GetBranch("genE") != 0);
   if (haveGen) in->SetBranchAddress("genE", &genE);
   if (in->GetBranch("genPid")) in->SetBranchAddress("genPid", &genPid);
   if (in->GetBranch("mcLabel")) in->SetBranchAddress("mcLabel", &mcLabel);

   double eTop = eMax;
   if (eTop <= 0) {
      eTop = in->GetMaximum("e");
      eTop = 10.0 * ceil(1.05 * eTop / 10.0);
   }
   if (eTop <= 0) eTop = 10;
   const int nEbin = 60;
   const char* sampTxt[3] = {"TRAINING half", "test half (held out)", "every cluster"};
   if (sample < 0 || sample > 2) sample = 0;
   printf("\n  cluster-energy axis 0 - %.3g GeV, %s\n", eTop, sampTxt[sample]);

   // ------------------------------------------------------------ histograms
   TH1F *hE[kNC], *hEn[kNC], *hGen[kNC];
   for (int c = 0; c < kNC; c++) {
      hE[c] = new TH1F(Form("hE_%s", kCShort[c]),
                       Form("cluster energy, %s;ECal cluster E [GeV];clusters", kCName[c]),
                       nEbin, 0, eTop);
      hEn[c] = new TH1F(Form("hEn_%s", kCShort[c]),
                        Form("cluster energy, %s;ECal cluster E [GeV];unit area", kCName[c]),
                        nEbin, 0, eTop);
      hGen[c] = new TH1F(Form("hGen_%s", kCShort[c]),
                         Form("generated energy, %s;E_{gen} [GeV];clusters", kCName[c]),
                         nEbin, 0, eTop);
      style(hE[c], kCColor[c]);
      style(hEn[c], kCColor[c]);
      style(hGen[c], kCColor[c]);
   }
   // per gun: what was kept, and what the ranking dropped
   TH1F *hKept[kNG], *hDrop[kNG];
   for (int g = 0; g < kNG; g++) {
      hKept[g] = new TH1F(Form("hKept_%s", kGShort[g]),
                          Form("%s;ECal cluster E [GeV];clusters", kGName[g]), nEbin, 0, eTop);
      hDrop[g] = new TH1F(Form("hDrop_%s", kGShort[g]),
                          Form("%s;ECal cluster E [GeV];clusters", kGName[g]), nEbin, 0, eTop);
      style(hKept[g], kGColor[g], 3, 1);
      style(hDrop[g], kGColor[g], 2, 2);
   }
   // class share per energy bin
   TH1F* hShare[kNC];
   for (int c = 0; c < kNC; c++) {
      hShare[c] = new TH1F(Form("hShare_%s", kCShort[c]),
                           "class share of the clusters in each energy bin;"
                           "ECal cluster E [GeV];share of the bin",
                           nEbin, 0, eTop);
      style(hShare[c], kCColor[c]);
   }

   // ------------------------------------------------------------ one pass
   StFcsTrainTestSplit::EventSplitter splitter;
   const Long64_t n = in->GetEntries();
   Long64_t nUsed = 0, nDropSeen = 0;
   for (Long64_t i = 0; i < n; i++) {
      in->GetEntry(i);
      // before any cut, or the split differs from the one trainTMVA.C used
      const int isTest = splitter.isTest(run, event);
      if (sample == 0 && isTest) continue;
      if (sample == 1 && !isTest) continue;
      if (e < eMin) continue;

      const int g = (genPid == 1) ? 0 : (genPid == 7) ? 1 : (genPid == 9) ? 2 : -1;
      const int c = (i < (Long64_t)lab.size()) ? (int)lab[(size_t)i] : -1;

      if (c >= 0 && c < kNC) {
         hE[c]->Fill(e);
         hEn[c]->Fill(e);
         if (haveGen) hGen[c]->Fill(genE);
         if (g >= 0) hKept[g]->Fill(e);
         nUsed++;
      } else {
         if (g >= 0) hDrop[g]->Fill(e);
         nDropSeen++;
      }
   }
   printf("  %lld clusters in a class, %lld above eMin but dropped by the ranking\n",
          nUsed, nDropSeen);

   // class share per bin
   for (int b = 1; b <= nEbin; b++) {
      double sum = 0;
      for (int c = 0; c < kNC; c++) sum += hE[c]->GetBinContent(b);
      for (int c = 0; c < kNC; c++)
         hShare[c]->SetBinContent(b, sum > 0 ? hE[c]->GetBinContent(b) / sum : 0.0);
   }
   for (int c = 0; c < kNC; c++)
      if (hEn[c]->Integral() > 0) hEn[c]->Scale(1.0 / hEn[c]->Integral());

   // ------------------------------------------------------------ the pages
   const TString pdf = TString(outName) + ".pdf";
   TCanvas* cv = new TCanvas("cv", "", 1400, 900);
   TLatex tx;
   tx.SetNDC();

   // page 1: cluster energy per class
   cv->Clear();
   cv->Divide(2, 2);
   cv->cd(1);
   gPad->SetLogy(1);
   {
      TLegend* lg = new TLegend(0.45, 0.70, 0.93, 0.88);
      lg->SetBorderSize(0);
      lg->SetFillStyle(0);
      double mx = 0;
      for (int c = 0; c < kNC; c++)
         if (hE[c]->GetMaximum() > mx) mx = hE[c]->GetMaximum();
      for (int c = 0; c < kNC; c++) {
         hE[c]->SetMaximum(mx > 0 ? mx * 3 : 1);
         hE[c]->SetMinimum(0.5);
         hE[c]->SetTitle("cluster energy, all three classes;ECal cluster E [GeV];clusters");
         hE[c]->Draw(c == 0 ? "hist" : "hist same");
         lg->AddEntry(hE[c], Form("%s (%.0f)", kCShort[c], hE[c]->GetEntries()), "l");
      }
      lg->Draw();
   }
   for (int c = 0; c < kNC; c++) {
      cv->cd(c + 2);
      gPad->SetLogy(1);
      drawLog(hE[c], "hist");
   }
   cv->cd();
   tx.SetTextSize(0.022);
   tx.DrawLatexNDC(0.01, 0.985, Form("%s, %s, eMin = %.2f GeV - sample definition "
                                     "(labelDef=2 / truthDef=3)", infile, sampTxt[sample], eMin));
   cv->Print(pdf + "(");

   // page 2: the same, unit area
   cv->Clear();
   cv->cd();
   gPad->SetLogy(1);
   {
      TLegend* lg = new TLegend(0.55, 0.72, 0.93, 0.89);
      lg->SetBorderSize(0);
      lg->SetFillStyle(0);
      double mx = 0;
      for (int c = 0; c < kNC; c++)
         if (hEn[c]->GetMaximum() > mx) mx = hEn[c]->GetMaximum();
      for (int c = 0; c < kNC; c++) {
         hEn[c]->SetMaximum(mx > 0 ? mx * 5 : 1);
         hEn[c]->SetMinimum(mx > 0 ? mx * 1e-4 : 1e-4);
         hEn[c]->SetTitle("cluster energy, unit area - the SHAPES;"
                          "ECal cluster E [GeV];unit area");
         hEn[c]->Draw(c == 0 ? "hist" : "hist same");
         lg->AddEntry(hEn[c], kCShort[c], "l");
      }
      lg->Draw();
   }
   cv->Print(pdf);

   // page 3: kept against dropped, per gun
   cv->Clear();
   cv->Divide(2, 2);
   for (int g = 0; g < kNG; g++) {
      cv->cd(g + 1);
      gPad->SetLogy(1);
      double mx = hKept[g]->GetMaximum();
      if (hDrop[g]->GetMaximum() > mx) mx = hDrop[g]->GetMaximum();
      hKept[g]->SetMaximum(mx > 0 ? mx * 3 : 1);
      hKept[g]->SetMinimum(0.5);
      hKept[g]->Draw("hist");
      hDrop[g]->Draw("hist same");
      TLegend* lg = new TLegend(0.42, 0.72, 0.93, 0.88);
      lg->SetBorderSize(0);
      lg->SetFillStyle(0);
      lg->AddEntry(hKept[g], Form("kept  (%.0f)", hKept[g]->GetEntries()), "l");
      lg->AddEntry(hDrop[g], Form("dropped by the ranking  (%.0f)", hDrop[g]->GetEntries()), "l");
      lg->Draw();
   }
   cv->cd(4);
   tx.SetTextSize(0.035);
   tx.SetTextAlign(12);
   tx.DrawLatexNDC(0.06, 0.88, "What the ranking throws away");
   tx.SetTextSize(0.028);
   tx.DrawLatexNDC(0.06, 0.78, "gamma: all but the leading cluster");
   tx.DrawLatexNDC(0.06, 0.70, "pi0, merged: everything but the merged cluster");
   tx.DrawLatexNDC(0.06, 0.62, "pi0, resolved: all but the two leading");
   tx.DrawLatexNDC(0.06, 0.54, "pi-: nothing - every cluster is kept");
   tx.DrawLatexNDC(0.06, 0.40, "A dropped curve that tracks the kept one in");
   tx.DrawLatexNDC(0.06, 0.34, "shape means the cut is reaching real objects,");
   tx.DrawLatexNDC(0.06, 0.28, "not just fragments.");
   cv->Print(pdf);

   // page 4: class share per energy bin
   cv->Clear();
   cv->cd();
   gPad->SetGridy();
   {
      TH1F* fr = (TH1F*)gPad->DrawFrame(0, 0, eTop, 1.15,
                                        "class share of the clusters in each energy bin;"
                                        "ECal cluster E [GeV];share of the bin");
      (void)fr;
      TLegend* lg = new TLegend(0.40, 0.80, 0.93, 0.93);
      lg->SetBorderSize(0);
      lg->SetFillStyle(0);
      lg->SetNColumns(3);
      for (int c = 0; c < kNC; c++) {
         hShare[c]->Draw("hist same");
         lg->AddEntry(hShare[c], kCShort[c], "l");
      }
      lg->Draw();
      TLine* one = new TLine(0, 1, eTop, 1);
      one->SetLineStyle(2);
      one->Draw();
   }
   cv->Print(pdf);

   // page 5: generated energy per class
   cv->Clear();
   if (haveGen) {
      cv->Divide(2, 2);
      cv->cd(1);
      gPad->SetLogy(1);
      {
         TLegend* lg = new TLegend(0.45, 0.70, 0.93, 0.88);
         lg->SetBorderSize(0);
         lg->SetFillStyle(0);
         double mx = 0;
         for (int c = 0; c < kNC; c++)
            if (hGen[c]->GetMaximum() > mx) mx = hGen[c]->GetMaximum();
         for (int c = 0; c < kNC; c++) {
            hGen[c]->SetMaximum(mx > 0 ? mx * 3 : 1);
            hGen[c]->SetMinimum(0.5);
            hGen[c]->SetTitle("generated (gun) energy, all three classes;"
                              "E_{gen} [GeV];clusters");
            hGen[c]->Draw(c == 0 ? "hist" : "hist same");
            lg->AddEntry(hGen[c], kCShort[c], "l");
         }
         lg->Draw();
      }
      for (int c = 0; c < kNC; c++) {
         cv->cd(c + 2);
         gPad->SetLogy(1);
         drawLog(hGen[c], "hist");
      }
   } else {
      cv->cd();
      tx.SetTextAlign(22);
      tx.SetTextSize(0.035);
      tx.DrawLatexNDC(0.5, 0.5, "no genE branch - re-dump for the generated-energy page");
   }
   cv->Print(pdf + ")");

   // ------------------------------------------------------------ the root file
   TFile* fout = new TFile(TString(outName) + ".root", "RECREATE");
   for (int c = 0; c < kNC; c++) {
      hE[c]->Write();
      hEn[c]->Write();
      hGen[c]->Write();
      hShare[c]->Write();
   }
   for (int g = 0; g < kNG; g++) {
      hKept[g]->Write();
      hDrop[g]->Write();
   }
   fout->Close();

   printf("\n  wrote %s and %s.root\n", pdf.Data(), outName);
}
