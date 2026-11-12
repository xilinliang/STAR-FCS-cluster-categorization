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
//     page 6  IS THE MERGED CLASS REALLY MERGED? Containment E_clu/E_gen per
//             class, then containment and photon separation for the MergedPi0
//             class alone, sliced in cluster energy. A cluster holding BOTH
//             photons must carry nearly the whole pi0, so its containment sits
//             near the single-photon value and its separation is small. A
//             "merged" cluster with containment ~0.2 is one lobe of a resolved
//             pair that the 11 cm match radius labelled twice - see BUILD.md
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
   Float_t mcSepCell = -1;
   in->SetBranchAddress("run", &run);
   in->SetBranchAddress("event", &event);
   in->SetBranchAddress("e", &e);
   const bool haveGen = (in->GetBranch("genE") != 0);
   if (haveGen) in->SetBranchAddress("genE", &genE);
   if (in->GetBranch("genPid")) in->SetBranchAddress("genPid", &genPid);
   if (in->GetBranch("mcLabel")) in->SetBranchAddress("mcLabel", &mcLabel);
   const bool haveSep = (in->GetBranch("mcSepCell") != 0);
   if (haveSep) in->SetBranchAddress("mcSepCell", &mcSepCell);

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
   // --- the merged-class diagnostic ---------------------------------------
   // containment per class, and for MergedPi0 the containment and the photon
   // separation sliced in cluster energy
   const int kNS = 3;
   const double kSlice[kNS + 1] = {0, 10, 30, 1e9};
   const char* kSliceTxt[kNS] = {"E_{clu} < 10 GeV", "10 - 30 GeV", "> 30 GeV"};
   const int kSliceCol[kNS] = {kRed + 1, kGreen + 2, kAzure + 2};
   TH1F* hCont[kNC];
   TH1F* hContM[kNS];
   TH1F* hSepM[kNS];
   for (int c = 0; c < kNC; c++) {
      hCont[c] = new TH1F(Form("hCont_%s", kCShort[c]),
                          "containment;E_{clu} / E_{gen};unit area", 60, 0, 1.2);
      style(hCont[c], kCColor[c]);
   }
   for (int k = 0; k < kNS; k++) {
      hContM[k] = new TH1F(Form("hContM_%d", k),
                           "MergedPi0 containment;E_{clu} / E_{gen};unit area", 60, 0, 1.2);
      hSepM[k] = new TH1F(Form("hSepM_%d", k),
                          "MergedPi0 photon separation;separation [towers];unit area", 50, 0, 5);
      style(hContM[k], kSliceCol[k]);
      style(hSepM[k], kSliceCol[k]);
   }
   double sumC[kNS], sumS[kNS], nSl[kNS];
   for (int k = 0; k < kNS; k++) sumC[k] = sumS[k] = nSl[k] = 0;

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
         if (haveGen && genE > 0) hCont[c]->Fill(e / genE);
         if (g >= 0) hKept[g]->Fill(e);
         if (c == 2) {  // MergedPi0, sliced in cluster energy
            int k = 0;
            while (k < kNS - 1 && e >= kSlice[k + 1]) k++;
            if (haveGen && genE > 0) {
               hContM[k]->Fill(e / genE);
               sumC[k] += e / genE;
            }
            if (haveSep && mcSepCell >= 0) {
               hSepM[k]->Fill(mcSepCell);
               sumS[k] += mcSepCell;
            }
            nSl[k] += 1;
         }
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
   for (int c = 0; c < kNC; c++) {
      if (hEn[c]->Integral() > 0) hEn[c]->Scale(1.0 / hEn[c]->Integral());
      if (hCont[c]->Integral() > 0) hCont[c]->Scale(1.0 / hCont[c]->Integral());
   }
   for (int k = 0; k < kNS; k++) {
      if (hContM[k]->Integral() > 0) hContM[k]->Scale(1.0 / hContM[k]->Integral());
      if (hSepM[k]->Integral() > 0) hSepM[k]->Scale(1.0 / hSepM[k]->Integral());
   }
   printf("\n  IS THE MERGED CLASS REALLY MERGED?\n");
   printf("    a cluster holding BOTH photons carries nearly the whole pi0, so its\n"
          "    containment should sit near the single-photon value, with a small separation\n");
   printf("    %-16s %10s %14s %14s\n", "MergedPi0 slice", "clusters", "<E_clu/E_gen>",
          "<separation>");
   for (int k = 0; k < kNS; k++)
      printf("    %-16s %10.0f %14.3f %14.2f\n", kSliceTxt[k], nSl[k],
             nSl[k] > 0 ? sumC[k] / nSl[k] : 0.0, nSl[k] > 0 ? sumS[k] / nSl[k] : 0.0);
   printf("    (compare with the 1EM containment on page 6 - a merged cluster that\n"
          "     contains far LESS of its pi0 than a photon does of its gamma is one lobe\n"
          "     of a resolved pair, mislabelled by the 11 cm match radius)\n");

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
      // clones, or setting the overlay's title and range here would follow the
      // originals onto the per-class pads below
      for (int c = 0; c < kNC; c++) {
         TH1F* o = (TH1F*)hE[c]->Clone(Form("hEall_%s", kCShort[c]));
         o->SetDirectory(0);
         o->SetMaximum(mx > 0 ? mx * 3 : 1);
         o->SetMinimum(0.5);
         o->SetTitle("cluster energy, all three classes;ECal cluster E [GeV];clusters");
         o->Draw(c == 0 ? "hist" : "hist same");
         lg->AddEntry(o, Form("%s (%.0f)", kCShort[c], hE[c]->GetEntries()), "l");
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
   gPad->SetLogx(0);
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
      TLegend* lg = new TLegend(0.50, 0.74, 0.90, 0.88);
      lg->SetBorderSize(0);
      lg->SetFillStyle(0);
      lg->SetTextSize(0.035);
      lg->AddEntry(hKept[g], Form("kept (%.0f)", hKept[g]->GetEntries()), "l");
      lg->AddEntry(hDrop[g], Form("dropped (%.0f)", hDrop[g]->GetEntries()), "l");
      lg->Draw();
   }
   cv->cd(4);
   gPad->SetLogy(0);
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
   // page 2 set log y on the CANVAS, not on a subpad, so it is still set here.
   // DrawFrame with ymin = 0 on a log axis renders an empty sliver - exactly
   // the bug qaWeights.C page 4 had. Every linear pad states its scale.
   gPad->SetLogy(0);
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
            TH1F* o = (TH1F*)hGen[c]->Clone(Form("hGenAll_%s", kCShort[c]));
            o->SetDirectory(0);
            o->SetMaximum(mx > 0 ? mx * 3 : 1);
            o->SetMinimum(0.5);
            o->SetTitle("generated (gun) energy, all three classes;"
                        "E_{gen} [GeV];clusters");
            o->Draw(c == 0 ? "hist" : "hist same");
            lg->AddEntry(o, Form("%s (%.0f)", kCShort[c], hGen[c]->GetEntries()), "l");
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
      gPad->SetLogy(0);
      tx.SetTextAlign(22);
      tx.SetTextSize(0.035);
      tx.DrawLatexNDC(0.5, 0.5, "no genE branch - re-dump for the generated-energy page");
   }
   cv->Print(pdf);

   // page 6: is the merged class really merged?
   cv->Clear();
   cv->Divide(2, 2);

   cv->cd(1);
   gPad->SetLogy(0);
   {
      double mx = 0;
      for (int c = 0; c < kNC; c++)
         if (hCont[c]->GetMaximum() > mx) mx = hCont[c]->GetMaximum();
      TLegend* lg = new TLegend(0.15, 0.70, 0.55, 0.88);
      lg->SetBorderSize(0);
      lg->SetFillStyle(0);
      for (int c = 0; c < kNC; c++) {
         hCont[c]->SetMaximum(mx > 0 ? mx * 1.3 : 1);
         hCont[c]->SetTitle("containment, per class;E_{clu} / E_{gen};unit area");
         hCont[c]->Draw(c == 0 ? "hist" : "hist same");
         lg->AddEntry(hCont[c], kCShort[c], "l");
      }
      lg->Draw();
   }

   cv->cd(2);
   gPad->SetLogy(0);
   {
      double mx = 0;
      for (int k = 0; k < kNS; k++)
         if (hContM[k]->GetMaximum() > mx) mx = hContM[k]->GetMaximum();
      TLegend* lg = new TLegend(0.15, 0.68, 0.55, 0.88);
      lg->SetBorderSize(0);
      lg->SetFillStyle(0);
      for (int k = 0; k < kNS; k++) {
         hContM[k]->SetMaximum(mx > 0 ? mx * 1.3 : 1);
         hContM[k]->SetTitle("MergedPi0 containment, by cluster energy;"
                             "E_{clu} / E_{gen};unit area");
         hContM[k]->Draw(k == 0 ? "hist" : "hist same");
         lg->AddEntry(hContM[k], kSliceTxt[k], "l");
      }
      lg->Draw();
   }

   cv->cd(3);
   gPad->SetLogy(0);
   {
      double mx = 0;
      for (int k = 0; k < kNS; k++)
         if (hSepM[k]->GetMaximum() > mx) mx = hSepM[k]->GetMaximum();
      TLegend* lg = new TLegend(0.50, 0.68, 0.90, 0.88);
      lg->SetBorderSize(0);
      lg->SetFillStyle(0);
      for (int k = 0; k < kNS; k++) {
         hSepM[k]->SetMaximum(mx > 0 ? mx * 1.3 : 1);
         hSepM[k]->SetTitle("MergedPi0 photon separation, by cluster energy;"
                            "separation of the two photons [towers];unit area");
         hSepM[k]->Draw(k == 0 ? "hist" : "hist same");
         lg->AddEntry(hSepM[k], kSliceTxt[k], "l");
      }
      lg->Draw();
      if (!haveSep) {
         tx.SetTextAlign(22);
         tx.SetTextSize(0.04);
         tx.DrawLatexNDC(0.5, 0.5, "no mcSepCell branch");
      }
   }

   cv->cd(4);
   gPad->SetLogy(0);
   tx.SetTextAlign(12);
   tx.SetTextSize(0.040);
   tx.DrawLatexNDC(0.05, 0.90, "Is the merged class really merged?");
   tx.SetTextSize(0.030);
   tx.DrawLatexNDC(0.05, 0.80, "A cluster holding BOTH photons carries nearly");
   tx.DrawLatexNDC(0.05, 0.74, "the whole pi0, so its containment should sit");
   tx.DrawLatexNDC(0.05, 0.68, "near the 1EM value and its separation be small.");
   tx.DrawLatexNDC(0.05, 0.56, "A MergedPi0 slice whose containment peaks far");
   tx.DrawLatexNDC(0.05, 0.50, "BELOW the 1EM peak is one lobe of a RESOLVED");
   tx.DrawLatexNDC(0.05, 0.44, "pair: the 11 cm match radius put both photons");
   tx.DrawLatexNDC(0.05, 0.38, "inside 2 towers of each centroid, so both");
   tx.DrawLatexNDC(0.05, 0.32, "clusters got mcLabel >= 2.");
   tx.DrawLatexNDC(0.05, 0.20, "Expect that in the low cluster-energy slice,");
   tx.DrawLatexNDC(0.05, 0.14, "where a pi0 opens up and cannot truly merge.");
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
   for (int c = 0; c < kNC; c++) hCont[c]->Write();
   for (int k = 0; k < kNS; k++) {
      hContM[k]->Write();
      hSepM[k]->Write();
   }
   fout->Close();

   printf("\n  wrote %s and %s.root\n", pdf.Data(), outName);
}
