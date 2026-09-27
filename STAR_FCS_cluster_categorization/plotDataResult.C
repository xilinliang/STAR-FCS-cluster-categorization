// plotDataResult.C - what the model changed, on one sample.
//
// Reads the file StFcsPicoCategoryMaker writes (runPicoDst_ml.C mode 1) and
// draws the two things worth looking at first on real data:
//
//   page 1  the cluster-pair mass with STAR's category and with the model's,
//           overlaid, plus the category migration matrix - which clusters
//           changed class, and in which direction
//   page 2  the three model scores, the ECal cluster multiplicity, the energy
//           asymmetry and the opening angle
//
// printed  the migration matrix as fractions, and for each mass spectrum the
//          yield in a pi0 window and in two sidebands, with a crude
//          signal-over-background - enough to see whether the model helped or
//          hurt, not a substitute for a fit
//
//   root4star -b -q 'plotDataResult.C+("cat_st_physics_....root","dataResult")'
//
// NOTE. The pair mass here is built from cluster pairs, not from fitted
// points: it is a quick diagnostic that travels with the category maker, not
// the pi0 analysis. Run your own pi0 finder for the physics number; what this
// macro is for is deciding whether the ML category is worth putting into it.
//
// author: generated for Xilin Liang

#include <cmath>
#include <cstdio>
#include <cstring>

#include "TCanvas.h"
#include "TFile.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TLatex.h"
#include "TLegend.h"
#include "TROOT.h"
#include "TString.h"
#include "TStyle.h"

namespace {

const char* kStarName[3] = {"0 ambiguous", "1 onePhoton", "2 twoPhoton"};
const char* kMlName[3] = {"hadronic", "single EM", "merged #pi^{0}"};

// counts in [lo,hi] of a mass histogram
double window(TH1F* h, double lo, double hi) {
   if (!h) return 0;
   double s = 0;
   for (int b = 1; b <= h->GetNbinsX(); b++) {
      const double c = h->GetXaxis()->GetBinLowEdge(b);
      if (c >= lo && c < hi) s += h->GetBinContent(b);
   }
   return s;
}

void massReport(const char* name, TH1F* h) {
   if (!h) {
      printf("  %-14s missing\n", name);
      return;
   }
   // pi0 window and two sidebands either side of it
   const double sig = window(h, 0.10, 0.17);
   const double lo = window(h, 0.04, 0.08);
   const double hi = window(h, 0.20, 0.28);
   const double bkg = lo + hi;
   printf("  %-14s pairs %8.0f   0.10-0.17 %8.0f   sidebands %8.0f   S/B ~ %5.2f\n", name,
          h->Integral(), sig, bkg, bkg > 0 ? sig / bkg : 0.0);
}

}  // namespace

void plotDataResult(const char* infile = "fcsPicoCategory.root", const char* outName = "dataResult") {
   gROOT->SetBatch(kTRUE);
   gStyle->SetOptStat(0);
   gStyle->SetPaintTextFormat("4.2f");
   gStyle->SetPaperSize(28.0, 12.7);

   TFile* f = TFile::Open(infile);
   if (!f || f->IsZombie()) {
      printf("cannot open %s\n", infile);
      return;
   }
   TH1F* mStar = (TH1F*)f->Get("h1_invmass_star");
   TH1F* mMl = (TH1F*)f->Get("h1_invmass_ml");
   TH2F* mig = (TH2F*)f->Get("h2_catStar_vs_catML");
   TH1F* prob[3];
   for (int k = 0; k < 3; k++) prob[k] = (TH1F*)f->Get(Form("h1_prob%d", k));
   TH1F* nclu = (TH1F*)f->Get("h1_nCluEcal");
   TH1F* zgg = (TH1F*)f->Get("h1_zgg");
   TH1F* dgg = (TH1F*)f->Get("h1_dgg");

   // ----------------------------------------------------------------- print
   printf("\n=== %s ===\n", infile);
   printf("\n  --- cluster-pair mass ---\n");
   massReport("FCS Cluster", mStar);
   massReport("ML category", mMl);

   if (mig) {
      printf("\n  --- category migration: rows = FCS Cluster, columns = model ---\n");
      printf("  %-14s", "");
      for (int k = 0; k < 3; k++) printf(" %14s", kMlName[k]);
      printf(" %10s\n", "clusters");
      for (int i = 0; i < 3; i++) {
         double rs = 0;
         for (int j = 0; j < 3; j++) rs += mig->GetBinContent(i + 1, j + 1);
         printf("  %-14s", kStarName[i]);
         for (int j = 0; j < 3; j++)
            printf(" %14.3f", rs > 0 ? mig->GetBinContent(i + 1, j + 1) / rs : 0.0);
         printf(" %10.0f\n", rs);
      }
      printf("  the interesting entries are STAR '1 onePhoton' -> model 'merged pi0'\n"
             "  (pi0s the standard chain would have fitted as one photon) and\n"
             "  STAR '1 onePhoton' -> model 'hadronic' (hadron contamination removed)\n");
   }

   // ----------------------------------------------------------------- plots
   const TString pdf = TString(outName) + ".pdf";
   TCanvas* cv = new TCanvas("cv", "dataResult", 1100, 500);

   // page 1: mass overlay and migration
   cv->Divide(2, 1);
   cv->cd(1);
   gPad->SetLeftMargin(0.14);
   if (mStar && mMl) {
      mStar->SetLineColor(kGray + 2);
      mStar->SetLineWidth(2);
      mMl->SetLineColor(kOrange + 7);
      mMl->SetLineWidth(2);
      const double ymax = (mStar->GetMaximum() > mMl->GetMaximum() ? mStar->GetMaximum() : mMl->GetMaximum());
      mStar->SetMaximum(1.25 * ymax);
      mStar->SetMinimum(0);
      mStar->Draw("hist");
      mMl->Draw("hist same");
      TLegend* lg = new TLegend(0.52, 0.72, 0.93, 0.88);
      lg->SetBorderSize(0);
      lg->SetFillStyle(0);
      lg->AddEntry(mStar, "FCS Cluster category", "l");
      lg->AddEntry(mMl, "ML category", "l");
      lg->Draw();
   } else {
      TLatex t;
      t.SetNDC();
      t.SetTextSize(0.05);
      t.DrawLatexNDC(0.15, 0.5, "no mass histograms in this file");
   }
   cv->cd(2);
   gPad->SetLeftMargin(0.17);
   gPad->SetRightMargin(0.14);
   if (mig) {
      TH2F* h = (TH2F*)mig->Clone("h2_mig_norm");
      for (int i = 0; i < 3; i++) {
         double rs = 0;
         for (int j = 0; j < 3; j++) rs += mig->GetBinContent(i + 1, j + 1);
         for (int j = 0; j < 3; j++)
            h->SetBinContent(i + 1, j + 1, rs > 0 ? mig->GetBinContent(i + 1, j + 1) / rs : 0.0);
         h->GetXaxis()->SetBinLabel(i + 1, kStarName[i]);
         h->GetYaxis()->SetBinLabel(i + 1, kMlName[i]);
      }
      h->SetTitle("fraction of each FCS Cluster category;FCS Cluster;model");
      h->SetMinimum(0);
      h->SetMaximum(1);
      h->SetMarkerSize(1.8);
      h->Draw("colz text");
   }
   cv->Print((pdf + "(").Data());

   // page 2: scores and event-level distributions
   cv->Clear();
   cv->Divide(3, 2);
   TH1* pads[6] = {prob[1], prob[0], prob[2], nclu, zgg, dgg};
   const char* ttl[6] = {"P(single EM)", "P(hadronic)", "P(merged #pi^{0})",
                         "ECal clusters per event", "Z_{#gamma#gamma}", "opening angle"};
   for (int p = 0; p < 6; p++) {
      cv->cd(p + 1);
      gPad->SetLeftMargin(0.15);
      if (!pads[p]) continue;
      if (p < 3) gPad->SetLogy();
      pads[p]->SetLineWidth(2);
      pads[p]->SetLineColor(p < 3 ? kAzure + 1 : kGray + 2);
      pads[p]->SetTitle(ttl[p]);
      pads[p]->Draw("hist");
   }
   cv->Print((pdf + ")").Data());
   printf("\nwrote %s\n", pdf.Data());
}
