// compareModels.C - put two to four trained models side by side, from the
// .root files evalCategory.C already wrote. No re-evaluation, no weight files:
// it reads the TEfficiency objects and overlays them.
//
//   root4star -b -q 'compareModels.C+("evalFcsCat13_BDTG.root","BDTG set 13",
//                     "evalFcsCat13_MLP.root","MLP set 13",
//                     "evalFcsCat3_BDTG.root","BDTG set 3")'
//   (one line in the shell; broken up here only for reading)
//
// Compile with ACLiC (the trailing '+'), like the other macros here.
//
// Output
//   <out>.pdf
//     page 1  efficiency vs energy, three panels in the ePIC order
//             (Single EM, Hadronic, Merged pi0), one marker per model
//     page 2  the same for purity
//     page 3  merged-pi0 efficiency vs photon separation
//   printed
//     per model and class, the efficiency and purity summed over all bins,
//     then the merged-pi0 efficiency bin by bin - the comparison that decides
//     which model to freeze
//
// The FCS Cluster category is taken from the FIRST file and drawn with open
// markers on every panel, so each comparison still shows what STAR's own
// category does on the same clusters. Pass withRef = 0 to leave it out.
//
// The files must come from evaluations of the SAME feature file with the same
// truthDef, or the comparison is meaningless; the macro cannot check that, so
// generate them in one go:
//   for m in BDTG MLP; do for s in 3 13; do
//     root4star -b -q "evalCategory.C+(\"feat_all.root\",\"weights/FcsCat${s}gen_${m}.weights.xml\",$s,\"$m\",1,\"\",0.5,0.8,\"clusters\",1,1)"
//   done; done
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
#include "TLatex.h"
#include "TLegend.h"
#include "TPad.h"
#include "TROOT.h"
#include "TString.h"
#include "TStyle.h"

namespace {

const int kNM = 4;  // at most four models on one page
const char* kClsName[3] = {"other", "onePhoton", "twoPhoton"};
// the ePIC panel order and names: Single EM = onePhoton, Hadronic = other,
// Merged pi0 = twoPhoton
const int kPanelCls[3] = {1, 0, 2};
const char* kPanelName[3] = {"Single EM", "Hadronic", "Merged #pi^{0}"};
const int kMColor[kNM] = {kBlack, kRed + 1, kBlue + 1, kGreen + 2};
const int kMMarker[kNM] = {20, 21, 22, 33};

TEfficiency* get(TFile* f, const char* name) {
   if (!f) return 0;
   return (TEfficiency*)f->Get(name);
}

// passed and total summed over every bin: the efficiency (or purity) of the
// whole sample, with a binomial error
void integrate(TEfficiency* e, double& pass, double& tot) {
   pass = tot = 0;
   if (!e) return;
   const TH1* hp = e->GetPassedHistogram();
   const TH1* ht = e->GetTotalHistogram();
   if (!hp || !ht) return;
   for (int b = 1; b <= ht->GetNbinsX(); b++) {
      pass += hp->GetBinContent(b);
      tot += ht->GetBinContent(b);
   }
}

double binomErr(double pass, double total) {
   if (total <= 0) return 0;
   const double p = pass / total;
   return sqrt(p * (1 - p) / total);
}

// TEfficiency::Draw cannot be overlaid in ROOT 5.34 - draw its graph instead
TGraphAsymmErrors* drawEff(TEfficiency* e, int color, int marker, double shift = 0) {
   if (!e) return 0;
   TGraphAsymmErrors* g = e->CreateGraph();
   if (!g) return 0;
   g->SetLineColor(color);
   g->SetMarkerColor(color);
   g->SetMarkerStyle(marker);
   g->SetMarkerSize(1.1);
   (void)shift;
   g->Draw("P");
   return g;
}

}  // namespace

void compareModels(const char* f1 = "evalFcsCat13_BDTG.root", const char* lab1 = "BDTG set 13",
                   const char* f2 = "evalFcsCat13_MLP.root", const char* lab2 = "MLP set 13",
                   const char* f3 = "", const char* lab3 = "",
                   const char* f4 = "", const char* lab4 = "",
                   const char* outName = "modelCompare", int withRef = 1) {
   gROOT->SetBatch(kTRUE);
   gStyle->SetOptStat(0);
   gStyle->SetPaperSize(28.0, 12.7);

   const char* fname[kNM] = {f1, f2, f3, f4};
   const char* flab[kNM] = {lab1, lab2, lab3, lab4};
   TFile* fin[kNM];
   TString lab[kNM];
   int nM = 0;
   for (int m = 0; m < kNM; m++) {
      if (!fname[m] || !strlen(fname[m])) continue;
      TFile* f = TFile::Open(fname[m]);
      if (!f || f->IsZombie()) {
         printf("cannot open %s - skipped\n", fname[m]);
         continue;
      }
      fin[nM] = f;
      lab[nM] = (flab[m] && strlen(flab[m])) ? flab[m] : fname[m];
      nM++;
   }
   if (nM == 0) {
      printf("no input files could be opened\n");
      return;
   }

   // ------------------------------------------------------------- printed
   printf("\n=== %d models, all bins summed ===\n", nM);
   printf("  %-16s %20s %20s %20s\n", "model", "Single EM", "Hadronic", "Merged pi0");
   for (int what = 0; what < 2; what++) {  // 0 efficiency, 1 purity
      printf("\n  --- %s ---\n", what == 0 ? "efficiency" : "purity");
      for (int m = 0; m < nM; m++) {
         printf("  %-16s", lab[m].Data());
         for (int j = 0; j < 3; j++) {
            const int k = kPanelCls[j];
            TEfficiency* e = get(fin[m], Form("%sU_ml_%s", what == 0 ? "eff" : "pur", kClsName[k]));
            double p, t;
            integrate(e, p, t);
            if (t > 0)
               printf("   %.3f +- %.3f (%5.0f)", p / t, binomErr(p, t), t);
            else
               printf("   %20s", "-");
         }
         printf("\n");
      }
      if (withRef) {
         printf("  %-16s", "FCS Cluster");
         for (int j = 0; j < 3; j++) {
            const int k = kPanelCls[j];
            TEfficiency* e = get(fin[0], Form("%sU_star_%s", what == 0 ? "eff" : "pur", kClsName[k]));
            double p, t;
            integrate(e, p, t);
            if (t > 0)
               printf("   %.3f +- %.3f (%5.0f)", p / t, binomErr(p, t), t);
            else
               printf("   %20s", "no class");
         }
         printf("\n");
      }
   }

   // merged pi0 versus separation, bin by bin: the number that matters most
   printf("\n  --- merged pi0 efficiency vs photon separation [towers] ---\n");
   {
      TEfficiency* ref = get(fin[0], "effSep_ml");
      const TH1* ht = ref ? ref->GetTotalHistogram() : 0;
      if (!ht) {
         printf("  (no effSep_ml in %s)\n", fname[0]);
      } else {
         printf("  %-14s", "separation");
         for (int m = 0; m < nM; m++) printf(" %16s", lab[m].Data());
         if (withRef) printf(" %16s", "FCS Cluster");
         printf(" %9s\n", "clusters");
         for (int b = 1; b <= ht->GetNbinsX(); b++) {
            const double lo = ht->GetXaxis()->GetBinLowEdge(b);
            const double hi = ht->GetXaxis()->GetBinUpEdge(b);
            if (ht->GetBinContent(b) <= 0) continue;
            printf("  %5.2f - %-6.2f", lo, hi);
            for (int m = 0; m < nM; m++) {
               TEfficiency* e = get(fin[m], "effSep_ml");
               const TH1* hp = e ? e->GetPassedHistogram() : 0;
               const TH1* tt = e ? e->GetTotalHistogram() : 0;
               if (hp && tt && tt->GetBinContent(b) > 0)
                  printf(" %8.3f +-%.3f", hp->GetBinContent(b) / tt->GetBinContent(b),
                         binomErr(hp->GetBinContent(b), tt->GetBinContent(b)));
               else
                  printf(" %16s", "-");
            }
            if (withRef) {
               TEfficiency* e = get(fin[0], "effSep_star");
               const TH1* hp = e ? e->GetPassedHistogram() : 0;
               const TH1* tt = e ? e->GetTotalHistogram() : 0;
               if (hp && tt && tt->GetBinContent(b) > 0)
                  printf(" %8.3f +-%.3f", hp->GetBinContent(b) / tt->GetBinContent(b),
                         binomErr(hp->GetBinContent(b), tt->GetBinContent(b)));
               else
                  printf(" %16s", "-");
            }
            printf(" %9.0f\n", ht->GetBinContent(b));
         }
         printf("  a bin with few clusters carries a large error - read the last column\n");
      }
   }

   // --------------------------------------------------------------- plots
   const TString out(outName);
   const TString pdf = out + ".pdf";
   TCanvas* cv = new TCanvas("cv", "compareModels", 1100, 500);
   TLatex tx;
   tx.SetNDC();

   for (int what = 0; what < 2; what++) {
      cv->Clear();
      cv->cd();
      tx.SetTextAlign(22);
      tx.SetTextFont(62);
      tx.SetTextSize(0.050);
      tx.DrawLatexNDC(0.5, 0.955, what == 0 ? "Efficiency vs energy" : "Purity vs energy");
      TPad* body = new TPad(Form("body_%d", what), "", 0.0, 0.0, 1.0, 0.90);
      body->SetFillStyle(0);
      body->Draw();
      body->Divide(3, 1);
      for (int j = 0; j < 3; j++) {
         const int k = kPanelCls[j];
         body->cd(j + 1);
         gPad->SetTopMargin(0.22);
         gPad->SetLeftMargin(0.15);
         gPad->SetRightMargin(0.04);
         gPad->SetBottomMargin(0.13);
         gPad->SetGridx();
         gPad->SetGridy();
         // the axis range comes from the first model's own histogram, so this
         // macro follows whatever range evalCategory.C used
         double xmax = 32;
         {
            TEfficiency* e = get(fin[0], Form("%sU_ml_%s", what == 0 ? "eff" : "pur", kClsName[k]));
            const TH1* ht = e ? e->GetTotalHistogram() : 0;
            if (ht) xmax = ht->GetXaxis()->GetBinUpEdge(ht->GetNbinsX());
         }
         TH1F* fr = gPad->DrawFrame(0, 0, xmax, 1.1,
                                    Form(";energy [GeV];%s", what == 0 ? "Efficiency" : "Purity"));
         (void)fr;
         TLegend* lg = new TLegend(0.15, 0.79, 0.97, 0.90);
         lg->SetBorderSize(0);
         lg->SetFillStyle(0);
         lg->SetTextSize(0.042);
         lg->SetNColumns(2);
         for (int m = 0; m < nM; m++) {
            TEfficiency* e = get(fin[m], Form("%sU_ml_%s", what == 0 ? "eff" : "pur", kClsName[k]));
            TGraphAsymmErrors* g = drawEff(e, kMColor[m], kMMarker[m]);
            if (g) lg->AddEntry(g, lab[m].Data(), "lp");
         }
         if (withRef) {
            TEfficiency* e = get(fin[0], Form("%sU_star_%s", what == 0 ? "eff" : "pur", kClsName[k]));
            TGraphAsymmErrors* g = drawEff(e, kGray + 2, 24);
            if (g) lg->AddEntry(g, "FCS Cluster", "lp");
         }
         lg->Draw();
         tx.SetTextAlign(12);
         tx.SetTextFont(62);
         tx.SetTextSize(0.055);
         tx.DrawLatexNDC(0.15, 0.955, kPanelName[j]);
      }
      cv->Print(what == 0 ? (pdf + "(").Data() : pdf.Data());
   }

   // page 3: merged pi0 versus photon separation
   cv->Clear();
   cv->cd();
   gPad->SetGridy();
   gPad->DrawFrame(0, 0, 4, 1.1,
                   "Merged #pi^{0} clusters: fraction identified as two-photon;"
                   "separation of the two photons at the ECal [towers];efficiency");
   TLegend* ls = new TLegend(0.55, 0.15, 0.90, 0.40);
   ls->SetBorderSize(0);
   ls->SetFillStyle(0);
   for (int m = 0; m < nM; m++) {
      TGraphAsymmErrors* g = drawEff(get(fin[m], "effSep_ml"), kMColor[m], kMMarker[m]);
      if (g) ls->AddEntry(g, lab[m].Data(), "lp");
   }
   if (withRef) {
      TGraphAsymmErrors* g = drawEff(get(fin[0], "effSep_star"), kGray + 2, 24);
      if (g) ls->AddEntry(g, "FCS Cluster (category 2)", "lp");
   }
   ls->Draw();
   cv->Print((pdf + ")").Data());

   printf("\nwrote %s\n", pdf.Data());
}
