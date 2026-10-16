// StFcsEnergyWeights.h
//
// THE single definition of the flat-in-cluster-energy training weight, so
// trainTMVA.C (which applies it) and qaWeights.C (which draws what it did)
// cannot drift apart. Same reason StFcsClusterFeatures.h exists for the input
// variables and StFcsTrainTestSplit.h for the labels.
//
// WHAT THE WEIGHT IS. A weight says how many times the training counts a
// cluster: a BDT sums weights in every node and boosting residual, an MLP
// multiplies each event's gradient by its weight. Evaluation never uses them,
// so efficiency and purity stay physical.
//
// WHAT THIS ONE DOES. Every populated cluster-energy bin of a class is given
// the same total weight, so each class becomes flat in cluster energy with
// mean weight 1. Where all three classes populate a bin that also equalises
// them inside it, which is the point: it stops cluster energy from carrying
// class information.
//
// WHAT IT CANNOT DO. A weight multiplies, so a bin a class does not populate
// stays empty however it is weighted. Flattening a class over a range it only
// half covers extrapolates from the few clusters at the sparse end - see
// BUILD.md for what that did to the hadronic class above 30 GeV.
//
// Header-only, no ROOT, no STAR.
//
// author: generated for Xilin Liang

#ifndef STAR_StFcsEnergyWeights_HH
#define STAR_StFcsEnergyWeights_HH

#include <cstdio>
#include <algorithm>
#include <vector>

namespace StFcsEnergyWeights {

const int kNCls = 3;
const int kNWB = 14;  // cluster-energy bins
const double kEdge[kNWB + 1] = {0.0, 1, 2, 3, 5,  7,  10, 15,
                                20,  25, 30, 40, 50, 60, 1e9};
const int kMinBin = 50;     // fewer clusters of a class than this: merge up
const double kWCap = 10.0;  // cap a weight at this times the class median

// Which bin a cluster energy falls in. Same walk trainTMVA.C used inline.
inline int bin(double e) {
   int b = 0;
   while (b < kNWB - 1 && e >= kEdge[b + 1]) b++;
   return b;
}

// What the recipe did to one class, for the printed table and for QA.
struct Report {
   double total;   // clusters of this class
   int nUsed;      // populated bins after merging
   double lo, hi;  // smallest and largest weight actually used
   int nMerged;    // bins folded into a neighbour
   int nCap;       // bins clipped at kWCap times the median
   Report() : total(0), nUsed(0), lo(0), hi(0), nMerged(0), nCap(0) {}
};

// counts[class][bin] in, w[class][bin] out (1.0 where a class has no clusters).
inline void build(const double counts[kNCls][kNWB], double w[kNCls][kNWB], Report rep[kNCls]) {
   for (int c = 0; c < kNCls; c++)
      for (int b = 0; b < kNWB; b++) w[c][b] = 1.0;

   for (int c = 0; c < kNCls; c++) {
      Report& r = rep[c];
      r = Report();

      // merge a bin with too few clusters into the next one up, so a handful
      // of clusters cannot become a huge weight
      double nEff[kNWB];
      int owner[kNWB];  // which bin a cluster of this bin is counted in
      for (int b = 0; b < kNWB; b++) {
         nEff[b] = counts[c][b];
         owner[b] = b;
      }
      for (int b = 0; b < kNWB - 1; b++) {
         if (nEff[b] <= 0 || nEff[b] >= kMinBin) continue;
         nEff[b + 1] += nEff[b];
         nEff[b] = 0;
         for (int q = 0; q <= b; q++)
            if (owner[q] == b) owner[q] = b + 1;
         r.nMerged++;
      }
      // the last bin cannot merge upward: fold it downward instead
      if (nEff[kNWB - 1] > 0 && nEff[kNWB - 1] < kMinBin) {
         for (int b = kNWB - 2; b >= 0; b--) {
            if (nEff[b] <= 0) continue;
            nEff[b] += nEff[kNWB - 1];
            nEff[kNWB - 1] = 0;
            for (int q = 0; q < kNWB; q++)
               if (owner[q] == kNWB - 1) owner[q] = b;
            r.nMerged++;
            break;
         }
      }

      for (int b = 0; b < kNWB; b++) {
         r.total += nEff[b];
         if (nEff[b] > 0) r.nUsed++;
      }
      if (r.total <= 0 || r.nUsed == 0) continue;

      // equal total weight in every populated bin, mean weight 1
      std::vector<double> ws;
      for (int b = 0; b < kNWB; b++) {
         const double nb = nEff[owner[b]];
         w[c][b] = (nb > 0) ? (r.total / r.nUsed) / nb : 1.0;
         if (counts[c][b] > 0) ws.push_back(w[c][b]);
      }
      std::sort(ws.begin(), ws.end());
      const double med = ws.empty() ? 1.0 : ws[ws.size() / 2];
      double lo = 1e30, hi = 0;
      for (int b = 0; b < kNWB; b++) {
         if (counts[c][b] <= 0) continue;
         if (w[c][b] > kWCap * med) {
            w[c][b] = kWCap * med;
            r.nCap++;
         }
         if (w[c][b] < lo) lo = w[c][b];
         if (w[c][b] > hi) hi = w[c][b];
      }

      // renormalise so the average weight of this class is 1, which keeps
      // TMVA's own NormMode doing what it did before
      double sw = 0;
      for (int b = 0; b < kNWB; b++) sw += counts[c][b] * w[c][b];
      const double scale = (sw > 0) ? (r.total / sw) : 1.0;
      for (int b = 0; b < kNWB; b++) w[c][b] *= scale;
      r.lo = lo * scale;
      r.hi = hi * scale;
   }
}

inline void printTable(const Report rep[kNCls], const char* const* clsname) {
   printf("\n--- flat-in-energy training weights (bins of CLUSTER energy) ---\n");
   printf("  %-11s %8s %9s %9s %9s %7s %7s\n", "class", "clusters", "bins used", "min w", "max w",
          "merged", "capped");
   for (int c = 0; c < kNCls; c++) {
      if (rep[c].total <= 0 || rep[c].nUsed == 0) continue;
      printf("  %-11s %8.0f %9d %9.3g %9.3g %7d %7d\n", clsname[c], rep[c].total, rep[c].nUsed,
             rep[c].lo, rep[c].hi, rep[c].nMerged, rep[c].nCap);
   }
   printf("  a class whose clusters all sit in one energy bin gets weight 1 everywhere\n");
}

}  // namespace StFcsEnergyWeights

#endif
