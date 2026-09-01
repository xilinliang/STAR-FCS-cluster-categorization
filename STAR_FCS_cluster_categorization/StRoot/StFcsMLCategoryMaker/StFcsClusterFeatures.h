// StFcsClusterFeatures.h
//
// THE single definition of the input variables. Both the training macro
// (trainTMVA.C, which has tree branches) and the inference maker
// (StFcsMLCategoryMaker, which has an StFcsCluster) call compute() through the
// same header, so there is no second implementation that can drift out of sync.
//
// Header-only and free of any STAR or ROOT dependency, so a ROOT macro can
// simply #include it.
//
// Two feature sets:
//
//   13 (default) - shape summary variables:
//        logE nTowers sigmaMax sigmaMin sigmaRatio theta
//        seedFrac e2Frac e1e2Asym sigX sigY sigXY nNeighbor
//
//   34 (ePIC-style):
//        e x y nHits radius dispersion sigmaMin sigmaMax
//        t00..t44 (5x5 tower energies around the seed) eOut
//
// author: generated for Xilin Liang

#ifndef STAR_StFcsClusterFeatures_HH
#define STAR_StFcsClusterFeatures_HH

#include <cmath>
#include <cstdlib>

namespace StFcsClusterFeatures {

const int kNVarMax = 34;
const int kNW5 = 5;  // 5x5 tower window used by the 34-variable set

// Tower energies (34-variable set) as fractions of the cluster energy instead
// of raw GeV. Raw energies make the model learn the energy spectrum of the
// training sample, which is the first thing that differs between the fast
// simulator and data.
const bool kTowerFractions = true;

// Logarithmic weight offset for the dispersion variable:
//   w_i = max(0, kW0 + ln(E_i/E))
const double kW0 = 4.5;

inline int nVar(int set) { return (set == 34) ? 34 : 13; }

inline const char** varNames(int set) {
   static const char* n13[13] = {"logE",     "nTowers", "sigmaMax", "sigmaMin", "sigmaRatio",
                                 "theta",    "seedFrac", "e2Frac",  "e1e2Asym", "sigX",
                                 "sigY",     "sigXY",   "nNeighbor"};
   static const char* n34[34] = {"e",   "x",   "y",   "nHits", "radius", "dispersion", "sigmaMin",
                                 "sigmaMax",
                                 "t00", "t01", "t02", "t03", "t04",
                                 "t10", "t11", "t12", "t13", "t14",
                                 "t20", "t21", "t22", "t23", "t24",
                                 "t30", "t31", "t32", "t33", "t34",
                                 "t40", "t41", "t42", "t43", "t44",
                                 "eOut"};
   return (set == 34) ? n34 : n13;
}

// Everything one cluster contributes. Rows and columns are the STAR FCS
// 1-based tower indices; x and y are the cluster centroid in COLUMN and ROW
// units (the StFcsCluster convention, not cm); xw and yw are the cell widths
// in cm and are only used by the 34-variable set.
struct ClusterInput {
   float e;
   float x;
   float y;
   float sigmaMin;
   float sigmaMax;
   float theta;
   int nTowers;
   int nNeighbor;
   float xw;
   float yw;
   int nTow;              // number of towers in the arrays below
   const float* towerE;   // tower energies    [nTow]
   const int* towerRow;   // tower row index   [nTow]
   const int* towerCol;   // tower column index[nTow]
};

// Fills out[0..nVar(set)-1]. Returns the number of variables written, or 0 if
// the cluster is unusable (no towers, no energy).
inline int compute(int set, const ClusterInput& c, float* out) {
   const int nv = nVar(set);
   for (int i = 0; i < nv; i++) out[i] = 0.0;
   if (c.nTow <= 0 || c.e <= 0) return 0;

   if (set != 34) {
      // --------------------------------------------------------------- 13
      double e1 = -1, e2 = -1;
      double wtot = 0, sx = 0, sy = 0, sxy = 0, mx = 0, my = 0;
      for (int k = 0; k < c.nTow; k++) {
         const double he = c.towerE[k];
         if (he <= 0) continue;
         const double col = c.towerCol[k];
         const double row = c.towerRow[k];
         if (he > e1) {
            e2 = e1;
            e1 = he;
         } else if (he > e2) {
            e2 = he;
         }
         wtot += he;
         mx += he * col;
         my += he * row;
         sx += he * col * col;
         sy += he * row * row;
         sxy += he * col * row;
      }
      if (wtot <= 0) return 0;
      mx /= wtot;
      my /= wtot;
      if (e2 < 0) e2 = 0;

      out[0] = log(c.e);
      out[1] = c.nTowers;
      out[2] = c.sigmaMax;
      out[3] = c.sigmaMin;
      out[4] = (c.sigmaMax > 0) ? c.sigmaMin / c.sigmaMax : 0.0;
      out[5] = c.theta;
      out[6] = e1 / c.e;
      out[7] = e2 / c.e;
      out[8] = (e1 + e2 > 0) ? (e1 - e2) / (e1 + e2) : 1.0;
      out[9] = sqrt(fabs(sx / wtot - mx * mx));
      out[10] = sqrt(fabs(sy / wtot - my * my));
      out[11] = sxy / wtot - mx * my;
      out[12] = c.nNeighbor;
      return 13;
   }

   // ------------------------------------------------------------------ 34
   int seedRow = 0, seedCol = 0;
   double emax = -1;
   for (int k = 0; k < c.nTow; k++) {
      if (c.towerE[k] > emax) {
         emax = c.towerE[k];
         seedRow = c.towerRow[k];
         seedCol = c.towerCol[k];
      }
   }

   const int half = kNW5 / 2;
   double wtot = 0, wr2 = 0, lwtot = 0, lwr2 = 0, e5x5 = 0;
   for (int k = 0; k < c.nTow; k++) {
      const double he = c.towerE[k];
      if (he <= 0) continue;
      const double dx = (c.towerCol[k] - c.x) * c.xw;
      const double dy = (c.towerRow[k] - c.y) * c.yw;
      const double d2 = dx * dx + dy * dy;

      wtot += he;
      wr2 += he * d2;

      const double lw = kW0 + log(he / c.e);
      if (lw > 0) {
         lwtot += lw;
         lwr2 += lw * d2;
      }

      const int dr = c.towerRow[k] - seedRow;
      const int dc = c.towerCol[k] - seedCol;
      if (abs(dr) <= half && abs(dc) <= half) {
         out[8 + (dr + half) * kNW5 + (dc + half)] = he;
         e5x5 += he;
      }
   }
   if (wtot <= 0) return 0;

   out[0] = c.e;
   out[1] = c.x;
   out[2] = c.y;
   out[3] = c.nTowers;
   out[4] = sqrt(wr2 / wtot);
   out[5] = (lwtot > 0) ? sqrt(lwr2 / lwtot) : 0.0;
   out[6] = c.sigmaMin;
   out[7] = c.sigmaMax;
   out[33] = c.e - e5x5;
   if (kTowerFractions)
      for (int i = 8; i < 34; i++) out[i] /= c.e;
   return 34;
}

}  // namespace StFcsClusterFeatures

#endif
