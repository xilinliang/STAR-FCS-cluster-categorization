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
const int kNW3 = 3;  // 3x3 tower window used by set 3

// Tower energies (sets 3 and 34) as fractions of the cluster energy instead of
// raw GeV. Raw energies make the model learn the energy spectrum of the
// training sample, which is the first thing that differs between the fast
// simulator and data.
//
// NOTE for set 3: with fractions on and the window centred on the seed tower,
// the central cell t11 IS seedFrac, exactly. The two inputs are the same
// number. That is harmless for a BDT but it wastes an input and it makes the
// TMVA variable ranking misleading. If you would rather spend that slot on new
// information, swap seedFrac for the energy outside the 3x3 - see the comment
// at out[3] in compute().
const bool kTowerFractions = true;

// Logarithmic weight offset for the dispersion variable:
//   w_i = max(0, kW0 + ln(E_i/E))
const double kW0 = 4.5;

// Set ids are NOT variable counts - set 3 means "the 3x3 set" and has 13
// variables, the same count as set 13 but different content.
inline int nVar(int set) {
   if (set == 34) return 34;
   if (set == 6) return 6;
   if (set == 3) return 13;  // 9 tower energies + e + sigmaMax + sigmaMin + seedFrac
   return 13;
}

inline const char** varNames(int set) {
   // Set 6 is the first six of set 13, same names and same definitions on
   // purpose: it is the subset that survives into StPicoDst, where the
   // cluster's tower list is not stored. Train on MuDst with set 6 and the
   // model applies unchanged to picoDst input.
   static const char* n6[6] = {"logE", "nTowers", "sigmaMax", "sigmaMin", "sigmaRatio", "theta"};
   // set 3: the raw 3x3 shower shape plus the four scalars that the 3x3 alone
   // cannot supply (absolute scale, and the rotation-invariant widths)
   static const char* n3[13] = {"e",   "sigmaMax", "sigmaMin", "seedFrac",
                                "t00", "t01", "t02",
                                "t10", "t11", "t12",
                                "t20", "t21", "t22"};
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
   if (set == 34) return n34;
   if (set == 6) return n6;
   if (set == 3) return n3;
   return n13;
}

// Everything one cluster contributes. Rows and columns are the STAR FCS
// 1-based tower indices; x and y are the cluster centroid in COLUMN and ROW
// units (the StFcsCluster convention, not cm); xw and yw are the cell widths
// in cm and are only used by the 34-variable set.
//
// Set 6 needs only e, sigmaMin, sigmaMax, theta and nTowers - no tower arrays.
// That is what makes it usable on StPicoDst, which stores the cluster summary
// but not the list of towers that went into it. Leave nTow at 0 there.
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
   if (c.e <= 0) return 0;
   if (set != 6 && c.nTow <= 0) return 0;  // set 6 needs no tower list

   if (set == 6) {
      // ---------------------------------------------------------------- 6
      // Identical definitions to the first six of set 13. Everything here is
      // available from StFcsCluster, StMuFcsCluster and StPicoFcsCluster
      // alike, so one model covers MuDst and picoDst input.
      out[0] = log(c.e);
      out[1] = c.nTowers;
      out[2] = c.sigmaMax;
      out[3] = c.sigmaMin;
      out[4] = (c.sigmaMax > 0) ? c.sigmaMin / c.sigmaMax : 0.0;
      out[5] = c.theta;
      return 6;
   }

   if (set == 3) {
      // ----------------------------------------------------------------- 3
      //   0  e         cluster energy [GeV] - the only absolute scale here
      //   1  sigmaMax  major-axis width  (rotation invariant, so it sees a
      //   2  sigmaMin  minor-axis width   split the 3x3 cannot express)
      //   3  seedFrac  e1 / E
      //   4..12  t00..t22  3x3 tower energies around the SEED tower, row major,
      //          (dr,dc) from (-1,-1) to (+1,+1), cluster towers only,
      //          divided by E when kTowerFractions
      //
      // To spend slot 3 on something the 3x3 does not already contain, replace
      // the seedFrac line below with the energy outside the window:
      //     out[3] = (c.e - e3x3) / c.e;   // and rename the variable to eOut
      // With kTowerFractions on, seedFrac and t11 are the same number.
      int seedRow = 0, seedCol = 0;
      double e1 = -1;
      for (int k = 0; k < c.nTow; k++) {
         if (c.towerE[k] > e1) {
            e1 = c.towerE[k];
            seedRow = c.towerRow[k];
            seedCol = c.towerCol[k];
         }
      }
      if (e1 <= 0) return 0;

      const int half3 = kNW3 / 2;
      for (int k = 0; k < c.nTow; k++) {
         const double he = c.towerE[k];
         if (he <= 0) continue;
         const int dr = c.towerRow[k] - seedRow;
         const int dc = c.towerCol[k] - seedCol;
         if (abs(dr) > half3 || abs(dc) > half3) continue;
         out[4 + (dr + half3) * kNW3 + (dc + half3)] = he;
      }

      out[0] = c.e;
      out[1] = c.sigmaMax;
      out[2] = c.sigmaMin;
      out[3] = e1 / c.e;
      if (kTowerFractions)
         for (int i = 4; i < 13; i++) out[i] /= c.e;
      return 13;
   }

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
